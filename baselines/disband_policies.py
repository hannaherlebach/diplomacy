"""Hard-coded policies for disbanding units, to be combined with SD network policies to create policies for WD."""
import os
import numpy as np
from functools import partial
import logging
import argparse
from typing import Any, Sequence, Tuple

from network import config, network_policy, parameter_provider
from environment import (
    diplomacy_state,
    game_runner,
    mila_actions,
    action_utils,
    province_order,
    human_readable_actions,
)
from environment import observation_utils as utils

# from diplomacy.engine.map import Map
# welfare_map = Map('standard_welfare')  # Errors on first repo use

adjacency_matrix = province_order.build_adjacency(province_order.get_mdf_content(province_order.MapMDF.STANDARD_MAP)) # a num_provinces by num_provinces array of 0s and 1s indicating province adjacency


class InstantDisbandPolicy:
    """Policy which disbands immediately."""

    def __init__(self):
        self.turn_num = 0
        self.disbanded = False

    def reset(self):
        pass

    def actions(self, slots_list, observation, legal_actions):
        board = observation.board
        season = observation.season

        units = board[:, 3:10]
        dislodgeds = board[:, 16:23]

        actions = [set() for _ in slots_list]

        if not self.disbanded:
            if season == utils.Season.BUILDS:
                for i, power_ix in enumerate(slots_list):
                    units_power = units[:, power_ix]
                    dislodgeds_power = dislodgeds[:, power_ix]
                    unit_area_ids = np.where(units_power == 1)[0]
                    disldoged_area_ids = np.where(dislodgeds_power == 1)[0]
                    all_area_ids = np.concatenate((unit_area_ids, disldoged_area_ids))

                    for area_id in all_area_ids:
                        # Construct disband action for unit
                        province_id, area_ix = utils.province_id_and_area_index(area_id)
                        coast_flag = 1 if area_ix > 0 else 0
                        province_tuple = (province_id, coast_flag)

                        disband_action = action_utils.construct_action(
                            8, province_tuple, None, None
                        )

                        # Add disband action to actions with probability p
                        actions[i].add(disband_action)

                self.disbanded = True
            # In all other phases, hold
        actions = [list(action_set) for action_set in actions]

        return [
            actions,
            {"values": None, "policy": None, "actions": None},
        ]  # fill out this shit later


class RandomDisbandPolicy:
    """Agent which disbands units randomly in BUILD phases, and holds otherwise."""

    def __init__(self, p=0.5):
        self.p = p

    def reset(self):
        pass

    def actions(self, slots_list, observation, legal_actions):
        """Produces a list of lists of actions, one for each slot.

        Args:
            slots_list: a list of slots (power indices) this policy should produce actions for.
        """

        board = observation.board
        season = observation.season

        units = board[:, 3:10]
        dislodgeds = board[:, 16:23]

        actions = [[] for _ in slots_list]

        if season == utils.Season.BUILDS:
            for i, power_ix in enumerate(slots_list):
                units_power = units[:, power_ix]
                dislodgeds_power = dislodgeds[:, power_ix]
                unit_area_ids = np.where(units_power == 1)[0]
                disldoged_area_ids = np.where(dislodgeds_power == 1)[0]
                all_area_ids = np.concatenate((unit_area_ids, disldoged_area_ids))

                for area_id in all_area_ids:
                    # Construct disband action for unit
                    province_id, area_ix = utils.province_id_and_area_index(area_id)
                    coast_flag = 1 if area_ix > 0 else 0
                    province_tuple = (province_id, coast_flag)

                    disband_action = action_utils.construct_action(
                        8, province_tuple, None, None
                    )

                    # Add disband action to actions with probability p
                    if np.random.uniform() < self.p:
                        actions[i].append(disband_action)

        # In all other phases, hold

        return [
            actions,
            {"values": None, "policy": None, "actions": None},
        ]  # fill out this shit later


# Unfinished
class SmartDisbandPolicy:
    """Agent which disbands the units with fewest adjacent (enemy) units first.

    Args:
        num_to_disband: Number of units to disband per BUILDS phase."""

    def __init__(self, num_to_disband=1):
        self.num_to_disband = num_to_disband

    def reset(self):
        pass

    def actions(self, slots_list, observation, legal_actions):
        board = observation.board
        season = observation.season

        actions = [[] for _ in slots_list]

        if season == utils.Season.BUILDS:
            for i, power_ix in enumerate(slots_list):
                sorted_units = sort_units_by_adjacency(power_ix, board)
                to_disband = min(self.num_to_disband, len(sorted_units))
                units_to_disband = sorted_units[:to_disband]
                for unit in units_to_disband:
                    province_id, area_ix = utils.province_id_and_area_index(unit)
                    coast_flag = 1 if area_ix > 0 else 0
                    province_tuple = (province_id, coast_flag)
                    action = action_utils.construct_action(8, province_tuple, None, None)
                    actions[i].append(action)

        # In all other phases, hold

        return [
            actions,
            {"values": None, "policy": None, "actions": None},
        ]  # fill out this shit later


def get_adjacent_provinces(province: utils.ProvinceID, adjacency: np.array):
    """Returns a list of ProvinceIDs of adjacent provinces for the given province.

    Args:
        province: integer denoting ProvinceID in [0, 1, ..., 74]
        adjacency: a num_provinces by num_provinces adjacency matrix

    Returns:
        A list of ProvinceIDs."""
    
    adjacent_provinces = np.nonzero(adjacency[province, :])[0]
    return list(adjacent_provinces)


def get_unit_adjacency(
    province: utils.ProvinceID, unit_owner: int, board: np.array, ignore_own=True
):
    """Returns the number of units in adjacent provinces to the given unit.

    Args:
        province: int in [0, 1, ..., 74], the province ID of the unit.
        unit_owner: int in [0, 1, ..., 6], the power index of the unit's owner.
        board: the current board state.
        ignore_own: if True, only count enemy units. If False, include own units.

    Returns:
        An integer."""

    adj_count = 0
    adj_provinces = get_adjacent_provinces(province, adjacency_matrix)

    for adj in adj_provinces:
        owner = utils.unit_power(adj, board)
        if ignore_own and owner != unit_owner or not ignore_own and owner is not None:
            adj_count += 1

    return adj_count


def sort_units_by_adjacency(power: int, board: np.array, ignore_own=True):
    """Returns a list of the area IDs of all of a power's units, sorted by the number of adjacent units (in ascending order).

    Args:
        power: int in [0, 1, ... 6], the power index of the power whose units are to be sorted.
        board: the current board state.
        ignore_own: if True, only count enemy units. If False, include own units.

    Returns:
        A list of area IDs (integers in [0, 1, ...80])."""
    
    units = set(np.nonzero(board[:, 3 + power])[0])
    dislodgeds = set(np.nonzero(board[:, 16 + power])[0])
    unit_areas = units | dislodgeds

    adjacencies = {}

    for area in list(unit_areas):
        province = utils.province_id_and_area_index(area)[0]
        adjacencies[area] = get_unit_adjacency(province, power, board, ignore_own)

    sorted_units = [area for area, _ in sorted(adjacencies.items(), key=lambda item: item[1])]
    # print('power', power, f'{sorted_units=}', f'{adjacencies=}')
    
    return sorted_units
