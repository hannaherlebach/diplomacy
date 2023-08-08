# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DiplomacyState protocol."""

from typing import Sequence
import numpy as np
import typing_extensions
from collections import OrderedDict

from diplomacy.environment import observation_utils as utils
from diplomacy.environment import mila_actions
from diplomacy.environment import human_readable_actions


class DiplomacyState(typing_extensions.Protocol):
  """Diplomacy State protocol."""

  def is_terminal(self) -> bool:
    """Whether the game has ended."""
    pass

  def observation(self) -> utils.Observation:
    """Returns the current observation."""
    pass

  def legal_actions(self) -> Sequence[Sequence[int]]:
    """A list of lists of legal unit actions.

    There are 7 sub-lists, one for each power, sorted alphabetically (Austria,
    England, France, Germany, Italy, Russia, Turkey).
    The sub-list has every unit action possible in the given position, for all
    of that power's units.
    """
    pass

  def returns(self) -> np.ndarray:
    """The returns of the game. All 0s if the game is in progress."""
    pass

  def step(self, actions_per_player: Sequence[Sequence[int]]) -> None:
    """Steps the environment forward a full phase of Diplomacy.

    Args:
      actions_per_player: A list of lists of unit-actions. There are 7
        sub-lists, one per power, sorted alphabetically (Austria, England,
        France, Germany, Italy, Russia, Turkey), each sublist is all of the
        corresponding player's unit-actions for that phase.
    """
    pass

# --- MY CODE BELOW --- #

from welfare_diplomacy.engine.game import Game

class WelfareDiplomacyState(DiplomacyState):
  
    def __init__(self, game: Game):
        self.game = game

        # Get a sorted list of power tuples of form 'power name: power instance'
        # game.powers is a dict of form {'FRANCE': FrancePower, ...}
        power_names_sorted = sorted(game.powers.keys())
        self.powers = OrderedDict((name, game.powers[name]) for name in power_names_sorted)

        self._build_numbers = None
        self._last_actions = None

    def is_terminal(self) -> bool:
        return self.game.is_game_done
    
    def observation(self) -> utils.Observation:
        """ Gets a utils.Observation namedtuple."""

        game = self.game
        powers = self.powers

        # SEASON: utils.Season

        # DeepMind <-> MILA conversions:
        # ? = NEWYEAR
        # SPRING_MOVES = SPRING MOVEMENT
        # SPRING_RETREATS = SPRING RETREATS3
        # AUTUMN_MOVES = FALL MOVEMENT
        # AUTUMN_RETREATS = FALL RETREATS
        # BUILDS = WINTER ADJUSTMENT
        
        # To do: figure out the NEWYEAR phase

        # game.phase: string containing long rep of current phase
        if 'SPRING' in game.phase:
            season = 'SPRING'
        elif 'FALL' in game.phase:
            season = 'AUTUMN'
        elif 'WINTER' in game.phase:
            season = 'BUILDS'
        else:
            raise ValueError('not a season')

        # game.phase_type: 'M' for Movement, 'R' for Retreats, 'A' for Adjustment, '-' for non-playing
        if 'M' in game.phase_type:
            season = getattr(utils.Season, season + '_MOVES')
            type = 'MOVES'
        elif 'R' in game.phase_type:
            season = getattr(utils.Season, season + '_RETREATS')
            type = 'RETREATS'
        elif 'A' in game.phase_type:
            season = getattr(utils.Season, season)
        else:
            raise ValueError('not a season')
        # (could make this cleaner)
        
        
        # BOARD STATE & BUILD NUMBERS
        # Board: np.array shape (81, 35)
        # Build numbers: np.array shape (7,)
 
        supply_centers = game.map.scs # tags
        supply_centers_owned = set()

        # Initialise build numbers
        build_numbers = [0, 0, 0, 0, 0, 0, 0]

        # Initialise matrix components
        unit_types = np.zeros((81,3), dtype=np.uint8)
        unit_owners = np.zeros((81,8), dtype=np.uint8)
        buildables = np.zeros((81,1), dtype=np.uint8)
        removables = np.zeros((81,1), dtype=np.uint8)
        dislodgeds = np.zeros((81,3), dtype=np.uint8)
        dislodged_owners = np.zeros((81,8), dtype=np.uint8)
        area_types = np.zeros((81,3), dtype=np.uint8)
        sc_owners = np.zeros((81,8), dtype=np.uint8)

        # Set default to no unit
        unit_types[:, -1] = 1
        unit_owners[:, -1] = 1
        dislodgeds[:, -1] = 1
        dislodged_owners[:, -1] = 1

        # Areas containing units and owned supply centres
        for power_ix, (power_name, power) in enumerate(powers.items()):

            #print('dislodged units', power.retreats)

            #print(power_name, 'build sites', game._build_sites(power))
            build_count = len(power.centers) - len(power.units)

            build_sites = game._build_sites(power)

            for unit in game.get_units(power_name): # contains dislodgement info (*), unlike power.units (returns power.units[:] + [<dislodged units>])
                
                is_dislodged = bool(unit[0]=="*")

                unit = unit[1:] if is_dislodged else unit
                unit_type = unit[0]
                mila_tag = unit[2:]
                province = mila_tag[:3]
                id = mila_actions.mila_to_dm_area(mila_tag)
                province_id, area_ix = utils.province_id_and_area_index(id)
                province_type = utils.province_type_from_id(province_id)

                # Old
                if area_ix > 0:
                    bicoastal_main_id = utils.area_from_province_id_and_area_index(province_id, 0)
                else:
                    bicoastal_main_id = None

                # Set ID to all 3 areas if bicoastal
                if province_type == utils.ProvinceType.BICOASTAL:
                    all_ids = [utils.area_from_province_id_and_area_index(province_id, n) for n in range(3)]
                
                # Fill in unit & owner info for area
                unit_type_ix = 0 if unit_type=='A' else 1

                # Old
                # If unit in coast of bicoastal, also add unit flag for main area
                if area_ix > 0:
                    id = [id, bicoastal_main_id]

                # NOTE: MAY ALSO NEED TO DO THIS FOR THE OTHER COAST SOMETIMES

                if not is_dislodged:
                    unit_types[id, unit_type_ix] = 1
                    unit_owners[id, power_ix] = 1
                    # Set 'no unit' to 0
                    unit_types[id, -1] = 0
                    unit_owners[id, -1] = 0
                else:
                    print('there is a dislodged unit here! in', province)
                    dislodgeds[id, unit_type_ix] = 1
                    dislodged_owners[id, power_ix] = 1
                    # Set 'no dislodged unit' to 0
                    dislodgeds[id, -1] = 0
                    dislodged_owners[id, -1] = 0


                # Removable: if power has more units than centers, then this unit is removable
                if build_count < 0:
                    removables[id] = 1
                    if province_type == utils.ProvinceType.BICOASTAL:
                        removables[all_ids] = 1
                    else:
                        removables[id] = 1
                    

            for sc in power.centers: # mainland only
                id = mila_actions.mila_to_dm_area(sc)
                province_id, area_ix = utils.province_id_and_area_index(id)
                province_type = utils.province_type_from_id(province_id)

                if province_type == utils.ProvinceType.BICOASTAL:
                    sc_ids = [utils.area_from_province_id_and_area_index(province_id, ix) for ix in [0,1,2]]
                else:
                    sc_ids = id

                # Supply center owner
                sc_owners[sc_ids, power_ix] = 1

                # MAYBE REMOVE
                # Set ID to all 3 areas if bicoastal
                if province_type == utils.ProvinceType.BICOASTAL:
                    all_ids = [utils.area_from_province_id_and_area_index(province_id, n) for n in range(3)]

                # Buildable
                # Use the build count if BUILD season, otherwise used stored build number
                if season == utils.Season.BUILDS:
                    if sc in build_sites and build_count > 0:
                        if province_type == utils.ProvinceType.BICOASTAL:
                            buildables[all_ids] = 1
                        else:
                            buildables[id] = 1
                else:
                    if sc in build_sites and build_numbers[power_ix] > 0:
                        if province_type == utils.ProvinceType.BICOASTAL:
                            buildables[all_ids] = 1
                        else:
                            buildables[id] = 1

                supply_centers_owned.add(sc)

            # Set build numbers if build season
            if season == utils.Season.BUILDS:
                if build_count < 0:
                    build_numbers[power_ix] = build_count
                else:
                    # game._build_limit(power) gives the number of unoccupied home supply centers
                    build_numbers[power_ix] = min(build_count, game._build_limit(power))
        
        # Store build numbers if build season
        if season == utils.Season.BUILDS:          
            self._build_numbers = build_numbers
        # Otherwise access numbers from last build season with positives zeroed out, if not first year
        elif self._build_numbers:
            build_numbers = [n if n < 0 else 0 for n in self._build_numbers]

        
        # Unoccupied supply centres
        supply_centers_unowned = [sc for sc in supply_centers if sc not in supply_centers_owned]
        for sc in supply_centers_unowned:
            id = mila_actions.mila_to_dm_area(sc)
            province_id, area_ix = utils.province_id_and_area_index(id)
            if utils.province_type_from_id(province_id) == utils.ProvinceType.BICOASTAL:
                sc_owners[id:id+3, -1] = 1
            else:
                sc_owners[id, -1] = 1

        # Area type: fill out from area id
        for id in range(81):
            province_id, area_ix = utils.province_id_and_area_index(id)
            if area_ix==1 or area_ix==2: # coasts of bicoastals
                area_types[id, 2] = 1
            elif province_id >= 14 and province_id < 33: # sea
                area_types[id,1] = 1
            else: # all land with 0 or 1 coast, or main area of bicoastal
                area_types[id, 0] = 1

        board = np.concatenate(
            (unit_types, unit_owners, buildables, removables, dislodgeds, dislodged_owners, area_types, sc_owners),
            axis=1)

        # LAST ACTIONS
        # Using the step method
        if self._last_actions:
            last_actions = [action for power in self._last_actions for action in power]
        else:
            last_actions = None

        return utils.Observation(season, board, build_numbers, last_actions)
    
    def legal_actions(self) -> Sequence[Sequence[int]]:
        game = self.game
        powers = self.powers

        # Get possible orders from MILA
        orders_by_power = {power: [] for power in powers.values()}

        build_sites = {power: game._build_sites(power) if game.phase_type == "A" else []
                             for power in powers.values()}
        build_sites_by_loc = {site: power for power, sites in build_sites.items() for site in sites}

        for loc, possible_orders in game.get_all_possible_orders().items():
            
            # If BUILD phase and loc is a build site for power, then all orders in that loc belong to that power
            if game.phase_type == "A" and loc in build_sites_by_loc.keys():
                power = build_sites_by_loc[loc]
                orders_by_power[power] += possible_orders

            # Otherwise, the possible_orders refer to a unit in loc. Check who owns the unit
            else:
                unit = game._occupant(loc)
                if unit:
                    unit_owner = game._unit_owner(unit) # returns power instance
                    orders_by_power[unit_owner] += possible_orders


        # I think these actions are all the legal (i.e., possible given the board state) actions; if not, may want to run commented out code instead of the code below

        # # Check if legal!
        # legal_actions_by_power = {power: [] for power in powers.values()}
        # season = self.observation().season

        # for power in powers.values():
        #     for orders in orders_by_power[power]:
        #         for order in orders:
        #            # Split order into unit and order on unit
        #             unit = order.split()[:2]
        #           unit_order = order.split()[2:]
        #             is_valid = game._valid_order(power, unit, unit_order)

        #           if is_valid in {-1, 1}:
        #                 # Convert MILA order to DM action
        #               action = mila_actions.mila_action_to_action(order, season)
        #               legal_actions_by_power[power].append(action)        

        # for step()
        #print('orders by power dict', orders_by_power)

        # dont need this rlly
        orders_by_power_list = [orders_by_power[power] for _, power in sorted(powers.items())]

        
        season = self.observation().season

        legal_actions = [[] for _ in powers.values()]


        # Convert MILA orders to DM actions
        for power_ix, power_orders in enumerate(orders_by_power_list):
            for order in power_orders:
                action = mila_actions.mila_action_to_action(order, season)
                legal_actions[power_ix].append(action)

        self.mila_legal_orders = orders_by_power_list
            
        return legal_actions
    
    
    def returns(self) -> np.ndarray:
        """The returns of the game. All 0s if the game is in progress."""

        # (Not sure if this should be cumulative, or only at end of game)
        if self.game.get_current_phase() == 'COMPLETED':
            welfare_points = [power.welfare_points for power in self.powers.values()]
            return np.array(welfare_points)
        else:
            return np.zeros(7) # dtype?
    
    def step(self, actions_per_player: Sequence[Sequence[int]]) -> None:
        # actions_per_player are given as 64-bit integers
        
        game = self.game
        powers = self.powers
        legal_actions = self.legal_actions()
        board = self.observation().board

        # Convert actions to MILA orders; orders will be lists, mostly with one item but some with multiple
        orders = [[mila_actions.action_to_mila_actions(act) for act in player] for player in actions_per_player]

        # print('mila orders first step', orders)
        
        orders_reduced = [[],[],[],[],[],[],[],]

        # Reduces lists of possible orders to a single order
        for player_ix, orders_per_player in enumerate(orders):
            for order in orders_per_player:
                # Order is a list of possible orders
                if len(order) > 1:
                    # Get legal one from list of possible actions
                    for possible_order in order:
                        if possible_order in self.mila_legal_orders[player_ix]:
                            orders_reduced[player_ix].append(possible_order)
                            break
                else:
                    # Get string from single-item list
                    orders_reduced[player_ix].append(order[0])

        orders = orders_reduced

        # Set orders for each power
        for power_ix, power_name in enumerate(powers.keys()):
            game.set_orders(power_name, orders[power_ix], expand=False)

        # Step forward the environment
        game.process()

        # Store actions for next observation
        self._last_actions = actions_per_player