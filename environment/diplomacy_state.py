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

from environment import observation_utils as utils
from environment import mila_actions

from diplomacy.engine.game import Game

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

class WelfareGame(Game):
    """Game based on the standard_welfare map, but using legal actions from the standard map (which is what the network policies expect)."""

    def __init__(self):
        super().__init__(map_name='standard_welfare')

    def get_all_possible_orders(self):
        """Same as get_all_possible_orders with welfare=False. (Not ideal; would prefer somehow to set welfare=False for super().get_all_possible_orders() and then back to True.)

        returns:
            A dictionary with locations as keys, and their respective list of possible orders as values
        """
        # 

        assert self.map is not None
        # pylint: disable=too-many-branches,too-many-nested-blocks
        possible_orders = {loc.upper(): set() for loc in self.map.locs}

        # Game is completed
        if self.get_current_phase() == "COMPLETED":
            return {loc: list(possible_orders[loc]) for loc in possible_orders}

        # Building a dict of (unit, is_dislodged, retreat_list, duplicate) for each power
        # duplicate is to indicate that the real unit has a coast, and that was added a duplicate unit without the coast
        unit_dict = {}
        for power in self.powers.values():
            # Regular units
            for unit in power.units:
                unit_loc = unit[2:]
                unit_dict[unit_loc] = (unit, False, [], False)
                if "/" in unit_loc:
                    unit_dict[unit_loc[:3]] = (unit, False, [], True)

            # Dislodged units
            for unit, retreat_list in power.retreats.items():
                unit_loc = unit[2:]
                unit_dict["*" + unit_loc] = (unit, True, retreat_list, False)

        # Building a list of build counts and build_sites
        build_counts = {
            power_name: len(power.centers) - len(power.units)
            if self.phase_type == "A"
            else 0
            for power_name, power in self.powers.items()
        }
        build_sites = {
            power_name: self._build_sites(power) if self.phase_type == "A" else []
            for power_name, power in self.powers.items()
        }

        # Movement phase
        if self.phase_type == "M":
            # Building a list of units and homes for each power
            power_units = {
                power_name: power.units[:] for power_name, power in self.powers.items()
            }

            # Hold
            for power_name in self.powers:
                for unit in power_units[power_name]:
                    order = unit + " H"
                    possible_orders[unit[2:]].add(order)
                    if "/" in unit:
                        possible_orders[unit[2:5]].add(order)

            # Move, Support, Convoy
            for power_name in self.powers:
                for unit in power_units[power_name]:
                    unit_type, unit_loc = unit[0], unit[2:]
                    unit_on_coast = "/" in unit_loc
                    for dest in self.map.dest_with_coasts[unit_loc]:
                        # Move (Regular)
                        if self._abuts(unit_type, unit_loc, "-", dest):
                            order = unit + " - " + dest
                            possible_orders[unit_loc].add(order)
                            if unit_on_coast:
                                possible_orders[unit_loc[:3]].add(order)

                        # Support (Hold)
                        if self._abuts(unit_type, unit_loc, "S", dest):
                            if dest in unit_dict:
                                other_unit, _, _, duplicate = unit_dict[dest]
                                if not duplicate:
                                    order = unit + " S " + other_unit[0] + " " + dest
                                    possible_orders[unit_loc].add(order)
                                    if unit_on_coast:
                                        possible_orders[unit_loc[:3]].add(order)

                        # Support (Move)
                        # Computing src of move (both from adjacent provinces and possible convoys)
                        # We can't support a unit that needs us to convoy it to its destination
                        abut_srcs = self.map.abut_list(dest, incl_no_coast=True)
                        convoy_srcs = self._get_convoy_destinations(
                            "A", dest, exclude_convoy_locs=[unit_loc]
                        )

                        # Computing coasts for source
                        src_with_coasts = [
                            self.map.find_coasts(src) for src in abut_srcs + convoy_srcs
                        ]
                        src_with_coasts = {
                            val for sublist in src_with_coasts for val in sublist
                        }

                        for src in src_with_coasts:
                            if src not in unit_dict:
                                continue
                            src_unit, _, _, duplicate = unit_dict[src]
                            if duplicate:
                                continue

                            # Checking if src unit can move to dest (through adj or convoy), and that we can support it
                            # Only armies can move through convoy
                            if (
                                src[:3] != unit_loc[:3]
                                and self._abuts(unit_type, unit_loc, "S", dest)
                                and (
                                    (src in convoy_srcs and src_unit[0] == "A")
                                    or self._abuts(src_unit[0], src, "-", dest)
                                )
                            ):
                                # Adding with coast
                                order = (
                                    unit
                                    + " S "
                                    + src_unit[0]
                                    + " "
                                    + src
                                    + " - "
                                    + dest
                                )
                                possible_orders[unit_loc].add(order)
                                if unit_on_coast:
                                    possible_orders[unit_loc[:3]].add(order)

                                # Adding without coasts
                                if "/" in dest:
                                    order = (
                                        unit
                                        + " S "
                                        + src_unit[0]
                                        + " "
                                        + src
                                        + " - "
                                        + dest[:3]
                                    )
                                    possible_orders[unit_loc].add(order)
                                    if unit_on_coast:
                                        possible_orders[unit_loc[:3]].add(order)

                    # Move Via Convoy
                    for dest in self._get_convoy_destinations(unit_type, unit_loc):
                        order = unit + " - " + dest + " VIA"
                        possible_orders[unit_loc].add(order)

                    # Convoy
                    if unit_type == "F":
                        convoy_srcs = self._get_convoy_destinations(
                            unit_type, unit_loc, unit_is_convoyer=True
                        )
                        for src in convoy_srcs:
                            # Making sure there is an army at the source location
                            if src not in unit_dict:
                                continue
                            src_unit, _, _, _ = unit_dict[src]
                            if src_unit[0] != "A":
                                continue

                            # Checking where the src unit can actually go
                            convoy_dests = self._get_convoy_destinations(
                                "A", src, unit_is_convoyer=False
                            )

                            # Adding them as possible moves
                            for dest in convoy_dests:
                                if self._has_convoy_path(
                                    "A", src, dest, convoying_loc=unit_loc
                                ):
                                    order = unit + " C A " + src + " - " + dest
                                    possible_orders[unit_loc].add(order)

        # Retreat phase
        if self.phase_type == "R":
            # Finding all dislodged units
            for unit_loc, (
                unit,
                is_dislodged,
                retreat_list,
                duplicate,
            ) in unit_dict.items():
                if not is_dislodged or duplicate:
                    continue
                unit_loc = unit_loc[1:]  # Removing the leading *
                unit_on_coast = "/" in unit_loc

                # Disband
                order = unit + " D"
                possible_orders[unit_loc].add(order)
                if unit_on_coast:
                    possible_orders[unit_loc[:3]].add(order)

                # Retreat
                for dest in retreat_list:
                    if dest[:3] not in unit_dict:
                        order = unit + " R " + dest
                        possible_orders[unit_loc].add(order)
                        if unit_on_coast:
                            possible_orders[unit_loc[:3]].add(order)

        # Adjustment phase
        if self.phase_type == "A":
            # Building a list of units for each power
            power_units = {
                power_name: power.units[:] for power_name, power in self.powers.items()
            }

            for power_name in self.powers:
                power_build_count = build_counts[power_name]
                power_build_sites = build_sites[power_name]

                # Disband
                if power_build_count < 0:
                    for unit in power_units[power_name]:
                        unit_on_coast = "/" in unit
                        order = unit + " D"
                        possible_orders[unit[2:]].add(order)
                        if unit_on_coast:
                            possible_orders[unit[2:5]].add(order)

                # Build
                if power_build_count > 0:
                    for site in power_build_sites:
                        for loc in self.map.find_coasts(site):
                            unit_on_coast = "/" in loc
                            if self.map.is_valid_unit("A " + loc):
                                possible_orders[loc].add("A " + loc + " B")
                            if self.map.is_valid_unit("F " + loc):
                                possible_orders[loc].add("F " + loc + " B")
                                if unit_on_coast:
                                    possible_orders[loc[:3]].add("F " + loc + " B")

                # Waive
                if power_build_count > 0:
                    for site in power_build_sites:
                        for loc in self.map.find_coasts(site):
                            possible_orders[loc].add("WAIVE")

        # Returning
        return {loc: list(possible_orders[loc]) for loc in possible_orders}

class WelfareDiplomacyState(DiplomacyState):
  
    def __init__(self, game: WelfareGame):
        self.game = game

        # Get a dictionary of power.name: power, ordered by power name
        power_names_sorted = sorted(game.powers.keys())
        self.powers = OrderedDict((name, game.powers[name]) for name in power_names_sorted)

        self._build_numbers = None
        self._last_actions = None
        
        self.turn_num = 0

    def is_terminal(self) -> bool:
        return self.game.is_game_done
    
    def observation(self) -> utils.Observation:
        """ Gets a utils.Observation namedtuple.
        
        Returns:
            utils.Observation(
                season: utils.Season,
                board: np.array shape (81, 35),
                build_numbers: np.array shape (7,),
                last_actions: list of actions (integers
            )
            """

        game = self.game
        powers = self.powers

        # SEASON
        
        # game.phase: string containing long rep of current phase
        if 'SPRING' in game.phase: 
            season = 'SPRING'
        elif 'FALL' in game.phase:
            season = 'AUTUMN'
        elif 'WINTER' in game.phase:
            # MILA calls 'BUILDS' phase 'WINTER ADJUSTMENT'
            season = 'BUILDS'
        else:
            raise ValueError('Not a valid season.')

        # game.phase_type: 'M' for Movement, 'R' for Retreats, 'A' for Adjustment, '-' for non-playing
        if 'M' in game.phase_type:
            season = getattr(utils.Season, season + '_MOVES')
        elif 'R' in game.phase_type:
            season = getattr(utils.Season, season + '_RETREATS')
        elif 'A' in game.phase_type:
            season = getattr(utils.Season, season)
        else:
            raise ValueError('Not a valid phase type.')
        
        
        # BOARD STATE & BUILD NUMBERS
 
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

            # A power can have at most as many units as supply centres it controls
            build_count = len(power.centers) - len(power.units)

            # How many units it can build depends on how many of its home supply centres it controls
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

                if area_ix > 0:
                    bicoastal_main_id = utils.area_from_province_id_and_area_index(province_id, 0)
                else:
                    bicoastal_main_id = None

                # Set ID to all 3 areas if bicoastal
                if province_type == utils.ProvinceType.BICOASTAL:
                    all_ids = [utils.area_from_province_id_and_area_index(province_id, n) for n in range(3)]
                
                # Fill in unit & owner info for area
                unit_type_ix = 0 if unit_type=='A' else 1

                # If unit in coast of bicoastal, also add unit flag for main area
                if area_ix > 0:
                    id = [id, bicoastal_main_id]

                if not is_dislodged:
                    unit_types[id, unit_type_ix] = 1
                    unit_owners[id, power_ix] = 1
                    # Set 'no unit' to 0
                    unit_types[id, -1] = 0
                    unit_owners[id, -1] = 0
                else:
                    dislodgeds[id, unit_type_ix] = 1
                    dislodged_owners[id, power_ix] = 1
                    # Set 'no dislodged unit' to 0
                    dislodgeds[id, -1] = 0
                    dislodged_owners[id, -1] = 0

                # Removable: if power has more units than centers, then this unit is removable. If the unit is on a coast of a bicoastal, only that coast is marked as removable.
                if build_count < 0:
                    if area_ix > 0:
                        removables[id[0]] = 1
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

        # Using the arg for the step() method
        if self._last_actions:
            last_actions = [action for power in self._last_actions for action in power]
        else:
            last_actions = []

        return utils.Observation(season, board, build_numbers, last_actions)
    
    def legal_actions(self) -> Sequence[Sequence[int]]:
        """Returns a list of legal actions for each power."""
        
        game = self.game
        powers = self.powers
        self.mila_legal_orders = []

        # Get possible orders with welfare set to False
        possible_orders_by_loc = game.get_all_possible_orders()

        # Get possible orders per power from possible orders by location
        possible_orders_by_power = {power: [] for power in powers.values()}

        # Retreats phase
        if game.phase_type == 'R' and game.dislodged:
            # Only possible moves are by dislodged units
            for power in powers.values():
                for unit in power.retreats:
                    loc = unit[2:]
                    possible_orders_by_power[power] += possible_orders_by_loc[loc]
        else:
            # Add all orders for non-dislodged units on board
            for power in powers.values():
                for unit in power.units:
                    loc = unit[2:]
                    orders = possible_orders_by_loc[loc]
                    possible_orders_by_power[power] += orders
            
            # In build phases, there can be orders for as-of-yet non-existent units.
            if game.phase_type == 'A':
                for power_ix, power in enumerate(powers.values()):
                # Check build number
                    if self.observation().build_numbers[power_ix] > 0:
                        for loc in game._build_sites(power):
                            orders = possible_orders_by_loc[loc]
                            possible_orders_by_power[power] += orders
        # Note: these orders are still in string form.

        # Store orders by power in MILA format for use in step() method
        orders_by_power_list = [possible_orders_by_power[power] for _, power in sorted(powers.items())]
        self.mila_legal_orders = orders_by_power_list
        
        season = self.observation().season

        legal_actions = [[] for _ in powers.values()]


        # Convert MILA orders to DM actions
        for power_ix, power_orders in enumerate(orders_by_power_list):
            for order in power_orders:
                # Returns an action_utils.Action integer
                action = mila_actions.mila_action_to_action(order, season)
                legal_actions[power_ix].append(action)

        # Sort actions and remove duplicates
        for power_ix in range(len(powers)):
            legal_actions[power_ix] = list(set(legal_actions[power_ix]))
            legal_actions[power_ix].sort()
            
        return legal_actions
    
    
    def returns(self) -> np.ndarray:
        """The returns of the game, equal to welfare points.
        
        (Could also set to all 0 while the game is still in progress.)"""

        welfare_points = [power.welfare_points for power in self.powers.values()]
        return np.array(welfare_points)
        
    
    def step(self, actions_per_player: Sequence[Sequence[int]]) -> None:
        """Steps the environment forward a full phase of Diplomacy.
        
        Args:
            actions_per_player: a list of lists of 64-bit integers corresponding to actions."""
        
        game = self.game
        powers = self.powers

        # Convert actions to MILA orders; orders will be lists, mostly with one item but some with multiple
        orders = [[mila_actions.action_to_mila_actions(act) if act else [] for act in player] for player in actions_per_player]
        
        orders_reduced = [[],[],[],[],[],[],[],]

        # Reduces lists of possible orders to a single order
        for power_ix, orders_per_power in enumerate(orders):
            for order in orders_per_power:
                # Order is a list of possible orders
                if len(order) > 1:
                    # Get legal one from list of possible actions
                    order_resolved = False
                    for possible_order in order:
                        if possible_order in self.mila_legal_orders[power_ix]:
                            orders_reduced[power_ix].append(possible_order)
                            order_resolved = True
                            break
                    
                    # If not in MILA legal orders, look for order with a unit that the power owns
                    if not order_resolved:
                        for possible_order in order:
                            word = possible_order.split()
                            unit = " ".join(word[:2])
                            if unit in list(powers.values())[power_ix].units:
                                orders_reduced[power_ix].append(possible_order)
                                order_resolved = True
                                break
        
                    assert order_resolved, 'None of possible orders in legal orders; possible orders are {}'.format(order)
                else:
                    # Get string from single-item list
                    orders_reduced[power_ix].append(order[0])

        orders = orders_reduced 

        # Set orders for each power
        for power_ix, power_name in enumerate(powers.keys()):
            game.set_orders(power_name, orders[power_ix], expand=False)

        # Step forward the environment
        game.process()
        self.turn_num += 1

        # Store actions for next observation
        self._last_actions = actions_per_player


