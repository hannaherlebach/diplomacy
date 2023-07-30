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

from diplomacy.environment import observation_utils as utils
from diplomacy.environment import mila_actions


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
from diplomacy.welfare-diplomacy.diplomacy.engine.game import Game

# -- HELPER FUNCTIONS --

# Season
def mila_to_dm_season(game):
    """Gets a utils.Season from the game.
    
    DeepMind / MILA conversion:
      ? = NEWYEAR
      SPRING_MOVES = SPRING MOVEMENT
      SPRING_RETREATS = SPRING RETREATS
      AUTUMN_MOVES = FALL MOVEMENT
      AUTUMN_RETREATS = FALL RETREATS
      BUILDS = WINTER ADJUSTMENT
    """

    # game.phase: string containing long rep of current phase
    if 'SPRING' in game.phase:
        season = 'SPRING'
    elif 'FALL' in game.phase:
        season = 'AUTUMN'
    elif 'WINTER' in game.phase:
        return getattr(utils.Season, 'BUILD')
    else:
        raise ValueError('not a season')

    # game.phase_type: 'M' for Movement, 'R' for Retreats, 'A' for Adjustment, '-' for non-playing
    if 'M' in game.phase_type:
        type = 'MOVES'
    elif 'R' in game.phase_type:
        type = 'RETREATS'
    else:
        raise ValueError('not a season')

    return getattr(utils.Season, season + '_' + type)

# Board
def mila_to_dm_board(game):
    """Returns an array of shape 81 x 35 capturing the board state,
    as required by the observation format in utils.Observation.

    Future improvements:
    - Replace values with variable names
    """

    power_names_sorted = game.powers.keys().sorted()
    powers = [game.powers[name] for name in power_names_sorted]

    supply_centers = game.map.scs # tags
    supply_centers_owned = set()

    provinces_with_units = ()

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
    for power_ix, power in enumerate(powers):

        build_count = len(power.centers) - len(power.units)
        self_occupied_homes = set() # for finding buildable homes

        for unit in power.units:
            
            is_dislodged = bool(unit[0]=="*")

            unit = unit[1:] if is_dislodged else unit
            unit_type = unit[0]
            mila_tag = unit[2:]
            province = mila_tag[:3]
            id = mila_actions.mila_to_dm_area(mila_tag)

            # Fill in unit & owner info for area
            unit_type_ix = 0 if unit_type=='A' else 1

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

            # Check if unit occupies one of power's homes
            if province in power.homes:
                self_occupied_homes.add(province)

            # Removable: if power has more units than centers, then this unit is removable
            if build_count < 0:
                removables[id] = 1

            provinces_with_units.add(mila_tag[:3]) # doesn't include bicoastal division


        for sc in power.centers: # mainland only
            id = mila_actions.mila_to_dm_area(sc)

            # Supply center owner
            sc_owners[id, power_ix] = 1

            # Buildable: if power controls its home province, there's not a unit in it, and power has more SCs than units, then this area is buildable (since the power controls its home supply center, and building only happens in winter, there cannot be a different power's unit in it at build time)
            if sc in power.homes and sc not in self_occupied_homes and build_count > 0:
               buildables[id] = 1

            supply_centers_owned.add(sc)

    # Unoccupied supply centres
    supply_centers_unowned = [sc for sc in supply_centers if sc not in supply_centers_owned]
    for sc in supply_centers_unowned:
       id = mila_actions.mila_to_dm_area(sc)
       sc_owners[id, -1] = 1

    # Area type: fill out from area id
    for id in range(81):
        province_id, area_ix = utils.province_id_and_area_index(id)
        if area_ix==1 or area_ix==2: # coasts of bicoastals
            area_types[id, 2] = 1
        elif province_id >= 14 and province_id < 33: # sea
            area_types[id,1] = 1
        else: # all land with 0 or 1 coast
            area_types[id, 0] = 1

    return np.concatenate(
        (unit_types, unit_owners, buildables, removables, dislodgeds, dislodged_owners, area_types, sc_owners),
        dim=1)


# -- DIPLOMACY STATE IMPLEMENTATION --

class WelfareDiplomacyState(DiplomacyState):
  
  def __init__(self, game: Game):
    self.game = game

  def is_terminal(self) -> bool:
    return self.game.is_game_done
  
  def observation(self) -> utils.Observation: