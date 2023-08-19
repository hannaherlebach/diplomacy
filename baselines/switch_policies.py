import os
import wandb
from tqdm import tqdm
import numpy as np
from functools import partial

from diplomacy.network import config
from diplomacy.network import network_policy
from diplomacy.network import parameter_provider
from diplomacy.environment import diplomacy_state
from diplomacy.environment import game_runner
from diplomacy.environment import observation_utils as utils
from diplomacy.environment import mila_actions
from diplomacy.environment import action_utils

_MILA_TO_DM_TAG_MAP = {v: k for k, v in mila_actions._DM_TO_MILA_TAG_MAP.items()}

file_path = '/Users/hannaherlebach/research/diplomacy_parameters/'

# Get base network policy
def get_network_policy_instance(SL=True):
    """Returns a network policy instance, using SL or FFPI-2 parameters."""
    if SL:
        params = 'sl_params.npz'
    else:
        params = 'fppi2_params.npz'
    with open(os.path.join(file_path, params), 'rb') as f:
        provider = parameter_provider.ParameterProvider(f)
    
    network_info = config.get_config()
    network_handler = parameter_provider.SequenceNetworkHandler(
        network_cls=network_info.network_class,
        network_config=network_info.network_kwargs,
        parameter_provider=provider,
        rng_seed=42)
    network_policy_instance = network_policy.Policy(
        network_handler=network_handler,
        num_players=7,
        temperature=0.1,
        calculate_all_policies=False)
    
    return network_policy_instance


# Get hard-coded policies
class RandomDisbandPolicy:
    """Agent which disbands units randomly in BUILD phases, and holds otherwise."""

    def __init__(self, p, num_players=7):
        self.num_players = num_players
        self.p = p

    def reset(self):
        pass

    def actions(self, slots_list, observation, legal_actions):
        """Produces a list of lists of actions, one for each slot.
        
        Args:
            slots_list: a list of slots (integers in range(num_players) this policy should produce actions for.)"""
        
        assert len(slots_list) <= self.num_players

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

                    disband_action = action_utils.construct_action(8, province_tuple, None, None)

                    # Add disband action to actions with probability p
                    if np.random.uniform() < self.p:
                        actions[i].append(disband_action)

        # In all other phases, hold
        
        return [actions, {'values': None, 'policy': None, 'actions': None}] #fill out this shit later
    
class SmartDisbandPolicy:
    """Agent which disbands the units with fewest adjacent enemy units first."""
    def __init__(self, num_to_disband):
        self.num_to_disband = num_to_disband

    def reset(self):
        pass

    def actions(self, slots_list, observation, legal_actions):
        board = observation.board
        season = observation.season

        units = board[:, 3:10]
        dislodgeds = board[:, 16:23]

        actions = [[] for _ in slots_list]

        if season == utils.Season.BUILDS:
            for i, power_ix in enumerate(slots_list):
                units_power = units[:, power_ix]
                dislodgeds_power = dislodgeds[:, power_ix]
                
                # Get adjacency order

        # In all other phases, hold
        
        return [actions, {'values': None, 'policy': None, 'actions': None}] #fill out this shit later


# Switch conditions
def switch_after_k_turns(turn_num, power_ix, observation, k):
    """Returns True if turn_num > k, else False.
    
    How to use:
        Create a partial function by specifying k, and pass to SwitchPolicy instance."""
    return turn_num > k

def switch_after_k_supply_centers(turn_num, power_ix, observation, k):
    """Returns True if power has > k supply centers, else False.

    How to use:
        Create a partial function by specifying k, and pass to SwitchPolicy instance."""
    distinct_provinces = [i for i in range(81) if i not in {73, 74, 76, 77, 79, 80}]
    supply_centers_board = observation.board[distinct_provinces,-8:]
    supply_centers_power = supply_centers_board[:, power_ix]
    supply_center_count = np.sum(supply_centers_power)
    return supply_center_count > k
    
# Example implementations of partial functions which can be passed to a SwitchPolicy instance
switch_after_10_supply_centers = partial(switch_after_k_supply_centers, k=10)
switch_after_10_turns = partial(switch_after_k_turns, k=10)


class SwitchPolicy:
    """Wrapper for network and hard-coded policies, with a switch when a given condition is met.
    
    Args:
        network_policy: a network policy instance (SL or FFPI-2).
        switch_condition: a function that takes a state and returns True or False.
        hard_coded_policy: a hard-coded policy instance.
    """
    def __init__(self, network_policy, switch_condition, hard_coded_policy):
        self.network_policy = network_policy # a network policy instance
        self.switch_condition = switch_condition
        self.hard_coded_policy = hard_coded_policy # a hard-coded policy instance
        self.switch = [False for _ in range(7)]
        self.turn_num = 0

    def actions(self, slots_list, observation, legal_actions):
        
        print(f'{self.turn_num=}')

        # actions = []

        # This will put actions in the wrong order

        # unswitched = [power_ix for power_ix in slots_list if not self.switch_condition(self.turn_num, power_ix, observation)]
        # switched = [power_ix for power_ix in slots_list if self.switch_condition(self.turn_num, power_ix, observation)]

        # network_actions = self.network_policy.actions(unswitched, observation, legal_actions)
        # hard_coded_actions = self.hard_coded_policy.actions(switched, observation, legal_actions)

        # # This doesn't quite work for switch_after_k_turns, not sure why
        # for power_ix in slots_list:
        #     if not self.switch_condition(self.turn_num, power_ix, observation):
        #         print('Network policy')
        #         actions.append(self.network_policy.actions([power_ix], observation, legal_actions)[0][0])
        #     else:
        #         print('Hard-coded policy')
        #         actions.append(self.hard_coded_policy.actions([power_ix], observation, legal_actions)[0][0])

        actions = self.network_policy.actions(slots_list, observation, legal_actions) #tuple

        for i, power_ix in enumerate(slots_list):
            if self.switch_condition(self.turn_num, power_ix, observation) or self.switch[power_ix]:
                self.switch[power_ix] = True
                print('power', power_ix, 'disbanding from now on')
                actions[0][i] = self.hard_coded_policy.actions([power_ix], observation, legal_actions)[0][0]
                
        # print(actions[0])

        self.turn_num += 1

        # print(len(actions[0]))

        return actions # return values and policy for network policy

    def reset(self):
        self.network_policy.reset()
        self.hard_coded_policy.reset()



# Simultaneous Switch class deals with the case where all players switch at the same time more efficiently than Switch.

def simultaneous_switch_after_k_turns(turn_num, observation, k):
    """Returns True if turn_num > k, else False. Like switch_after_k_turns, but omits power_ix argument.
    
    How to use:
        Create a partial function by specifying k, and pass to SwitchPolicy instance."""
    return turn_num > k

class SimultaneousSwitchPolicy:
    """All players switch from the network policy to the hard-coded policy at the same time."""
    def __init__(self, network_policy, switch_condition, hard_coded_policy):
        self.network_policy = network_policy # a network policy instance
        self.switch_condition = switch_condition # must be a property of the state, not individual powers
        self.hard_coded_policy = hard_coded_policy # a hard-coded policy instance
        self.turn_num = 0

    def actions(self, slots_list, observation, legal_actions):
        
        print(f'{self.turn_num=}')

        if not self.switch_condition(self.turn_num, observation):
            print('Network policy')
            actions = self.network_policy.actions(slots_list, observation, legal_actions)
            self.turn_num +=1
            return actions
        else:
            print('Hard-coded policy')
            actions = self.hard_coded_policy.actions(slots_list, observation, legal_actions)
            self.turn_num +=1
            return actions

    def reset(self):
        self.network_policy.reset()
        self.hard_coded_policy.reset()


def run_experiment(policies, slots_to_policies, max_length=20):
    """Runs the chosen set of policies on Welfare Diplomacy."""
    game_instance = diplomacy_state.BaselineGame() # defo cursed
    initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)
    trajectory = game_runner.run_game(
        state = initial_state,
        policies = policies,
        slots_to_policies = slots_to_policies,
        max_length = max_length
)
    print('Trajectory', trajectory)
    print(trajectory.observations[-1].season)


if __name__ == '__main__':


    switch_after_10_turns = partial(switch_after_k_turns, k=10)
    switch_after_20_turns = partial(switch_after_k_turns, k=20)
    switch_after_5_supply_centers = partial(switch_after_k_supply_centers, k=5)


    mixed_policy_1 = SwitchPolicy(get_network_policy_instance(), switch_after_10_turns, RandomDisbandPolicy(p=0.5))
    mixed_policy_2 = SwitchPolicy(get_network_policy_instance(), switch_after_20_turns, RandomDisbandPolicy(p=0.5))
    mixed_policy_3 = SwitchPolicy(get_network_policy_instance(), switch_after_5_supply_centers, RandomDisbandPolicy(p=0.5))

    simultaneous_switch_after_10_turns = partial(simultaneous_switch_after_k_turns, k=10)
    mixed_policy_4 = SimultaneousSwitchPolicy(get_network_policy_instance(), simultaneous_switch_after_10_turns, RandomDisbandPolicy(p=0.5))



#     # Experiment 1
#     print('Experiment 1')
#     trajectory = run_experiment(
#         policies = (mixed_policy_1, random_disband_policy),
#         slots_to_policies = [0] + [1] * 6
# )
#     # Experiment 2
#     print('Experiment 2')
#     trajectory = run_experiment(
#         policies = (mixed_policy_1, mixed_policy_2),
#         slots_to_policies = [0] * 6 + [1],
#         max_length=30
# )
#     # Experiment 3
#     print('Experiment 3')
#     trajectory = run_experiment(
#         policies = (sl_policy,),
#         slots_to_policies = [0] * 7,
# )
#     # Experiment 4
#     print('Experiment 4')
#     trajectory = run_experiment(
#         policies = (mixed_policy_2,),
#         slots_to_policies = [0]*7
# )
#     # Experiment 5
#     print('Experiment 5')
#     trajectory = run_experiment(
#         policies = (mixed_policy_1,),
#         slots_to_policies = [0] * 7 # + [1] * 6
# )

    # Experiment 6
#     print('Experiment 6')
#     trajectory = run_experiment(
#         policies = (mixed_policy_2, mixed_policy_2, mixed_policy_2, mixed_policy_2, mixed_policy_2, mixed_policy_2, mixed_policy_2),
#         slots_to_policies = list(range(7))
# )

    # Experiment 7
    # print('Experiment 7')
    # trajectory = run_experiment(
    #     policies = (mixed_policy_1,),
    #     slots_to_policies= [0] * 7,
    #     max_length=20
    # )
    
#     # Experiment 8
#     print('Experiment 8')
#     trajectory = run_experiment(
#         policies = (mixed_policy_3,),
#         slots_to_policies = [0] * 7,
#         max_length=30
# )

    # Experiment 9
    print('Experiment 9')
    trajectory = run_experiment(
        policies = (mixed_policy_4,),
        slots_to_policies = [0] * 7,
        max_length=30
    )
