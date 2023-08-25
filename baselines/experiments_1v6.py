"""Run as a notebook.

This notebook allows you to run a series of 1v6 games of one focal agent vs a population of identical background agents, varying the number of turns after which the policies switch from the zero-sum (network) policy to a random disbanding policy.

Parameters to specify:
    alg: string denoting the network policy, in {'FPPI2', 'SL'}
    max_length: integer denoting number of turns to be played per game
    p: float for probability of disbanding each unit in the random disbanding policy
    focal: integer denoting the power index of the focal agent (in [Austria, England, France, Germany, Italy, Russia, Turkey])
    step: integer denoting step size in the sweep over number of turns after which to switch policies

All policies use the same network policy and disband policy; all that varies is the number of turns after which the policies switch.
"""
#%%
import logging
logging.basicConfig(filename='turn_sweep_experiments.log', filemode='a', level=logging.INFO)

import sys
sys.path.append('/Users/hannaherlebach/research/welfare-diplomacy/welfare_diplomacy_baselines/')

import os
import numpy as np
import nashpy as nash
from tqdm import tqdm
from functools import partial
from typing import Sequence

from network import parameter_provider, config, network_policy
from baselines import switch_policies as sp
from environment import diplomacy_state, game_runner
#%%

def print_payoff_matrix(matrix):
    # Assuming matrix shape is m x n x 2
    m, n, _ = matrix.shape
    
    # Print header
    print(" " * 10, end="")  # Some spacing
    for j in range(n):
        print(f"Player 2 Action {j + 1}", end="\t")
    print()  # New line

    # Print rows with payoffs
    for i in range(m):
        print(f"Player 1 Action {i + 1} |", end="\t")
        for j in range(n):
            print(tuple(matrix[i, j]), end="\t")
        print()  # New line

#%%
def nash_social_welfare(payoffs: Sequence[int]):
    """Calculates the Nash Social Welfare (i.e., geometric mean) of a sequence of payoffs."""
    welfare = 1
    for payoff in payoffs:
        welfare *= payoff
    return welfare ** (1/len(payoffs))

#%%

def get_network_policy_instance(algorithm='SL', file_path='/Users/hannaherlebach/research/diplomacy_parameters/'):
    """Returns a network policy instance, using SL or FFPI-2 parameters.
    
    Args:
        algorithm: str in ['SL', 'FFPI2']"""

    if algorithm=='SL':
        params = 'sl_params.npz'
    elif algorithm=='FPPI2':
        params = 'fppi2_params.npz'
    else:
        raise ValueError('Algorithm must be SL or FFPI2.')
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
#%%
class SwitchTurnPolicy:
    def __init__(self, network_policy, k, hard_coded_policy):
        self.network_policy = network_policy # a network policy instance
        self.k = k # number of turns after which to switch
        self.hard_coded_policy = hard_coded_policy # a hard-coded policy instance
        self.turn_num = 0

    def actions(self, slots_list, observation, legal_actions):
        """Plays the network policy for the first k turns, then plays the hard-coded policy."""
        if self.turn_num < self.k:
            #print('Network policy')
            actions = self.network_policy.actions(slots_list, observation, legal_actions)
        else:
            #print('Hard-coded policy')
            actions = self.hard_coded_policy.actions(slots_list, observation, legal_actions)
        self.turn_num +=1
        return actions

    def reset(self):
        self.network_policy.reset()
        self.hard_coded_policy.reset()

#%%
# Specify & log experiment parameters
alg = 'FPPI2'
max_length = 10
p = 0.5
focal = 1
step = 4

params = {
    'alg': alg,
    'max_length': max_length,
    'p': p,
    'focal': focal,
    'step': step
}

for key, value in params.items():
    logging.info(f"{key} = {value}")

#%%

# Create an array to store the payoffs.
payoff_matrix = np.zeros((max_length, max_length, 7))
disband_policy = sp.RandomDisbandPolicy(p=p)

# Sweep over i and j for focal and background policies.
for i in range(0, max_length, step): # focal 
    for j in range(0, max_length, step): # background       
        print(f'Focal: switches after {i} turns')
        print(f'Background: switches after {j} turns')

        # Create a new game instance
        game_instance = diplomacy_state.WelfareGame()
        initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)

        # Focal policy
        network_policy_instance = get_network_policy_instance(alg)
        focal_policy = SwitchTurnPolicy(network_policy_instance, i, disband_policy)

        # Background policy
        network_policy_instance = get_network_policy_instance(alg)
        background_policy = SwitchTurnPolicy(network_policy_instance, j, disband_policy)

        # Run game
        policies = (background_policy, focal_policy)
        slots_to_policies = [int(i==focal) for i in range(7)]
        
        trajectory = trajectory = game_runner.run_game(
            state = initial_state,
            policies = policies,
            slots_to_policies = slots_to_policies,
            max_length = max_length
            )
        
        # Store payoffs (i.e., final welfare points)
        payoffs = np.array(trajectory.returns)
        payoff_matrix[i,j] = payoffs

        logging.info("Payoffs for focal policy switching after %d turns and background policy switching after %d turns: %s", i, j, payoffs)

logging.info(f"Raw Payoff Matrix:\n{payoff_matrix}")
payoff_matrix = np.array(payoff_matrix)

#%%
"""Getting payoff matrices."""
# Slice matrix only to include games played
payoff_matrix_reduced = payoff_matrix[list(range(0, max_length, step)), :,:]
payoff_matrix_reduced = payoff_matrix_reduced[:, list(range(0, max_length, step)),:]

# Calculate NSW for everyone
nsw_everyone = np.apply_along_axis(nash_social_welfare, axis=-1, arr=payoff_matrix_reduced)
logging.info(f"Nash Social Welfare for everyone:\n{nsw_everyone}")

# Calculate NSW for background players
background_slots = [i for i in range(7) if slots_to_policies[i]==0]
focal_payoffs = payoff_matrix_reduced[:,:, focal]
background_payoffs = payoff_matrix_reduced[:,:, background_slots]
averaged_background_payoffs = np.apply_along_axis(nash_social_welfare, axis=-1, arr=background_payoffs)
# Last dimension is (focal payoff, averaged background payoff)
nsw_background = np.stack((focal_payoffs, averaged_background_payoffs), axis=-1)
print_payoff_matrix(nsw_background)
logging.info(f"Focal payoff against Nash Social Welfare of background:\n{nsw_background}")

#%%
"""Equilibrium analysis."""

# Focal against NSW background
final_focal_payoffs = nsw_background[:,:,0]
final_background_payoffs = nsw_background[:,:,1]
matrix_game = nash.Game(final_focal_payoffs, final_background_payoffs)

# Get a generator of all equilibria
equilibria = matrix_game.support_enumeration()

for n, equ in enumerate(equilibria): # eq is a tuple of arrays
    focal_equ, background_equ = equ
    print(f"Equilibrium {n}:")
    # Equilibrium strategy with prob for each k to switch on
    focal_strategy = {step * i: probs for i, probs in enumerate(focal_equ)}
    background_strategy = {step * i: probs for i, probs in enumerate(background_equ)}

    logging.info("Equilibrium focal strategy %d", n)
    for k, v in focal_strategy.items():
        logging.info("Disband after %d turns with probability %d", k, v)
    output = ",".join(f"\n\t{k} turns with probability {v}" for k, v in focal_strategy.items())
    print(f"Focal: disband after{output}")

    logging.info("Equilibrium background strategy %d", n)
    for k, v in background_strategy.items():
        logging.info("Disband after %d turns with probability %d", k, v)
    output = ",".join(f"\n\t{k} turns with probability {v}" for k, v in background_strategy.items())
    print(f"Background: disband after {output}")

#%%
"""Optimal social welfare analysis."""

optimal_game = np.unravel_index(np.argmax(nsw_everyone), nsw_everyone.shape)
logging.info("Optimal social welfare: when focal switches after %d turns and background switches after %d turns", optimal_game[0]*step, optimal_game[1]*step)
print(f"Optimal social welfare when focal switches after {optimal_game[0]*step} turns and background switches after {optimal_game[1]*step} turns")
#%%
