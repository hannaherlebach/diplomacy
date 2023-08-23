"""Running 1v6 games while varying the switch condition for the focal and background plaeyrs.

All policies use the same network policy and disband policy. All that varies is k in the switch condition (so far, k = number of turns, or k = number of supply centres).
"""

import argparse
import numpy as np
from functools import partial

from baselines import switch_policies as sp

def main():
    args = parse_args()
    alg = args.network_algorithm
    max_length = args.max_length
    p = args.disband_p
    focal = args.focal_slot
    step = args.step

    # Switching based on turn number.
    if args.switch == 'turn':
        payoffs = run_experiments_turns(max_length, step, p, alg, focal)
        print(payoffs)

    # Finish off

def run_experiments_turns(max_length, step, p, alg, focal):
    # Create an array to store the payoffs.
    #num_k = int(max_length/step)
    #payoff_matrix = np.zeros((num_k, num_k, 7))
    payoff_matrix = np.zeros((max_length, max_length, 7))
    
    # Sweep over i and j for focal and background policies.
    for i in range(0, max_length, step):
        for j in range(0, max_length, step):
            # Focal policy
            network_policy = sp.get_network_policy_instance(alg)
            switch_condition = partial(sp.switch_after_k_turns, k=i)
            disband_policy = sp.RandomDisbandPolicy(p=p)
            focal_policy = sp.SwitchPolicy(network_policy, switch_condition, disband_policy)

            # Background policy
            network_policy = sp.get_network_policy_instance(alg)
            switch_condition = partial(sp.switch_after_k_turns, k=j)
            disband_policy = sp.RandomDisbandPolicy(p=p)
            background_policy = sp.SwitchPolicy(network_policy, switch_condition, disband_policy)

            # Run game
            policies = (focal_policy, background_policy)
            slots_to_policies = [int(i!=focal) for i in range(7)]
            trajectory = sp.run_experiment(
                policies=policies,
                slots_to_policies=slots_to_policies,
                max_length=max_length)
            
            # Store payoffs (i.e., final welfare points)
            payoffs = np.array(trajectory.returns)
            payoff_matrix[i,j] = payoffs
    
    return payoff_matrix

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run a game of Welfare Diplomacy.')

    # Arguments are None if not set in command line. Args need to be set with 0, 1 or num_policies elements. If an arg has length 1 but num_policies > 1, it is assumed that all policies take the same arg value.
    parser.add_argument("--max_length", type=int, default=30, help="Maximum number of game steps.")
    parser.add_argument('--switch', type=str, choices=['turn', 'sc'], help='Switch condition type.')
    parser.add_argument('--disband_p', type=float, help='Probability for RandomDisbandPolicy')
    parser.add_argument('--network_algorithm', type=str, default='SL', choices=['SL', 'FPPI2'], help='Learning algorithm used to get network policy parameters.')
    parser.add_argument('--focal_slot', type=int, default=1, choices=list(range(7)), help='Index of power which will be the focal player. Default is England.')
    parser.add_argument('--step', type=int, default=1, help='Step by which to vary k in switch condition across experiments.')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()