""""Code for no-press policies to play Welfare Diplomacy against language models.

These policies are modifications on a network policy trained for Standard Diplomacy.
The algorithm used for the network policy is FPPI-2.
See https://arxiv.org/abs/2006.04635 for more details."""

import os
import numpy as np
from functools import partial
import logging
import argparse
from typing import Any, Sequence, Tuple

import sys
sys.path.append('/Users/hannaherlebach/research/welfare-diplomacy/welfare_diplomacy_baselines/')

from network import config, network_policy, parameter_provider
from environment import diplomacy_state, game_runner, mila_actions, action_utils
from environment import observation_utils as utils

from baselines import disband_policies

from diplomacy.engine.map import Map
welfare_map = Map('standard_welfare')

from diplomacy.engine.game import Game

class SwitchPolicy:
    """A policy that switches from the network policy to disbanding policy after a certain number of years.
    If year_to_switch = n, then the disbanding policy will start taking effect in the adjustment phase of year n.
    
    Args:
        disband_policy: instance of the chosen disbanding policy
        year_to_switch: int, year in which to switch to disbanding policy"""
    
    def __init__(self, disband_policy, year_to_switch=None):
        self.disband = False
        self.disband_policy = disband_policy
        self.network_policy = get_network_policy_instance()
        self.year = 0

        if year_to_switch is None:
            year_to_switch = float('inf')
        self.year_to_switch = year_to_switch

    def reset(self):
        self.network_policy.reset()
        self.disband_policy.reset()
        
    def actions(self, slots_list, observation, legal_actions):
        season = observation.season
        if season == utils.Season.BUILDS:
            if self.year == self.year_to_switch:
                self.disband = True
            self.year += 1

        if not self.disband:
            # Run network policy
            actions = self.network_policy.actions(slots_list, observation, legal_actions)
        else:
            # Run disband policy
            actions = self.disband_policy.actions(slots_list, observation, legal_actions)

        return actions

def get_network_policy_instance(algorithm='FPPI2', file_path='/Users/hannaherlebach/research/diplomacy_parameters/'):
    """Returns a network policy instance.
     
    By default all experiments should use the FFPI-2 parameters, but the SL parameters are also available.
    
    Args:
        algorithm: str in ['SL', 'FFPI2']
        file_path: str, path to directory containing the parameters"""

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

policy_map = {
    0: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=0),
    1: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=1),
    2: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=2),
    3: lambda: SwitchPolicy(disband_policies.InstantDisbandPolicy(), year_to_switch=3),
    10: lambda: SwitchPolicy(disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=0),
    11: lambda: SwitchPolicy(disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=1),
    12: lambda: SwitchPolicy(disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=2),
    13: lambda: SwitchPolicy(disband_policies.RandomDisbandPolicy(p=0.5), year_to_switch=3),
    20: lambda: SwitchPolicy(disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=0),
    21: lambda: SwitchPolicy(disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=1),
    22: lambda: SwitchPolicy(disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=2),
    23: lambda: SwitchPolicy(disband_policies.SmartDisbandPolicy(num_to_disband=1), year_to_switch=3),
}


# Test on DM setup
def main():
    game_instance = Game()
    initial_state = diplomacy_state.WelfareDiplomacyState(game_instance)
    policies = (SwitchPolicy(disband_policies.SmartDisbandPolicy(), year_to_switch=1),)#(SwitchPolicy(disband_policy = disband_policies.InstantDisbandPolicy(), year_to_switch=1),)
    slots_to_policies = [0]*7
    trajectory = game_runner.run_game(
        state=initial_state,
        policies=policies,
        slots_to_policies=slots_to_policies,
        max_years = 3
    )

if __name__ == '__main__':
    main()