import os
import wandb
from tqdm import tqdm

from network import config
from network import network_policy
from network import parameter_provider
from environment import diplomacy_state
from environment import game_runner
from environment import observation_utils as utils

# To get around hyphen in welare-diplomacy
import sys
sys.path.append('/Users/hannaherlebach/research/welfare_diplomacy_baselines/welfare-diplomacy')
from diplomacy.engine.game import Game

class WelfareDiplomacyGameRunner:
    """Runs the provided DeepMind agents on a game of Welfare Diplomacy."""
    def __init__(self, SL=True, welfare=True):

        if welfare:
            game_instance = diplomacy_state.BaselineGame()
            assert game_instance.welfare
        else:
            game_instance = Game()
        self.game = game_instance
        
        print(game_instance.rules)

        self.SL = SL
        self.file_path = '/Users/hannaherlebach/research/diplomacy_parameters/'

        # wandb.init(project='welfare-diplomacy-v0')

    def get_parameter_provider(self) -> parameter_provider.ParameterProvider:
        if self.SL:
            params = 'sl_params.npz'
        else:
            params = 'fppi2_params.npz'
        with open(os.path.join(self.file_path, params), 'rb') as f:
            provider = parameter_provider.ParameterProvider(f)
        return provider

    def get_diplomacy_state(self) -> diplomacy_state.DiplomacyState:
        state = diplomacy_state.WelfareDiplomacyState(self.game)
        return state


    def run_network(self):

        diplomacy_state_instance = self.get_diplomacy_state()

        network_info = config.get_config()
        provider = self.get_parameter_provider()
        network_handler = parameter_provider.SequenceNetworkHandler(
            network_cls=network_info.network_class,
            network_config=network_info.network_kwargs,
            parameter_provider=provider,
            rng_seed=42)

        network_policy_instance = network_policy.Policy(
            network_handler=network_handler,
            num_players=7,
            temperature=0.1,
            calculate_all_policies=True)

        trajectory = game_runner.run_game(state=diplomacy_state_instance,
                                        policies=(network_policy_instance,), 
                                        slots_to_policies=[0] * 7, 
                                        max_length=20,
                                        disband_after_k_turns=True,
                                        k=10)
        
        if wandb.run is not None:
            wandb.finish() 
    
        return trajectory
    
if __name__ == '__main__':
    game_instance = WelfareDiplomacyGameRunner(SL=False)
    trajectory = game_instance.run_network()
    print("Trajectory", trajectory)
    print(trajectory.observations[-1].season)