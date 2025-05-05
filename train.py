import argparse
import json
import os
import logging
from tqdm import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment import CityFlowEnv

class ProgressBarCallback(BaseCallback):
    """
    A callback that displays a tqdm progress bar for training steps.
    """
    def __init__(self, total_timesteps: int, disable: bool):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.disable = disable

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", disable=self.disable)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

class RolloutCallback(BaseCallback):
    """
    Logs episode-wise statistics: steps, total reward, average reward/step, etc.
    """
    def __init__(self, disable: bool):
        super().__init__(verbose=0)
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_count = 0
        self.disable = disable

    def _on_step(self) -> bool:
        if self.disable:
            pass # change to return True
            
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                ep_rew = ep_info['r']
                ep_len = ep_info['l']
                self.ep_rewards.append(ep_rew)
                self.ep_lengths.append(ep_len)
                self.ep_count += 1
                avg_rew_step = ep_rew / ep_len if ep_len > 0 else 0.0
                print(avg_rew_step, flush=True)
                return True
                avg_rew_epis = sum(self.ep_rewards) / len(self.ep_rewards)
                if len(self.ep_rewards) >= 2 and self.ep_rewards[-2] != 0:
                    perc_change = ((ep_rew - self.ep_rewards[-2]) / self.ep_rewards[-2]) * 100
                    perc_str = f"{perc_change:.2f}%"
                else:
                    perc_str = "0 Division Error"
                print('#' + '-'*30 + '#')
                print(f"# Episode:          {self.ep_count}")
                print(f"# Episode Steps:    {ep_len}")
                print(f"# Total Reward:     {ep_rew:.2f}")
                print(f"# Avg Reward/Step:  {avg_rew_step:.2f}")
                print(f"# Avg Reward/Epis:  {avg_rew_epis:.2f}")
                print(f"# Episode % Change: {perc_str}")
                print('#' + '-'*30 + '#')
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='global_config.json', help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps to train')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--buffer_size', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_freq', type=int, default=4, help='Training frequency (in steps)')
    parser.add_argument('--target_update_interval', type=int, default=1000)
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--exploration_final_eps', type=float, default=0.05)
    parser.add_argument('--checkpoint_freq', type=int, default=100_000)
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Tensorboard log dir')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Verbose output')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Load and update CityFlow config
    with open(args.config) as f:
        config = json.load(f)
    cf_file = config['cityflow_config_file']
    cf = json.load(open(cf_file))
    # cf.update({'saveReplay': False, 'printLog': False, 'logFile': None})
    # json.dump(cf, open(cf_file, 'w'), indent=2)

    # Initialize environment with Monitor for episode logging
    raw_env = CityFlowEnv(config)
    env = Monitor(raw_env)

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.model_dir,
        name_prefix='dqn_model'
    )
    progress_callback = ProgressBarCallback(args.timesteps, disable=not args.verbose)
    rollout_callback = RolloutCallback(disable=not args.verbose)

    # Initialize DQN agent
    model = DQN(
        'MlpPolicy', env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=args.log_dir,
        verbose=0
    )

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, progress_callback, rollout_callback]
    )

    # Save final model
    model.save(os.path.join(args.model_dir, 'dqn_final'))

if __name__ == '__main__':
    main()
