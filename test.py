import argparse
import json
import logging
import pandas as pd
from stable_baselines3 import DQN
from environment import CityFlowEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='global_config.json', help='Config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of steps for inference')
    parser.add_argument('--save_replay', action='store_true', help='Whether to save CityFlow replay')
    parser.add_argument('--output', type=str, default='inference_scores.csv', help='Output CSV file')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)
    cf_file = config['cityflow_config_file']
    cf = json.load(open(cf_file))
    if args.save_replay:
        cf['saveReplay'] = True
    json.dump(cf, open(cf_file, 'w'), indent=2)

    env = CityFlowEnv(config)
    model = DQN.load(args.model, env=env)

    obs, _ = env.reset()
    rewards = []
    for t in range(args.timesteps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        logging.info(f'Step {t+1}/{args.timesteps} - Action: {action}, Reward: {reward:.4f}')

    pd.DataFrame({'reward': rewards}).to_csv(args.output, index=False)
    logging.info(f'Inference complete. Rewards saved to {args.output}')

if __name__ == '__main__':
    main()
