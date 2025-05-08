import argparse
import json
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from environment import CityFlowEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cityflow_config.json', help='Config file')
    parser.add_argument('--dir', type=str, required=True, help='Path to directory where trained model is saved')
    parser.add_argument('--model', type=str, default="dqn_final.zip", help='Name of the file containing the model')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of steps for inference')
    parser.add_argument('--save_replay', action='store_true', help='Whether to save CityFlow replay')
    parser.add_argument('--output', type=str, default='inference_scores.csv', help='Output CSV file')
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    env = CityFlowEnv(args.config,
                      phase_step=6,
                      max_steps=512,
                      interval=1.0)
    model = DQN.load(os.path.join(args.dir,args.model), env=env)

    obs, _ = env.reset()
    rewards = []
    t = 0
    spawn_times = {}
    travel_durations = []
    for t in range(args.timesteps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        t += 1

        # Record newly spawned vehicles
        for vid in env.eng.get_vehicles(include_waiting=False):
            info = env.eng.get_vehicle_info(vid)
            if info['running'] and vid not in spawn_times:
                spawn_times[vid] = t
        
        # check which vehicles have finished
        finished = [
            vid for vid, start in spawn_times.items()
            if vid not in env.eng.get_vehicles(include_waiting=False)
        ]
        for vid in finished:
            duration_steps = t - spawn_times.pop(vid)
            # convert to seconds using the environment's interval
            travel_durations.append(duration_steps * 1.0)

        rewards.append(reward)
        logging.info(f'Step {t+1}/{args.timesteps} - Action: {action}, Reward: {reward:.4f}')

    avg_time = sum(travel_durations) / len(travel_durations)
    print(f"Average travel time: {avg_time:.2f} seconds over {len(travel_durations)} vehicles")

    pd.DataFrame({'travel_time_s': travel_durations}) \
      .to_csv(os.path.join(args.dir, 'travel_times.csv'), index=False)

if __name__ == '__main__':
    main()
