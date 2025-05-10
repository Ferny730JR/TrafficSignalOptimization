import argparse
import json
import logging
import pandas as pd
import os
from stable_baselines3 import DQN
from environment import CityFlowEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cityflow_config.json', help='Config file')
    parser.add_argument('--dir', type=str, required=True, help='Path to directory where trained model is saved')
    parser.add_argument('--model', type=str, default="dqn_final.zip", help='Name of the file containing the model')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of steps for inference')
    parser.add_argument('--seed', type=int, default=42, help='Seed to be used by the engine')
    parser.add_argument('--save_replay', action='store_true', help='Whether to save CityFlow replay')
    parser.add_argument('--output', type=str, default='inference_scores.csv', help='Output CSV file')
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Initialize environment
    env = CityFlowEnv(args.config,
                      phase_step=6,
                      max_steps=512,
                      interval=1.0)
    model = DQN.load(os.path.join(args.dir,args.model), env=env)

    # Prepare tracking variables
    obs, _ = env.reset(seed=args.seed)
    t = 0
    rewards = []
    spawn_times = {}
    travel_durations = []

    # Inference loop: track spawn and finish times
    for _ in range(args.timesteps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        t += 1

        # Record newly spawned vehicles
        for vid in env.eng.get_vehicles(include_waiting=False):
            vinfo = env.eng.get_vehicle_info(vid)
            if vinfo['running'] and vid not in spawn_times:
                spawn_times[vid] = t

        # Identify finished vehicles
        active = set(env.eng.get_vehicles(include_waiting=False))
        finished = [vid for vid in list(spawn_times) if vid not in active]
        for vid in finished:
            duration = t - spawn_times.pop(vid)
            travel_durations.append(duration * 1.0)

        rewards.append(reward)
        logging.info(f"Step {t}/{args.timesteps} - Action: {action}, Reward: {reward:.4f}")

    # Account for vehicles still active at end
    for vid, start in spawn_times.items():
        duration = t - start
        travel_durations.append(duration * 1.0)

    # Compute and report average travel time
    avg_time = sum(travel_durations) / len(travel_durations) if travel_durations else 0.0
    print(f"Average travel time: {avg_time:.2f} seconds over {len(travel_durations)} vehicles")

    # Save travel durations
    out_df = pd.DataFrame({'travel_time_s': travel_durations})
    out_path = os.path.join(args.dir, 'travel_times.csv')
    out_df.to_csv(out_path, index=False)
    logging.info(f"Saved travel times to {out_path}")

if __name__ == '__main__':
    main()
