import json
import numpy as np
import cityflow
from numpy import exp
from gymnasium import Env, spaces
from os import path
from typing import Callable, Dict, Any

def parse_roadnet(roadnet_file):
    """
    Parse CityFlow roadnet JSON to extract lane-phase information.
    Returns a dict mapping intersection_id to:
    - start_lane: list of lane IDs
    - end_lane: list of lane IDs
    - phase: list of phase indices
    - phase_startLane_mapping: {phase: [start_lanes]}
    - phase_roadLink_mapping: {phase: [(start_lane, end_lane)]}
    """
    roadnet = json.load(open(roadnet_file))
    lane_phase_info = {}
    for inter in roadnet['intersections']:
        if inter.get('virtual', False):
            continue
        info = {
            'start_lane': [],
            'end_lane': [],
            'phase': [],
            'phase_startLane_mapping': {},
            'phase_roadLink_mapping': {}
        }
        road_links = inter['roadLinks']
        roadLink_lane_pair = {i: [] for i in range(len(road_links))}
        for i, rl in enumerate(road_links):
            for ll in rl['laneLinks']:
                sl = f"{rl['startRoad']}_{ll['startLaneIndex']}"
                el = f"{rl['endRoad']}_{ll['endLaneIndex']}"
                info['start_lane'].append(sl)
                info['end_lane'].append(el)
                roadLink_lane_pair[i].append((sl, el))
        info['start_lane'] = sorted(set(info['start_lane']))
        info['end_lane'] = sorted(set(info['end_lane']))

        phases = inter['trafficLight']['lightphases']
        for pid, p in enumerate(phases):
            start_lanes = []
            lane_pairs = []
            for rl_idx in p['availableRoadLinks']:
                lane_pairs.extend(roadLink_lane_pair[rl_idx])
                sl0 = roadLink_lane_pair[rl_idx][0][0]
                if sl0 not in start_lanes:
                    start_lanes.append(sl0)
            info['phase'].append(pid)
            info['phase_startLane_mapping'][pid] = start_lanes
            info['phase_roadLink_mapping'][pid] = lane_pairs

        lane_phase_info[inter['id']] = info
    return lane_phase_info


# --- State function definitions ---
def state_wait_count(env: 'CityFlowEnv') -> np.ndarray:
    """Feature 1: number of waiting vehicles per start lane"""
    waiting = env.eng.get_lane_waiting_vehicle_count()
    s = [ waiting.get(lane, 0) for lane in env.start_lane ]
    return np.append(s, env.phase_list.index(env.current_phase)).astype(np.float32)


def state_end_count(env: 'CityFlowEnv') -> np.ndarray:
    """Feature 2: number of vehicles in end lanes"""
    total = env.eng.get_lane_vehicle_count()
    end_lanes = env.lane_phase_info[env.intersection_id]['end_lane']
    s = [ total.get(lane, 0) for lane in end_lanes ]
    return np.append(s, env.phase_list.index(env.current_phase)).astype(np.float32)


def state_avg_wait(env: 'CityFlowEnv') -> np.ndarray:
    """Feature 3: average wait time (in steps) per start lane"""
    vehicle_ids = env.eng.get_lane_vehicles()
    s = []
    for lane in env.start_lane:
        # Count waiting vehicles (speed < 0.1)
        for vid in vehicle_ids[lane]:
            info = env.eng.get_vehicle_info(vid)
            if info['running'] and float(info['speed']) < 0.1:
                env.wait_spawn_times[lane][vid] = env.wait_spawn_times[lane].get(vid, 0) + 1

        # Prune moving or departed vehicles
        active = set(vehicle_ids[lane])
        for vid in list(env.wait_spawn_times[lane].keys()):
            info = env.eng.get_vehicle_info(vid)
            if vid not in active or float(info['speed']) >= 0.1:
                del env.wait_spawn_times[lane][vid]
        
        # Compute the average wait time (in engine steps)
        counts = list(env.wait_spawn_times[lane].values())
        avg = sum(counts) / len(counts) if counts else 0.0
        s.append(avg)
    return np.append(s, env.phase_list.index(env.current_phase)).astype(np.float32)


# --- Reward function definitions ---
def reward_wait_sum(env: 'CityFlowEnv') -> float:
    """Reward 1: negative sum of waiting vehicles"""
    waiting = env.eng.get_lane_waiting_vehicle_count()
    return -sum(waiting.values())


def reward_avg_wait(env: 'CityFlowEnv') -> float:
    """Reward 2: negative average of per-lane average wait times"""
    state = state_avg_wait(env)[:-1]  # drop phase index
    return -np.mean(state)


class CityFlowEnv(Env):
    """A Gymnasium environment wrapping CityFlow for traffic signal control via
    DQN.
    Action: discrete phase index.
    State: Stacked waiting counts + current phase.

    Args:
        config (str, optional): Path to the CityFlow main configuration JSON
                                file. Defaults to 'cityflow_config.json'.
        roadnet (str, optional): Filename of the road network JSON within
                                 `filedir`. Defaults to 'roadnet.json'.
        flow (str, optional): Filename of the traffic flow JSON within
                              `filedir`. Defaults to 'flow.json'.
        filedir (str, optional): Directory containing the CityFlow JSON files
                                 (`config`, `roadnet`, `flow`). Defaults to '.'.
        phase_step (int, optional): Number of simulation steps the environment
                                    remains in a selected traffic-light phase
                                    before allowing a new action. Defaults to 6.
        max_steps (int, optional): Maximum number of step calls per episode 
                                   before termination. Defaults to 512.
        thread_num (int, optional): Number of threads the CityFlow engine uses
                                    for parallel simulation. Defaults to 1.
        interval (float, optional): Simulation time interval (in seconds) per
                                    engine step. Defaults to 1.0.
        seed (int, optional): Random seed for reproducible CityFlow simulations.
                              Defaults to 0.
        save_replay (bool, optional): Whether to save replay logs of the
                                      simulation to disk. Defaults to True.
        replay_log_dir (str, optional): Directory where replay logs 
                                        (`roadnetLogFile` and `replayLogFile`)
                                        are stored. Defaults to 'replay_logs'.
        replay_save_rate (int, optional): Episode interval at which to enable
                                          saving replays (e.g., every N 
                                          episodes). Defaults to 32.
        state_fn (Callable, optional): function to compute state.
        reward_fn (Callable, optional): function to compute single-step reward.
    """
    def __init__(self, config: str='cityflow_config.json',
                 roadnet: str='roadnet.json',
                 flow='flow.json',
                 filedir='./',
                 phase_step: int=6,
                 max_steps: int=512,
                 thread_num: int=1,
                 interval: float=1.0,
                 seed: int=0,
                 save_replay=True,
                 replay_log_dir='replay_logs',
                 replay_save_rate: int=32,
                 state_fn: Callable[['CityFlowEnv'], np.ndarray] = state_wait_count,
                 reward_fn: Callable[['CityFlowEnv'], float] = reward_wait_sum
                ) -> None:
        super().__init__()

        # Load config data if it exists, create it otherwise
        try:
            with open(config, 'r') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            config_data = {} # Either doesn't exist or file is empty

        # Update values in the config file
        config_data['interval']       = interval
        config_data['seed']           = seed
        config_data['dir']            = filedir
        config_data['roadnetFile']    = roadnet
        config_data['flowFile']       = flow
        config_data['rlTrafficLight'] = True # always true since, well, we're doing RL
        config_data['saveReplay']     = save_replay
        config_data['roadnetLogFile'] = path.join(replay_log_dir, 'roadnet_log.json')
        config_data['replayLogFile']  = path.join(replay_log_dir, 'replay.txt')

        # Write the new json file
        with open(config, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Declare environment variable
        self.phase_step       = phase_step
        self.step_count       = 0
        self.max_steps        = max_steps
        self.episode          = 0
        self.replay_save_rate = replay_save_rate

        # Load and parse roadnet
        self.lane_phase_info = parse_roadnet(path.join(filedir, roadnet))
        self.intersection_id = list(self.lane_phase_info.keys())[0]
        self.phase_list = self.lane_phase_info[self.intersection_id]['phase']
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']

        # Define state & action sizes
        self.state_size = len(self.lane_phase_info[self.intersection_id]['start_lane']) + 1
        self.action_size = len(self.phase_list)

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_size)

        # Initialize CityFlow engine
        self.eng = cityflow.Engine(config, thread_num=thread_num)
        self.current_phase = self.phase_list[0]

        # State/reward functions
        self.state_fn = state_fn
        self.reward_fn = reward_fn

        # Initialize trackers (used by feature 3)
        self.wait_spawn_times = {lane: {} for lane in self.start_lane}

    def reset(self, *, seed=None, options=None):
        """
        Reset the simulation. Compatibility with Gymnasium: accepts seed and options.
        Returns (observation, info).
        """
        self.step_count = 0
        self.episode += 1

        # Save this episode's replay?
        self.eng.set_save_replay((self.episode-1) % self.replay_save_rate == 0)

        # Set the seed if specified
        if seed is not None:
            try:
                self.eng.set_random_seed(seed)
            except AttributeError:
                pass
        self.eng.reset()
        self.current_phase = self.phase_list[0]

        # Clear wait trackers
        self.wait_spawn_times = {lane: {} for lane in self.start_lane} # for feature 3
        obs = self.state_fn(self)

        return obs, {}

    def step(self, action_idx):
        self.step_count += 1
        self.current_phase = self.phase_list[action_idx]
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)

        total_reward = 0.0
        for _ in range(self.phase_step):
            self.eng.next_step()
            total_reward += self.reward_fn(self)
        total_reward /= self.phase_step

        # Get state
        obs = self.state_fn(self)
        terminated = self.step_count >= self.max_steps
        return obs, total_reward, terminated, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.eng.close()
