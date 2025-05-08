import json
import numpy as np
import math
import cityflow
from numpy import exp
from gymnasium import Env, spaces
from os import path

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
                 replay_save_rate: int=32) -> None:            
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

    def reset(self, *, seed=None, options=None):
        """
        Reset the simulation. Compatibility with Gymnasium: accepts seed and options.
        Returns (observation, info).
        """
        self.step_count = 0
        self.episode += 1

        if ((self.episode-1) % self.replay_save_rate) == 0:
            self.eng.set_save_replay(True)
        else:
            self.eng.set_save_replay(False)


        if seed is not None:
            try:
                self.eng.set_random_seed(seed)
            except AttributeError:
                pass
        self.eng.reset()
        self.current_phase = self.phase_list[0]
        obs = self._get_state()
        info = {}

        return obs, info

    def step(self, action_idx):
        self.step_count += 1
        phase = self.phase_list[action_idx]
        self.current_phase = phase
        self.eng.set_tl_phase(self.intersection_id, phase)

        total_reward = 0.0
        for _ in range(self.phase_step):
            self.eng.next_step()
            total_reward += self._get_reward()
        total_reward /= self.phase_step

        obs = self._get_state()
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {}
        return obs, total_reward, terminated, truncated, info

    def _get_state(self):
        waiting = self.eng.get_lane_waiting_vehicle_count()
        vals = list(waiting.values())
        s = np.zeros(8, dtype=np.float32)
        s[0] = vals[1] + vals[15]
        s[1] = vals[3] + vals[13]
        s[2] = vals[0] + vals[14]
        s[3] = vals[2] + vals[12]
        s[4] = vals[1] + vals[0]
        s[5] = vals[14] + vals[15]
        s[6] = vals[3] + vals[2]
        s[7] = vals[12] + vals[13]
        return np.append(s, self.phase_list.index(self.current_phase))

    def _get_reward(self):
        waiting = self.eng.get_lane_waiting_vehicle_count()
        return -sum(waiting.values())
        # return (-2 / (1 + exp(-0.05 * waiting_cars_count))) + 1

    def render(self, mode='human'):
        pass

    def close(self):
        self.eng.close()
