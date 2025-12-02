#!/usr/bin/env python3

import os
import re
from typing import Dict

import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import socket
import time



# -----------------------
# Utility
# -----------------------

# Checks the largest existing checkpoint folder, increments number 
# and creates new folder for new training
def get_new_checkpoint_dir(base_name="checkpoints"):
    existing = [d for d in os.listdir("..") if re.match(rf"{base_name}_\d+", d)]
    new_idx = max([int(re.findall(r"\d+", d)[-1]) for d in existing], default=0) + 1
    new_dir = f"{base_name}_{new_idx:02d}"
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

# Between training episodes sumo is restarted
# This method checks if traci-sumo connection exists after restart
def wait_for_port(port, host="localhost", timeout=5.0):
    """Wait until SUMO opens the TraCI port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False

# -----------------------
# Traffic Light Environment
# Gym-like environment stepping the sumo environment, collecting observation and reward, and applying the action
# -----------------------
class TrafficLightEnv(gym.Env):

    def __init__(self, sumocfg="rl_threelane_cross.sumocfg", use_gui=False,
                 step_length=2.0, decision_interval=1, port=833):
        super().__init__()
        self.sumocfg = sumocfg
        self.use_gui = use_gui
        self.step_length = step_length
        self.decision_interval = decision_interval
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.port = port
        traci.start([
            self.sumo_binary,
            "-c", self.sumocfg,
            "--start",
            "--end", "10000",
            #"--quit-on-end",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--waiting-time-memory", "30",
            "--scale","2.0",
            "--step-length", str(self.step_length)
        ], port=port)
        wait_for_port(self.port)

        self.tls_ids = list(traci.trafficlight.getIDList())
        self.current_phase = 0
        self.number_of_states = 4
        self.min_state_length = 5.0
        self.state_start = 0.0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(99,), dtype=np.float32)
        self.departed_count = 0
        self.arrived_count = 0
        self.arrived_id = []
        self.light_states = {tls: "red" for tls in self.tls_ids}
        self.prev_speeds: Dict[str, float] = {}
        self.step_count = 0
        self.last_phases: Dict[str, int] = {}
        self.last_switch_times: Dict[str, int] = {}

        self.metrics = {k: [] for k in
                        ["step", "avg_speed", "total_waiting", "total_queue", "reward", "throughput_vph"]}
        self.veh_past_count = 0

        self.lane_detectors = {
            "N_0": ("north_right_close", "north_right_far"),
            "N_1": ("north_middle_close", "north_middle_far"),
            "N_2": ("north_left_close", "north_left_far"),
            "S_0": ("south_right_close", "south_right_far"),
            "S_1": ("south_middle_close", "south_middle_far"),
            "S_2": ("south_left_close", "south_left_far"),
            "W_0": ("west_right_close", "west_right_far"),
            "W_1": ("west_middle_close", "west_middle_far"),
            "W_2": ("west_left_close", "west_left_far"),
            "E_0": ("east_right_close", "east_right_far"),
            "E_1": ("east_middle_close", "east_middle_far"),
            "E_2": ("east_left_close", "east_left_far"),
        }

        self.induction_detectors = [
            "east_right", "east_middle", "east_left",
            "north_right", "north_middle", "north_left",
            "south_right", "south_middle", "south_left",
            "west_right", "west_middle", "west_left"
        ]

    # Reset SUMO environment
    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()

        traci.start([
            self.sumo_binary,
            "-c", self.sumocfg,
            "--start",
            "--end", "10000",
            #"--quit-on-end",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--waiting-time-memory", "30",
            "--scale","2.0",
            "--step-length", str(self.step_length)
        ], port=self.port)
        wait_for_port(self.port)

        warmup_time = 300  # e.g., 5 minutes
        for _ in range(int(warmup_time / (self.step_length*4))):
            traci.simulationStep()

        for tls in self.tls_ids:
            try:
                traci.trafficlight.setPhase(tls, 0)
            except Exception:
                pass
            self.light_states[tls] = "red"

        self.step_count = 0
        self.departed_count = 0
        self.arrived_count = 0
        self.arrival_durations = []
        self.prev_speeds.clear()

        for tls in self.tls_ids:
            self.last_phases[tls] = 0
            self.last_switch_times[tls] = 0

        return self._get_observation(), {}

    # Step SUMO environment,call observation collection and reward calculation
    def step(self, action):
        self._apply_action(action)
        
        traci.simulationStep()
  
        self.step_count += 1

        reward = self._compute_reward()
        obs = self._get_observation()

        done = traci.simulation.getMinExpectedNumber() <= 0
        info = {}

        return obs, float(reward), done, False, info

    # Applies the agents action in the SUMO environment
    def _apply_action(self, action):        
        if action == 1 and traci.simulation.getTime() - self.state_start > self.min_state_length:
            self.current_phase = (self.current_phase + 1) % self.number_of_states
            tls_id = self.tls_ids[0]
            traci.trafficlight.setPhase(tls_id, self.current_phase)
            self.state_start = traci.simulation.getTime()
    # Collects observation data from SUMO simulation
    def _get_observation(self):
        obs = np.zeros(99, dtype=np.float32)

        close_length = 20.0
        far_length = 60.0
        avg_vehicle_length = 6.0
        vehicle_top_speed = 13.89
        normalizing_factor_close = np.ceil(close_length/avg_vehicle_length)
        normalizing_factor_far = np.ceil(far_length/avg_vehicle_length)
        idx = 0

        # E2 lane area detectors
        for lane, (close_det, far_det) in self.lane_detectors.items():
            occ_close = traci.lanearea.getLastStepOccupancy(close_det)/100
            occ_far = traci.lanearea.getLastStepOccupancy(far_det)/100
            halted_close = traci.lanearea.getLastStepHaltingNumber(close_det)/normalizing_factor_close  # normalized
            halted_far = traci.lanearea.getLastStepHaltingNumber(far_det)/normalizing_factor_far
            speed_close = traci.lanearea.getLastStepMeanSpeed(close_det)/vehicle_top_speed
            speed_far = traci.lanearea.getLastStepMeanSpeed(far_det)/vehicle_top_speed

            obs[idx:idx+2] = [occ_close, occ_far]; idx += 2
            obs[idx:idx+2] = [halted_close, halted_far]; idx += 2
            obs[idx:idx+2] = [speed_close, speed_far]; idx += 2

        # E1 induction loops
        for det in self.induction_detectors:
            try:
                veh_count = traci.inductionloop.getLastStepVehicleNumber(det)
                mean_speed = traci.inductionloop.getLastStepMeanSpeed(det)/vehicle_top_speed
            except traci.exceptions.TraCIException:
                veh_count = 0
                mean_speed = 0.0
            obs[idx] = veh_count; idx += 1
            obs[idx] = mean_speed; idx += 1

        # Current traffic light phase
        tls_id = self.tls_ids[0]
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
        time_since_start = traci.simulation.getTime() - self.state_start
        time_since_norm = np.clip(time_since_start / phase_duration, 0.0, 1.0)
        remaining_norm = np.clip((self.min_state_length - time_since_start)/self.min_state_length, 0.0, 1.0)

        obs[idx:idx+3] = [current_phase, time_since_norm, remaining_norm]
        
        return obs

    # Computes the reward based on the SUMO simulation
    def _compute_reward(self):

        close_length = 20.0
        far_length = 60.0
        avg_vehicle_length = 6.0
        vehicle_top_speed = 13.89
        normalizing_factor_close = np.ceil(close_length/avg_vehicle_length)
        normalizing_factor_far = np.ceil(far_length/avg_vehicle_length)

        veh_ids = traci.vehicle.getIDList()
        num_vehicles = max(len(veh_ids), 1)

        # Queue length
        total_halted = sum(traci.lanearea.getLastStepHaltingNumber(det)
                       for det_pair in self.lane_detectors.values()
                       for det in det_pair)
        queue_norm = np.clip(total_halted / (len(self.lane_detectors)*(normalizing_factor_close+normalizing_factor_far)), 0.0, 1.0)  # max 10 vehicles per detector

        # Waiting time
        waiting_time_total = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        max_wait_possible = num_vehicles * 300.0  # assume 5 min waiting per vehicle
        wait_norm = waiting_time_total / max_wait_possible  # 0..1

        # Throughput
        throughput = self.arrived_count / max(self.departed_count, 1)
        throughput_norm = min(throughput, 1.0)  # bound to 0..1

        # Phase change penalty
        tls_id = self.tls_ids[0]
        current_phase = traci.trafficlight.getPhase(tls_id)
        phase_change_penalty = 1.0 if hasattr(self, "last_phase") and current_phase != self.last_phase else 0.0
        self.last_phase = current_phase

        # CO2 Emissions
        total_co2 = sum(traci.vehicle.getCO2Emission(v) for v in veh_ids)
        co2_norm = np.clip(total_co2 / (num_vehicles * 50.0), 0.0, 1.0)

        # Weights (all roughly similar magnitude)
        w_queue = -0.5
        w_wait = -1.0
        w_throughput = 1.0
        w_phase = -0.1
        w_co2 = -0.25

        # Compute reward
        reward = (
            w_queue * queue_norm +
            w_wait * wait_norm +
            w_throughput * throughput_norm +
            w_phase * phase_change_penalty +
            w_co2 * co2_norm
        )

        return reward

    # Close traci-sumo connection
    def close(self):
        if traci.isLoaded():
            traci.close()

# -----------------------
# Main
# -----------------------

def main():
    SUMO_CFG = "sumo_cfgs/rl_threelane_cross_uneven_traffic.sumocfg"
    USE_GUI = False
    TOTAL_TIMESTEPS = 1000
    BASE_PORT = 8810

    #Create new checkpoint directory for saving model weigths
    CHECKPOINT_DIR = get_new_checkpoint_dir("checkpoints")
    print(f"Saving checkpoints to: {CHECKPOINT_DIR}")

    #Initialize Gym-like environment
    env = TrafficLightEnv(sumocfg=SUMO_CFG, use_gui=USE_GUI, port=BASE_PORT)

    #Register checkpoint saving callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_traffic_light"
    )

    #Initialize DDQN model 
    #StabelBaselines DQN implementation acts as a DDQN, unless parameterised otherwise
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        device="cpu",
    )

    #Start the learning process
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback])
    #Save the final model
    model.save(os.path.join(CHECKPOINT_DIR, "dqn_traffic_light_final"))
    print(f"Training finished. Model saved in {CHECKPOINT_DIR}")
    #Shut down the environment
    env.close()


if __name__ == "__main__":
    main()
