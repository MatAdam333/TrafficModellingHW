#!/usr/bin/env python3

##################################################################################################################
### NOTE: At the beginning of the project this code was written                                                ###
### with the possibility in mind, that it may need to control multiple traffic lights.                         ###
### Later this idea was abandoned, however an artifact remained in the code,                                   ###
### where traffic light ids are handeld as an array, instead of a single value, and later indexed.             ###
### Due to project deadline, and the saying "If it is not broken, do not fix it", the artifact is not modified.###
##################################################################################################################
import argparse
import os
import random
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import traci
except Exception:
    traci = None

try:
    import gymnasium as gym
    import stable_baselines3 as sb3
    HAS_RL = True
except Exception:
    HAS_RL = False

CURRENT_PHASE = 0
NUMBER_OF_STATES = 4
MIN_STATE_LENGTH = 5.0
STATE_START = 0.0

#Open traci-sumo connection
def start_sumo(sumocfg: str, use_gui: bool = False, port: int = 8813) -> None:
    if traci is None:
        raise RuntimeError("TraCI / SUMO not available.")
    sumo_binary = os.environ.get("SUMO_BINARY", "sumo-gui" if use_gui else "sumo")
    traci.start([sumo_binary, "-c", sumocfg])
#Close traci-sumo connection
def stop_sumo():
    if traci.isLoaded():
        traci.close()

#Get all available traffic light IDs
def get_all_tls_ids() -> List[str]:
    return traci.trafficlight.getIDList()

#Get the Traffic Light info from sumo based on TL ID
def tls_phase_info(tls_id):
    try:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    except Exception as e:
        print(f"Could not get logic for {tls_id}: {e}")
        return {"num_phases": 0, "phases": [], "lanes": []}
    phases = [(p.state, p.duration) for p in logic.phases]
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    return {"num_phases": len(phases), "phases": phases, "lanes": lanes}

#Assemble the observation vector based on the sumo simulation
def get_observation(tls_id):  
    # E2 lane area detectors (close + far) for each lane
    e2_detectors = {
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
    
    # E1 induction loops
    e1_detectors = [
            "east_right", "east_middle", "east_left",
            "north_right", "north_middle", "north_left",
            "south_right", "south_middle", "south_left",
            "west_right", "west_middle", "west_left"
        ]

    obs = np.zeros(99, dtype=np.float32)

    close_length = 20.0
    far_length = 60.0
    avg_vehicle_length = 6.0
    vehicle_top_speed = 13.89
    normalizing_factor_close = np.ceil(close_length/avg_vehicle_length)
    normalizing_factor_far = np.ceil(far_length/avg_vehicle_length)
    idx = 0

    # E2 lane area detectors
    for lane, (close_det, far_det) in e2_detectors.items():
        occ_close = traci.lanearea.getLastStepOccupancy(close_det)/100
        occ_far = traci.lanearea.getLastStepOccupancy(far_det)/100
        halted_close = traci.lanearea.getLastStepHaltingNumber(close_det)/normalizing_factor_close  # normalized
        halted_far = traci.lanearea.getLastStepHaltingNumber(far_det)/normalizing_factor_far
        speed_close = traci.lanearea.getLastStepMeanSpeed(close_det)/vehicle_top_speed
        speed_far = traci.lanearea.getLastStepMeanSpeed(far_det)/vehicle_top_speed

        obs[idx:idx+2] = [occ_close, occ_far]; idx += 2
        obs[idx:idx+2] = [halted_close, halted_far]; idx += 2
        obs[idx:idx+2] = [speed_close, speed_far]; idx += 2

    # E1 features: vehicle count, mean speed
    for det in e1_detectors:
            try:
                veh_count = traci.inductionloop.getLastStepVehicleNumber(det)
                mean_speed = traci.inductionloop.getLastStepMeanSpeed(det)/vehicle_top_speed
            except traci.exceptions.TraCIException:
                veh_count = 0
                mean_speed = 0.0
            obs[idx] = veh_count; idx += 1
            obs[idx] = mean_speed; idx += 1

    # Traffic light phase
    current_phase = traci.trafficlight.getPhase(tls_id)
    obs[idx]=current_phase; idx+=1

    phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
    phase_ellapsed_time = np.clip((traci.simulation.getTime() - STATE_START)/phase_duration,0,1)
    obs[idx]= phase_ellapsed_time; idx+=1
    remaining_norm = np.clip((MIN_STATE_LENGTH - phase_ellapsed_time)/MIN_STATE_LENGTH, 0.0, 1.0)
    obs[idx]= remaining_norm
    return obs

#Apply the agents action in the SUMO simulation
def apply_action(tls_ids, action):
    global CURRENT_PHASE, STATE_START

    # Iterate over all TLS IDs and corresponding actions
    for i, tls_id in enumerate(tls_ids):
        current_time = traci.simulation.getTime()

        # Only change phase if minimum time elapsed
        #NOTE: This part is responsible for enforcing intergreen times
        if action == 1 and current_time - STATE_START > MIN_STATE_LENGTH:
            CURRENT_PHASE = (CURRENT_PHASE + 1) % NUMBER_OF_STATES
            traci.trafficlight.setPhase(tls_id, CURRENT_PHASE)
            STATE_START = current_time

#Collecting runtime metrics to evaluate agent
def collect_metrics(prev_arrived):
    veh_ids = traci.vehicle.getIDList()
    avg_speed = np.mean([traci.vehicle.getSpeed(v) for v in veh_ids]) if veh_ids else 0.0
    avg_wait = np.mean([traci.vehicle.getWaitingTime(v) for v in veh_ids]) if veh_ids else 0.0
    queue_len = sum([traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList()])

    total_co2 = sum([traci.vehicle.getCO2Emission(v) for v in veh_ids]) if veh_ids else 0.0
    total_co2_gph = total_co2 / 1000.0 * 3600.0

    # Vehicles that finished this step
    arrived_total = traci.simulation.getArrivedNumber() + prev_arrived
    throughput_per_hour = (arrived_total / (traci.simulation.getTime() + 1e-6)) * 3600.0
    return avg_speed, avg_wait, queue_len, throughput_per_hour, arrived_total,total_co2_gph

#Plot the metrics collected during runtime
def plot_metrics(control_mode,metrics_rl,metrics_classic):
    if control_mode == "rl":
        mode = ["RL"]
        metrics = [metrics_rl]
    elif control_mode == "classic":
        mode = ["Timed"]
        metrics = [metrics_rl]
    else:
        mode=["RL","Classic"]
        metrics = [metrics_rl,metrics_classic]
    steps = np.arange(len(metrics[0]["avg_speed"]))
    window = 250                   # smooth over 250 steps
    kernel = np.ones(window) / window

    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Traffic Performance Metrics ({control_mode} mode)", fontsize=14)

    # 1 — Avg Speed
    plt.subplot(3, 2, 5)
    if len(metrics) > 1:
        avg_speed_smooth = np.convolve(metrics[1]["avg_speed"], kernel, mode='same')
        plt.plot(steps, avg_speed_smooth, label=f"{mode[1]} Avg Speed (m/s)", alpha=0.5, color="blue")
    avg_speed_smooth = np.convolve(metrics[0]["avg_speed"], kernel, mode='same')
    plt.plot(steps, avg_speed_smooth, label=f"{mode[0]} Avg Speed (m/s)", color="red")
    plt.xlabel("Step")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()

    # 2 — Avg Wait
    plt.subplot(3, 2, 2)
    if len(metrics) > 1:
        avg_wait_smooth = np.convolve(metrics[1]["avg_wait"], kernel, mode='same')
        plt.plot(steps,avg_wait_smooth, label=f"{mode[1]} Avg Waiting Time (s)", color="blue", alpha=0.5)
    avg_wait_smooth = np.convolve(metrics[0]["avg_wait"], kernel, mode='same')
    plt.plot(steps, avg_wait_smooth, label=f"{mode[0]} Avg Waiting Time (s)", color="red")
    plt.xlabel("Step")
    plt.ylabel("Waiting Time (s)")
    plt.grid(True)
    plt.legend()

    # 3 — Queue Length
    plt.subplot(3, 2, 3)
    if len(metrics) > 1:
        queue_len_smooth = np.convolve(metrics[1]["queue_len"], kernel, mode='same')
        plt.plot(steps, queue_len_smooth, label=f"{mode[1]} Queue Length", color="blue", alpha=0.5)
    queue_len_smooth = np.convolve(metrics[0]["queue_len"], kernel, mode='same')
    plt.plot(steps, queue_len_smooth, label=f"{mode[0]} Queue Length", color="red")
    plt.xlabel("Step")
    plt.ylabel("Stopped Vehicles")
    plt.grid(True)
    plt.legend()

    # 4 — Throughput
    plt.subplot(3, 2, 4)    
    if len(metrics) > 1:
        throughput_smooth = np.convolve(metrics[1]["throughput"], kernel, mode='same')
        plt.plot(steps, throughput_smooth, label=f"{mode[1]} Throughput (veh/hour)", color="blue", alpha=0.5)
    throughput_smooth = np.convolve(metrics[0]["throughput"], kernel, mode='same')
    plt.plot(steps, throughput_smooth, label=f"{mode[0]} Throughput (veh/hour)", color="red")
    plt.xlabel("Step")
    plt.ylabel("Vehicles/hour")
    plt.grid(True)
    plt.legend()

    # 5 — CO2 Emissions
    plt.subplot(3, 2, 1)

    if len(metrics) > 1:
        co2_raw = np.array(metrics[1]["co2"])
        if len(co2_raw) >= window:
            co2_smooth = np.convolve(co2_raw, kernel, mode='same')
        else:
            co2_smooth = co2_raw  # not enough points to smooth
        plt.plot(steps, co2_smooth, label=f"{mode[1]} CO₂ Emissions (g/h)", color="blue", alpha=0.5)

    co2_raw = np.array(metrics[0]["co2"])
    if len(co2_raw) >= window:
        kernel = np.ones(window) / window
        co2_smooth = np.convolve(co2_raw, kernel, mode='same')
    else:
        co2_smooth = co2_raw  # not enough points to smooth
    plt.plot(steps, co2_smooth, label=f"{mode[0]} CO₂ Emissions (g/h)", color="red")
    
    plt.xlabel("Step")
    plt.ylabel("CO₂ (g/h)")
    plt.grid(True)
    plt.legend()

    #6 - CO2 Sum
    plt.subplot(3, 2, 6)
    co2_sums = []
    if len(metrics) > 1:
        co2_sum = np.sum(metrics[1]["co2"])
        co2_sums.insert(0,co2_sum)
    co2_sum = np.sum(metrics[0]["co2"])
    co2_sums.insert(0,co2_sum)
    plt.bar(mode,co2_sums) 
    plt.xlabel("Mode")
    plt.ylabel("Cumulative CO₂ (g)")
    plt.legend()

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#Main loop of the simulation
#Runs the simulation according to the start parameters
#Collects metrics, and prints basic info to console during runtime
def run_simulation(metrics,tls_ids,control_mode,steps,rl_model=None):
    arrived_total = 0
    try:
        for step in range(steps):
            if step % 10 == 0:
                if control_mode == "classic":
                    actions = []
                    for tl in tls_ids:
                        phase_length = traci.trafficlight.getPhaseDuration(tl)
                        if((traci.simulation.getTime() - STATE_START) < phase_length):
                            actions.append(0)
                        else:
                            actions.append(1)
                else:  # RL mode
                    obs = get_observation(tls_ids[0])
                    actions, _states = rl_model.predict(obs, deterministic=True)                    
                    actions = [actions]
                apply_action(tls_ids, actions[0])

            traci.simulationStep()

            avg_speed, avg_wait, queue_len, throughput, arrived_total,co2_raw = collect_metrics(arrived_total)

            
            metrics["avg_speed"].append(avg_speed)
            metrics["avg_wait"].append(avg_wait)
            metrics["queue_len"].append(queue_len)
            metrics["throughput"].append(throughput)
            metrics["co2"].append(co2_raw)


            if step % 50 == 0:
                if control_mode == "rl":
                    print(f"Step {step:5d} | Actions: {actions} | Avg wait: {avg_wait:.2f} | Speed: {avg_speed:.2f} | Throughput: {throughput:.2f} veh/h")
                if control_mode == "classic":
                    print(f"Step {step:5d} | Actions: {actions} | Phase length: {phase_length} | Time: {traci.simulation.getTime()} | Start: {STATE_START}")
    except KeyboardInterrupt:
        print("Interrupted by user")

#Main function, entry point of the script
def main():
    #Set simulation params
    global STATE_START, CURRENT_PHASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", required=True)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--control-mode", choices=["classic", "rl","compare"], default="classic",
                        help="Choose between classic heuristic or RL control")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to RL checkpoint (DQN)")
    args = parser.parse_args()

    #Start sumo simulation
    start_sumo(args.sumocfg, use_gui=args.use_gui)
    #The root of the artifact mentioned in above NOTE
    tls_ids = get_all_tls_ids()
    if not tls_ids:
        print("No traffic lights found in the network. Exiting.")
        traci.close()
        sys.exit(1)

    print(f"Found TLS IDs: {tls_ids}")
    for tl in tls_ids:
        info = tls_phase_info(tl)
        print(f"TLS {tl}: phases={info['num_phases']} lanes={info['lanes']}")

    #Load DDQN model from saved checkpoint
    rl_model = None
    if args.control_mode == "rl" or args.control_mode == "compare":
        if not HAS_RL:
            print("RL packages not available. Cannot run in RL mode.")
            traci.close()
            sys.exit(1)
        if args.checkpoint is None or not os.path.exists(args.checkpoint + ".zip"):
            print(f"RL checkpoint not found at {args.checkpoint}. Exiting.")
            traci.close()
            sys.exit(1)
        rl_model = sb3.DQN.load(args.checkpoint)
        print(f"Loaded DQN model from {args.checkpoint}.zip")
    
    #Initialize metrics dictionaries
    metrics_rl = {"avg_speed": [], "avg_wait": [], "queue_len": [], "throughput": [],"co2": []}
    metrics_classic = {"avg_speed": [], "avg_wait": [], "queue_len": [], "throughput": [],"co2": []}
    
    #Start simulation
    #If mode is compare, the simulation will be started two times, with different control modes
    #If mode is "rl" or "classic" the simulation is only started once with the respected control type
    if args.control_mode == "compare":
        print(f"Running control loop (rl) for {args.steps} steps...")
        run_simulation(metrics_rl,tls_ids,"rl",args.steps,rl_model)
        stop_sumo()
        STATE_START = 0.0
        CURRENT_PHASE = 0
        start_sumo(args.sumocfg, use_gui=args.use_gui)
        print(f"Running control loop (classic) for {args.steps} steps...")
        run_simulation(metrics_classic,tls_ids,"classic",args.steps,rl_model)
    else:
        print(f"Running control loop ({args.control_mode}) for {args.steps} steps...")
        run_simulation(metrics_rl,tls_ids,args.control_mode,args.steps,rl_model)
    
    #Close the simulation
    stop_sumo()
    print("Simulation finished.")
    #Plot metrics
    plot_metrics(args.control_mode,metrics_rl,metrics_classic)


if __name__ == "__main__":
    main()
