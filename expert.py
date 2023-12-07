"""
#################################
# Python API: Trajectory Interface for Simulation
#################################
"""

#########################################################
# import libraries
import io
import re
import os
import lzma
import time
import glob
import torch
import queue
import pickle
import datetime
import torchvision
import cv2
import warnings
import wandb
import sys

import numpy as np
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from venv import create
from queue import Queue
from copy import deepcopy
from config import Config_BC
from config import Config_TRJ
from config import Config_Path
from rule_based import RuleBasedDriver
from utils.vis_utils import visualize_road, visualize, visualize_dy_objects
try:
    from pynput.keyboard import Key, Listener
except:
    print("pynput can't be imported")
from utils.collector_utils import SELECTION_EGO_OBS, EGO_OBS_MAP, HEADERS_TO_SAVE
from utils.collector_utils import create_folder, selection_size, select_from_array, convert_image_to_lane_ids, args_to_wandbnanme
from utils.trajectory_utils import data_ready_to_send
from utils.sim_env import SimPilotEnv
from collections import deque
from utils.data_utils import evaluate_data
from utils.trajectory_utils import map_key_to_yaw, human_input_to_trajectory
from mlagents_envs.exception import UnityCommunicatorStoppedException

warnings.filterwarnings("ignore")
queue_keys_steer = Queue()
queue_keys_acc = Queue()


#########################################################
# General Parameters
# Configurable parameters for rule based driver
TIME_PER_STEP = 0.02
EPSILON = 0.0001
LANE_CHANGE_TIME_LMT = 10 # Seconds
LANE_CHANGE_STEP_LMT = LANE_CHANGE_TIME_LMT / TIME_PER_STEP # Steps
FIRST_LANE_CHANGE_STEP_LMT = 1000
NUM_FUTURE_TRJ = Config_TRJ.get("NUMBER_POINTS")
NUM_CONTROL_ELEMENTS = Config_TRJ.get("NUM_CONTROL_ELEMENTS")
# NUM_EGO_ELEMENTS = Config_TRJ.get("NUM_EGO_ELEMENTS")
# TRJ_TIME_INTERVAL = Config_TRJ.get("TRJ_TIME_INTERVAL")
CONTROLLER_LANE_CHANGE_LMT = 3 # Speed limit for controller lane change command
CRASHED_DISPOSAL_STEPS = 250
EGO_COLLISION = [256.0, 512.0]
LANE_SWITCH = 2.0

CURRENT_TRAVEL_ASSIST = 0
LEFT_TRAVEL_ASSIST = 1
RIGHT_TRAVEL_ASSIST = 2
MAX_SPEED_TRAVEL_ASSIST = 44.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
run_date_time = time.strftime("%Y_%m_%d-%H_%M")

ta_map = {0: "None",
          1: "Instantiated",
          2: "Ready to change Lane",
          3: "Started Movement",
          4: "None",
          5: "None",
          6: "None"}
ta_map_new = {0: "None",
              1: "Instantiated",
              2: "Ready to change Lane",
              3: "Started Movement",
              4: "Interrupted",
              5: "Success",
              6: "Failed"}
# 4, 5, and 6 are for newer versions of simpilot (12.0.0 and after)
#########################################################
# Function definition
def expert_metrics(current_speed, speed_limit):
    """_summary_

    Args:
        current_speed (_type_): _description_
        speed_limit (_type_): _description_
    """
    wandb.log({
              'ABS difference': abs(current_speed - speed_limit),
              'Relative difference': ((current_speed - speed_limit) / speed_limit),
              'Speed Difference': current_speed - speed_limit,
              'current speed':current_speed,
              'Speed Limit':speed_limit
    })


def expert_drive(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    if args.randomize_rule_based:
        REACT_TIME = np.random.randint(2, 4) # Time of reaction between the ego and vehicle in front in (s)
        SPEED_DIFF = np.random.randint(2, 5) # Speed difference that you can ignore (m/s)
        SAFE_DIST_FRONT = np.random.randint(6, 10) # Safe distance with the vehicle in front and behind ego (m)
        SAFE_DIST_REAR = np.random.randint(4, 6)
    wandb_project_name = args_to_wandbnanme(args, run_date_time)
    if args.track:
        wandb.init(
            project=args.algo,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=wandb_project_name,
            save_code=True,
            )
        if args.evaluate:
            wandb.define_metric("epoch_step")
            wandb.define_metric("epoch/*", step_metric="epoch_step")
    visualize_flag = args.visu

    env_simpilot = SimPilotEnv(
        args=args,
        exec_name=args.exec_path,
        no_graphic=args.no_graphic
    )
    env = env_simpilot.load_env_unity()
    env_sumo = env_simpilot.load_env_sumo()
    env_visualize = env_simpilot.load_env_visualization()
    env_string = env_simpilot.load_env_string_channel()

    if args.print_flag:
        print(f"ego_obs_map={EGO_OBS_MAP}")
        print(f"check total: {selection_size(EGO_OBS_MAP, SELECTION_EGO_OBS)}")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    initials = args.initials
    collection_time = int(time.time())
    milestone = args.milestone
    collection_task = args.task
    recording_data_path = args.rawdata_path

    if args.record_data:
        recording_data_path = args.rawdata_path
        if not os.path.exists(os.path.join(recording_data_path, "datafiles")):
            os.makedirs(os.path.join(recording_data_path, "datafiles"))

        if args.RoadIDSensor:
            roadidsensor_folder, _ = create_folder(os.path.join(recording_data_path, "images/RoadIDSensor"))
        if args.LaneIDSensor:
            laneidsensor_folder, i_LaneIDSensor = create_folder(os.path.join(recording_data_path, "images/LaneIDSensor"))
        if args.DrivableSensor:
            drivablesensor_folder, _ = create_folder(os.path.join(recording_data_path, "images/DrivableSensor"))

    # Dict to get observations by name
    obs_idx_map = dict()
    for key in env.behavior_specs:
        print(f"Behavior: {key}")
        spec = env.behavior_specs[key]
        print(f"\tAction Spec: discrete_size={spec.action_spec.discrete_size}; continuous_size={spec.action_spec.continuous_size}")
        print("\tObservation Specs:")
        for i, _ in enumerate(spec.observation_specs):
            obs_spec = spec.observation_specs[i]
            name = ("EgoObservation" if "VectorSensor_size" in obs_spec.name else obs_spec.name)
            obs_idx_map[name] = i
            print(f"\t\tName={obs_spec.name} | Shape={obs_spec.shape} | Type={obs_spec.observation_type.name}")

    if not args.no_graphic:
        listener = Listener(on_press=detect)
        listener.start()

    behavior_name = list(env.behavior_specs)[0]
    keyboardinterrupt = False
    max_driven_distance = 0
    ego_yaw = 0
    fig_obj, fig_obj_grid, fig_dy_obj = None, None, None
    total_num_steps = 0
    driver = RuleBasedDriver()
    for eps in np.arange(0, args.num_eps):
        if fig_obj is not None:
            plt.close(fig_obj)
        if fig_obj_grid is not None:
            plt.close(fig_obj_grid)
        fig_obj, sc_trj, fig_obj_grid = None, None, None
        driver.reset_dist_lane_change()
        frame_id = 0
        min_acc = np.inf
        max_vel = -np.inf
        step = 0
        init_x, init_y = None, None
        init_time = time.time()
        completed_loop = False
        num_lane_change = 0
        total_distance = 0
        total_num_lane_changes = 0
        # *********** RANDOM ENVIRONMENT
        if args.randomization_env:
            laneid_wanted = args.randomization_laneid
            spawnpath = args.spawnpoints_path
            df_rand = pd.read_csv(spawnpath, delimiter=',')
            if laneid_wanted == 0:
                df_filtered = df_rand
            else:
                df_filtered = df_rand[df_rand['continuous_lane_id'] == laneid_wanted].sample(frac=1)
            sampled_row = df_filtered.sample(n=1)
            env.reset()
            env_simpilot.agent_channel.set_init_transform(x=float(sampled_row.px.values),
                                                          y=float(sampled_row.py.values),
                                                          yaw=float(sampled_row.yaw.values))

        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name=behavior_name)
        actions = spec.action_spec.empty_action(n_agents=decision_steps.agent_id.size)
        agent_ids = list(decision_steps)
        num_agents = len(agent_ids)
        tracked_agent = -1
        done = False
        crashed = False
        x_old, y_old = 0, 0
        current_yaw_deg = 0
        start_time = time.time()
        data_frame_episode = []
        reset = True
        lane_change_sent = False
        previous_lane = None

        df = pd.DataFrame(None, columns=HEADERS_TO_SAVE)

        lane_id_map_lst = []
        road_id_map_lst = []
        drivable_area_map_lst = []

        target_speed = int(args.target_speed)
        speed_action = target_speed
        lane_change_command = 0
        lane_change_action = 0

        previous_lane_change_step = -1000
        previous_lane_change_command = 0

        # Reading report from Simpilot through SideChannel
        # simpilot_report = env_string.parameters["EpisodeReport"]

        if args.visu:
            fig_scatter = plt.figure(figsize=(4, 6))
            ax_speed = fig_scatter.add_subplot(111)
            # ax_pos = fig_scatter.add_subplot(212)
            ax_speed.grid(True)
            # ax_pos.grid(True)
            # plt.subplots_adjust(
            #                     wspace=0.4,
            #                     hspace=0.9)
            plt.show(block=False)
            rule_based_speeds = []
            ego_speeds = []
            ax_positions_x = deque(maxlen=250)
            ax_positions_y = deque(maxlen=250)
            lane_ids = []

        while done is False:
            try:
                if tracked_agent == -1 and len(decision_steps) >= 1:
                    tracked_agent = decision_steps.agent_id[0]

                # Retrieve ego observations
                ego_obs = decision_steps.obs[obs_idx_map["EgoObservation"]][0]

                timestamp = ego_obs[0]
                ego_id = ego_obs[1]
                position_x = ego_obs[2]
                position_y = ego_obs[3]
                velocity_x = ego_obs[4]
                velocity_y = ego_obs[5]
                acceleration_x = ego_obs[6]
                acceleration_y = ego_obs[7]
                orientation = ego_obs[8]
                heading_x = ego_obs[9]
                heading_y = ego_obs[10]
                continuous_lane_id = ego_obs[11]
                lane_relative_t = ego_obs[12]
                angle_to_lane = ego_obs[13]
                vehicle_switching_lane = ego_obs[14]
                ego_collision_type = ego_obs[15]
                controller_state = ego_obs[16]
                nav_turn_command_n = ego_obs[17]
                nav_point_position_n_x = ego_obs[18]
                nav_point_position_n_y = ego_obs[19]
                nav_point_distance_n = ego_obs[20]
                nav_point_angle_n = ego_obs[21]
                nav_turn_command_n1 = ego_obs[22]
                nav_point_position_n1_x = ego_obs[23]
                nav_point_position_n1_y = ego_obs[24]
                nav_point_distance_n1 = ego_obs[25]
                left_lane_available = ego_obs[26]
                right_lane_available = ego_obs[27]
                allowed_speed = ego_obs[28]
                acc_target_speed = ego_obs[29]
                travel_assist_lane_change_state = ego_obs[30]

                total_distance += velocity_x * TIME_PER_STEP
                if args.visu:
                    lane_ids.append(continuous_lane_id)
                if step == 0:
                    init_x = position_x
                    init_y = position_y

                if previous_lane is None:
                    previous_lane = continuous_lane_id

                if args.print_flag and step % args.print_rate == 0:
                    print(f"Left lane availability is {left_lane_available} "
                          f"Right lane availability is {right_lane_available} ")
                ego_obs_selected_dict = select_from_array(
                    ego_obs, EGO_OBS_MAP, SELECTION_EGO_OBS
                )

                # Retrieve static lane observations
                static_lanes = decision_steps.obs[obs_idx_map["StaticLanes"]][0].astype(np.float16)
                single_lane_0 = static_lanes[0]
                lane_0_id = single_lane_0[0]
                lane_0_speed_limit = single_lane_0[1]
                lane_0_type = single_lane_0[2]

                speed_limit = lane_0_speed_limit

                # Retrieve BEV observations
                if args.DrivableSensor:
                    drivable_area_map = decision_steps.obs[obs_idx_map["DrivableAreaSensor"]][0]
                    if args.print_flag:
                        print(f"drivable_area_map: {drivable_area_map.shape} {drivable_area_map[:, :, 0]}")
                if args.LaneIDSensor:
                    lane_id_map = decision_steps.obs[obs_idx_map["LaneIdSensor"]][0]
                    lane_id_map = convert_image_to_lane_ids(lane_id_map)
                    if args.print_flag:
                        print(f"lane_id_map: {lane_id_map.shape} {lane_id_map[:, :, 0]}")
                if args.RoadIDSensor:
                    road_id_map = decision_steps.obs[obs_idx_map["RoadIdSensor"]][0]
                    road_id_map = convert_image_to_lane_ids(road_id_map)
                    if args.print_flag:
                        print(f"road_id_map: {road_id_map.shape} {road_id_map[:, :, 0]}")

                # Retrieve movable object observations
                movable_obj = decision_steps.obs[obs_idx_map["MovableObjects"]][0]
                num_available_dy_obj = np.count_nonzero(movable_obj[:, 0])
                if step % args.print_rate == 0 and args.print_flag:
                    for _, dy_obj in enumerate(movable_obj[0:num_available_dy_obj]):
                        print(f"Object_{int(dy_obj[0])}: vx={dy_obj[3]:.2f} | "
                            f"pos_x={dy_obj[1]:.2f} | pos_y={dy_obj[2]:.2f} | "
                            f"lane={dy_obj[6]} | dist={np.sqrt(dy_obj[1]**2 + dy_obj[2]**2):.2f} |"
                            f"orientation={dy_obj[5]:.2f} | length={dy_obj[7]:.2f} | width={dy_obj[8]:.2f}")

                obj_1 = movable_obj[0]
                obj_1_id = obj_1[0]
                obj_1_position_x = obj_1[1]
                obj_1_position_y = obj_1[2]
                obj_1_velocity_x = obj_1[3]
                obj_1_velocity_y = obj_1[4]
                obj_1_orientation = obj_1[5]
                obj_1_lane_id = obj_1[6]
                obj_1_box_length = obj_1[7]
                obj_1_box_width = obj_1[8]
                obj_1_type = obj_1[9]
                obj_1_dist_to_center_of_lane = obj_1[10]
                obj_1_signal = obj_1[11]
                
                if step % args.print_rate == 0 and args.print_flag:
                    movable_obj = decision_steps.obs[obs_idx_map["MovableObjects"]][0]
                    print(f"Number of movable objects is {len(movable_obj)}\n")
                    print(f"Ego collision type is {ego_collision_type}")
                    for i in range(len(movable_obj)):
                        obj_1 = movable_obj[i]
                        obj_1_id = obj_1[0]
                        obj_1_position_x = obj_1[1]
                        obj_1_position_y = obj_1[2]
                        obj_1_velocity_x = obj_1[3]
                        obj_1_velocity_y = obj_1[4]
                        obj_1_orientation = obj_1[5]
                        obj_1_lane_id = obj_1[6]
                        obj_1_box_length = obj_1[7]
                        obj_1_box_width = obj_1[8]
                        obj_1_type = obj_1[9]
                        obj_1_dist_to_center_of_lane = obj_1[10]
                        if obj_1_lane_id != 0:
                            print(f"Movable object {i} with the id {obj_1_id}\n"
                                  f"Location: ({obj_1_position_x}, {obj_1_position_y})\n"
                                  f"Speed: ({obj_1_velocity_x}, {obj_1_velocity_y})\n"
                                  f"Lane id is {obj_1_lane_id}\n"
                                  f"Orientation is {obj_1_orientation}\n")
                            print("-" * 50)
                    print('_' * 100)

                if step % args.print_rate == 0 and args.print_flag:

                    print("Ego position x {position_x}, position y {position_y}, and heading {orientation}")
                    print("Ego heading x {heading_x} and heading y {heading_y}")
                    print("Movable object 1 position x {movable_obj[0][1]}, position y {movable_obj[0][2]}, and orientation {movable_obj[0][5]}")
                    print("Movable object 2 position x {movable_obj[1][1]}, position y {movable_obj[1][2]}, and orientation {movable_obj[1][5]}")

                # ***************************** Key pressed
                if args.human:
                    try:
                        key_pressed_steer = queue_keys_steer.get(block=False)
                        # print('\nYou Entered (from queue) {0}'.format(key_pressed_steer))
                    except queue.Empty:
                        key_pressed_steer = None

                    try:
                        key_pressed_acc = queue_keys_acc.get(block=False)
                        # print('\nYou Entered (from queue) {0}'.format(key_pressed_steer))
                    except queue.Empty:
                        key_pressed_acc = None

                    current_yaw_deg, current_yaw_rad, target_speed = map_key_to_yaw(
                        key_pressed_steer, current_yaw_deg, key_pressed_acc, target_speed
                    )

                    x_trj, y_trj, vx_trj = human_input_to_trajectory(
                        velocity_x, target_speed, current_yaw_rad
                    )

                    control_points, control_ego = data_ready_to_send(
                        x_ref=x_trj, y_ref=y_trj, vx_ref=vx_trj
                    )

                if args.sumo and args.no_graphic is False:
                    # Lane change command
                    try:
                        key_pressed_steer = queue_keys_steer.get(block=False)
                    except queue.Empty:
                        key_pressed_steer = None

                    # Overtake command
                    try:
                        key_pressed_acc = queue_keys_acc.get(block=False)
                    except queue.Empty:
                        key_pressed_acc = None

                    if args.semi_auto and args.controller == "SumoController":
                        if key_pressed_steer == Key.left:
                            env_simpilot.sumo_channel.changeLaneRelative('EgoCar_0', 1, 10)
                        if key_pressed_steer == Key.right:
                            env_simpilot.sumo_channel.changeLaneRelative('EgoCar_0', -1, 10)

                    if args.semi_auto and args.controller in ("TravelAssist", "TravelAssistUnsafe"):
                        lane_change_action = CURRENT_TRAVEL_ASSIST
                        if key_pressed_steer == Key.left:
                            lane_change_action = LEFT_TRAVEL_ASSIST
                        if key_pressed_steer == Key.right:
                            lane_change_action = RIGHT_TRAVEL_ASSIST

                        if key_pressed_acc == Key.up:
                            target_speed += 1
                            # print("Increasing target speed")
                        elif key_pressed_acc == Key.down:
                            target_speed -= 1
                            # print("Decreasing target speed")

                    x_trj, y_trj, vx_trj = human_input_to_trajectory(
                        ego_speed=0, target_speed=0, ego_yaw=0
                    )
                    control_points, control_ego = data_ready_to_send(
                        x_ref=x_trj, y_ref=y_trj, vx_ref=vx_trj
                    )

                if visualize_flag and (step % args.vis_rate == 0):
                    # fig_obj_grid = visualize_road(fig_obj_grid, drivable_area_map,
                    # lane_id_map, road_id_map,)
                    fig_dy_obj = visualize_dy_objects(args, fig_dy_obj, lane_id_map, movable_obj)

                if args.human:
                    action_trj_vector = control_points.T.flatten()
                    actions_trj = spec.action_spec.empty_action(n_agents=1)
                    actions_trj.add_continuous(np.expand_dims(action_trj_vector, axis=0))
                    env.set_actions(behavior_name=behavior_name, action=actions_trj)

                if args.sumo:
                    target_speed = lane_0_speed_limit
                    if args.controller == "SumoController":
                        env_sumo.setSpeed('EgoCar_0', target_speed)
                    if args.controller in ("TravelAssist", "TravelAssistUnsafe"):

                        if args.semi_auto:
                            actions_travel_assist = spec.action_spec.empty_action(n_agents=1)
                            actions_travel_assist.add_discrete(np.expand_dims([lane_change_action], axis=0))
                            actions_travel_assist.add_continuous(np.expand_dims([float(target_speed / MAX_SPEED_TRAVEL_ASSIST)], axis=0))
                            env.set_actions(behavior_name=behavior_name, action=actions_travel_assist)

                        if args.rule_based:
                            objs = driver.get_near_objs(movable_obj, continuous_lane_id)
                            left_rear, left_front, cur_rear, cur_front, right_rear, right_front = objs
                            #Each vehicle (x, y, v, orientation)
                            if step % args.print_rate == 0:
                                print("Vehicle switching lane:", vehicle_switching_lane)
                                print("Lane change sent: ", lane_change_sent)
                                print("Left rear:", left_rear)
                                print("Left front:", left_front)
                                print("Cur rear:", cur_rear)
                                print("Cur front:", cur_front)
                                print("Right rear:", right_rear)
                                print("Right front:", right_front)

                            if ta_map[travel_assist_lane_change_state] == "None":
                                lane_change_command, speed_action = driver.change_lane(objs,
                                                                                       velocity_x,
                                                                                       acceleration_x,
                                                                                       target_speed,
                                                                                       continuous_lane_id,
                                                                                       left_lane_available,
                                                                                       right_lane_available)
                                if lane_change_command != 0:
                                    if step - previous_lane_change_step >= LANE_CHANGE_STEP_LMT \
                                                        and velocity_x > CONTROLLER_LANE_CHANGE_LMT \
                                                        and step > FIRST_LANE_CHANGE_STEP_LMT:
                                        num_lane_change += 1
                                    else:
                                        lane_change_command = 0
                                        speed_action = driver.keep_current_lane(objs,
                                                                                velocity_x,
                                                                                acceleration_x,
                                                                                target_speed)
                            else:
                                lane_change_command = 0
                                if ta_map[travel_assist_lane_change_state] in \
                                    ("Started Movement", "Ready to change lane") and \
                                    vehicle_switching_lane != LANE_SWITCH:
                                    speed_action = driver.get_speed_lane_change(objs,
                                                                                velocity_x,
                                                                                acceleration_x,
                                                                                target_speed,
                                                                                previous_lane_change_command)
                                else:
                                    speed_action = driver.keep_current_lane(objs,
                                                                            velocity_x,
                                                                            acceleration_x,
                                                                            target_speed)
                                if previous_lane != continuous_lane_id:
                                    #Lane changed
                                    if args.track:
                                        wandb.log({"Steps took for a lane change": step - previous_lane_change_step})

                            speed_action /= MAX_SPEED_TRAVEL_ASSIST
                            # 44.45 = ego max speed
                            actions = spec.action_spec.empty_action(n_agents=decision_steps.agent_id.size)
                            actions.add_discrete(np.expand_dims([lane_change_command], axis=0))
                            if args.adaptive_cruise_control:
                                actions.add_continuous(np.expand_dims([speed_limit / MAX_SPEED_TRAVEL_ASSIST], axis=0))
                            else:
                                actions.add_continuous(np.expand_dims([speed_action], axis=0))
                            env.set_actions(behavior_name=behavior_name, action=actions)
                            previous_lane = continuous_lane_id
                            if lane_change_command != 0:
                                previous_lane_change_step = step
                                previous_lane_change_command = lane_change_command
                            if args.visu:
                                rule_based_speeds.append(speed_action * MAX_SPEED_TRAVEL_ASSIST)
                                ego_speeds.append(velocity_x)
                                ax_positions_x.append(position_x)
                                ax_positions_y.append(position_y)

                # ************************************ Data Collection
                if args.record_data:
                    image_suffix = f"{initials}_{collection_time}_{milestone}_{collection_task}_{eps}_{step}.png"
                    image_name = "laneidsensor_" + image_suffix

                    df.loc[len(df.index)] = [
                        initials,
                        collection_time,
                        milestone,
                        collection_task,
                        eps,
                        step,
                        time.time() - start_time,
                        velocity_x,
                        position_x,
                        position_y,
                        timestamp,
                        heading_x,
                        heading_y,
                        acceleration_x,
                        acceleration_y,
                        orientation,
                        continuous_lane_id,
                        lane_relative_t,
                        angle_to_lane,
                        controller_state,
                        vehicle_switching_lane,
                        x_trj,
                        y_trj,
                        vx_trj,
                        control_points.flatten(),
                        static_lanes.flatten(),
                        image_name,
                        speed_limit,
                        'Sumo' if args.sumo else 'Human',
                        ego_collision_type,
                        left_lane_available,
                        right_lane_available,
                        allowed_speed,
                        movable_obj,
                        speed_action,
                        lane_change_command,
                        travel_assist_lane_change_state
                    ]

                    if args.LaneIDSensor:
                        lane_id_map_lst.append(lane_id_map)
                    if args.RoadIDSensor:
                        road_id_map_lst.append(road_id_map)
                    if args.DrivableSensor:
                        drivable_area_map_lst.append(drivable_area_map)

                if step % args.print_rate == 0:
                    speed_action_print = speed_action * MAX_SPEED_TRAVEL_ASSIST if args.rule_based else speed_action
                    print(f" *********** Eps = {eps} "
                          f" *********** Step = {step} "
                          f" *********** Action Speed = {speed_action_print} "
                          f" *********** Action Lane = {lane_change_command}"
                          f" *********** Current Speed = {velocity_x:.2f}"
                          f" *********** Speed limit = {lane_0_speed_limit:.2f}"
                          f" *********** TA Status : {ta_map_new[travel_assist_lane_change_state]}"
                          f" *********** Lane id : {continuous_lane_id}"
                    )

                if step > 2000 and \
                np.sqrt((init_x - position_x) ** 2 + (init_y - position_y) ** 2) < 10 and \
                not completed_loop:
                    done = True
                    completed_loop = True
                    if args.track:
                        wandb.log({"Steps to finish 1 loop": step,
                                   "Time to finish 1 loop": time.time() - init_time,
                                   "Number of Lane Changes in 1 loop": num_lane_change},
                                  step=eps)
                try:
                    env.step()
                except UnityCommunicatorStoppedException:
                    exit(" ********************* Exit: UnityCommunicatorStoppedException"
                         " *********************")

                time_stamp = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")[:-3]
                decision_steps, terminal_steps = env.get_steps(behavior_name=behavior_name)

                if args.track:
                    expert_metrics(velocity_x, target_speed)
                
                if args.visu and step % 45 == 0:
                    ax_speed.plot(rule_based_speeds, color='orange', label='Action')
                    ax_speed.plot(ego_speeds, color='b', label='Ego speed')
                    ax_speed.plot(lane_ids, color = 'r', label='Lane ID')
                    # ax_pos.plot(ax_positions_x, ax_positions_y, color='r', label='Positions')
                    if step == 0:
                        plt.legend(bbox_to_anchor=(1.02, 0.1), borderaxespad=0)
                        plt.title("Speed comparison")                
                    plt.pause(0.00000000001)

                step += 1
                total_num_steps += 1
                if tracked_agent in terminal_steps:
                    done = True
                if ego_collision_type in EGO_COLLISION:
                    done = True
                    crashed = True
                    if args.track:
                        wandb.log({"Distance before collision": total_distance}, step=eps)

            except KeyboardInterrupt:
                keyboardinterrupt = True
                done = True
                break
        if args.visu:
            plt.close()
        # Saving data before interrupt
        if args.record_data:
            list_len = max(len(road_id_map_lst),
                           max(len(lane_id_map_lst),
                               len(drivable_area_map_lst)
                               )
                           )
            # Don't throw away all the collected data in the case of an accident
            if crashed:
                list_len = max(0, list_len - CRASHED_DISPOSAL_STEPS)
                df = df.drop(df.index[-CRASHED_DISPOSAL_STEPS:])

            if list_len != 0:
                if not args.evaluate:
                    for ind in range(list_len):
                        image_suffix = f"{initials}_{collection_time}_{milestone}_{collection_task}_{eps}_{ind}.png"

                        if args.RoadIDSensor:
                            road_id_map = road_id_map_lst[ind]
                            cv2.imwrite(os.path.join(roadidsensor_folder, "roadidsensor_" + image_suffix,),
                                        road_id_map,)
                        if args.LaneIDSensor:
                            lane_id_map = lane_id_map_lst[ind]
                            cv2.imwrite(os.path.join(laneidsensor_folder, "laneidsensor_" + image_suffix,),
                                        lane_id_map,)
                        if args.DrivableSensor:
                            drivable_area_map = drivable_area_map_lst[ind]
                            cv2.imwrite(os.path.join(drivablesensor_folder,
                                                    "driveablesensor_" + image_suffix,),
                                        drivable_area_map,)

                if args.RoadIDSensor or args.LaneIDSensor or args.DrivableSensor:
                    df.to_pickle(os.path.join(recording_data_path, "datafiles/{}_{}_{}_{}_{}.pkl".format(initials,
                                                                                                        milestone,
                                                                                                        collection_task,
                                                                                                        collection_time,
                                                                                                        i_LaneIDSensor)))
                if args.track:
                    log_dict = {
                        "epoch_step": eps + 1,
                        "epoch/Time_to_finish_epoch": time.time() - init_time,
                        "epoch/Steps_to_finish_epoch": step,
                        "epoch/Distance_travelled_before_accident": total_distance,
                        "epoch/Number_of_Lane_Changes_in_one_epoch": total_num_lane_changes,
                        }
                    wandb.log(log_dict)

        if keyboardinterrupt:
            env.close()
            exit(" ********************* Exit: keyboardinterrupt *********************")

        # Creating new files and folders for the new episode
        if args.record_data:
            recording_data_path = args.rawdata_path

            if not os.path.exists(os.path.join(recording_data_path, "datafiles")):
                os.makedirs(os.path.join(recording_data_path, "datafiles"))

            if eps != args.num_eps - 1 and list_len != 0:
                if args.RoadIDSensor:
                    roadidsensor_folder, _ = create_folder(os.path.join(recording_data_path,
                                                                        "images/RoadIDSensor"))
                if args.LaneIDSensor:
                    laneidsensor_folder, i_LaneIDSensor = create_folder(os.path.join(recording_data_path, "images/LaneIDSensor"))
                if args.DrivableSensor:
                    drivablesensor_folder, _ = create_folder(os.path.join(recording_data_path,
                                                                          "images/DrivableSensor"))

        if args.randomization_env and \
            eps % args.new_rand_eps == args.new_rand_eps - 1 and \
                eps != args.num_eps - 1:
            # env_simpilot.configure_vtype(args)
            env, env_sumo = env_simpilot.hard_reset(args)
    if args.track:
        wandb.log({"Average number of steps before accident": total_num_steps / args.num_eps}, step=1)
    env.close()
    if args.evaluate:
        print('EVALUATION STARTS NOW')
        args.model_name = ""
        evaluate_data(args)
    exit(" ********************* Exit: Done collecting data *********************")


def detect(key):
    """_summary_

    Args:
        key (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print('\nYou Entered {0}'.format(key))
    if (key == Key.left) or (key == Key.right):
        queue_keys_steer.put(key)
    if (key == Key.up) or (key == Key.down):
        queue_keys_acc.put(key)

    if key == Key.delete:
        # Stop listener
        return False
