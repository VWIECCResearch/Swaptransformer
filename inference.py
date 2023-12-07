"""
#################################
# Python API: Imitation Learning Inference
#################################
"""

#########################################################
# import libraries
import re
import os
import time
import torch
import wandb
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from torchvision import transforms
from config import Config_General
from utils.sim_env import SimPilotEnv
from torch.utils.data import DataLoader
from model.network import BehavioralCloning, Generator
from utils.vis_utils import visualize_inference, visualize_dy_objects
from utils.collector_utils import HEADERS_TO_LOAD, HEADERS_TO_PREDICT, HEADERS_TO_SAVE_INFERENCE
from mlagents_envs.exception import UnityCommunicatorStoppedException
from utils.collector_utils import convert_image_to_lane_ids, args_to_wandbnanme, create_folder
from utils.data_utils import PickleDataset, pose_to_lane, lanes_to_commands, lanes_to_travel_assist, quartic_bezier_curve, evaluate_data
from rule_based import RuleBasedDriver

#########################################################
# General Parameters
BEZIER_DIM = 4 * 2
NUM_MOVE_OBJS = 20
MAX_SPEED_TRAVEL_ASSIST = 44.5
TIME_PER_STEP = 0.02
NUM_MOVE_OBJS = 20
EGO_COLLISION = [256.0, 512.0, 262400.0,263168.0]

CURRENT_TRAVEL_ASSIST = 0
LEFT_TRAVEL_ASSIST = 1
RIGHT_TRAVEL_ASSIST = 2
TRANSITION_TRAVEL_ASSIST = 3

move_obj_columns_hybrid = {"id": 0, "x": 1, "y": 2, "vx": 3, "vy": 4, "theta": 5,
                    "lane": 6, "length": 7, "width": 8, "type": 9, "relative_t": 10}
move_obj_columns = {'pos_x': 1,
                    'pos_y': 2,
                    'velocity': 3,
                    'Continuous Lane Id': 6,
                    'Bounding box length': 7}
ta_map_new = {0: "None",
              1: "Instantiated",
              2: "Ready to change Lane",
              3: "Started Movement",
              4: "Interrupted",
              5: "Success",
              6: "Failed"}

FLOAT_DECIMAL = Config_General.get("FLOAT_DECIMAL")
transform = transforms.Compose([transforms.ToTensor()])
current_file_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################
# Function definition

def inference_metrics(current_speed, speed_limit):
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


def model_name_to_args(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    if "_Single_" in args.model_name:
        args.single_head = True
    else:
        args.single_head = False
    if "NonBezier" in args.model_name:
        args.bezier = False
    else:
        args.bezier = True
    if "NonTA" in args.model_name:
        args.travelassist_pred = False
    else:
        args.travelassist_pred = True
    if "NonResidual" in args.model_name:
        args.residual = False
    else:
        args.residual = True
    if "Singleopt" in args.model_name:
        args.multi_opt = False
    else:
        args.multi_opt = True
    if "CarNet" in args.model_name:
        args.car_network = True
    else:
        args.car_network = False
    if "transformer" in args.model_name:
        args.base_model = "transformer"
    elif "mhsa" in args.model_name:
        args.base_model = "mhsa"
    else:
        args.base_model = "mlp"
    if "NoSwap" in args.model_name:
        args.swap = False
    else:
        args.swap = True
    args.activation = re.search('act_(.*?)_', args.model_name).group(1)
    args.encoder = re.search('encoder_(.*?)_', args.model_name).group(1)


def inference(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    if args.infer_type == 'Online':

        # This is the Inference Mode
        # LOAD MODEL

        if args.evaluate:
            for dirmodel in os.listdir(args.model_path):
                NON_BEZIER_DIM = args.num_poses * args.num_featurespose
                controller = "Safe" if args.controller == "TravelAssist" else "UnSafe"
                if args.track:
                    wandb.init(
                        project=args.algo,
                        entity=args.wandb_entity,
                        sync_tensorboard=False,
                        config=vars(args),
                        name=f"Dev_{args.initials}_METRICS_{controller}=" + dirmodel,
                        save_code=False,
                    )
                    wandb.define_metric("epoch_step")
                    wandb.define_metric("epoch/*", step_metric="epoch_step")
                args.model_name = dirmodel
                recording_data_path = os.path.join(args.rawdata_path, args.model_name)

                if not os.path.exists(recording_data_path):
                    os.makedirs(recording_data_path)
                if not os.path.exists(os.path.join(recording_data_path, "datafiles")):
                    os.makedirs(os.path.join(recording_data_path, "datafiles"))
                model_name_to_args(args)

                if args.algo == "BC":
                    if args.single_head:
                        model_infer = BehavioralCloning(
                            args=args,
                            input_c=args.num_framestack,
                            output_size= (BEZIER_DIM + args.num_poses) if args.bezier else NON_BEZIER_DIM).to(device)
                    else:
                        model_infer = BehavioralCloning(
                            args=args,
                            input_c=args.num_framestack,
                            output_size=BEZIER_DIM if args.bezier else NON_BEZIER_DIM).to(device)
                    run_model(args, model_infer)
        else:

            run_date_time = time.strftime("%Y_%m_%d-%H_%M")
            model_name_to_args(args)
            wandb_project_name = args_to_wandbnanme(args, run_date_time)
            NON_BEZIER_DIM = args.num_poses * args.num_featurespose
            if args.track:
                wandb.init(
                    project=args.algo,
                    entity=args.wandb_entity,
                    sync_tensorboard=False,
                    config=vars(args),
                    name=wandb_project_name,
                    save_code=False,
                )

                wandb.define_metric("epoch_step")
                wandb.define_metric("epoch/*", step_metric="epoch_step")
            if args.algo == "BC":
                if args.single_head:
                    model_infer = BehavioralCloning(
                        args=args,
                        input_c=args.num_framestack,
                        output_size= (BEZIER_DIM + args.num_poses) if args.bezier else NON_BEZIER_DIM).to(device)
                else:
                    model_infer = BehavioralCloning(
                        args=args,
                        input_c=args.num_framestack,
                        output_size=BEZIER_DIM if args.bezier else NON_BEZIER_DIM).to(device)
                run_model(args, model_infer)
    else:
        dataset = PickleDataset(file_path=args.training_df_path,
                                image_folder=args.training_image_path,
                                column_names=HEADERS_TO_LOAD,
                                transform=transform,
                                predict_columns=HEADERS_TO_PREDICT,
                                num_framestack=args.num_framestack,
                                dim_input_feature=args.dim_input_feature,
                                args=args)

    if args.infer_type == 'Offline':
        dataloader_inference = DataLoader(dataset, batch_size=1, shuffle=False)
        iterator = iter(dataloader_inference)
        if args.replay_data:
            for _ in range(len(dataset)):
                df_stacked, stacked_images, future_points = next(iterator)
                image = torch.squeeze(stacked_images)[0]

                future_pose_x = torch.squeeze(future_points)[0:args.num_poses]
                future_pose_y = torch.squeeze(future_points)[args.num_poses:2 * args.num_poses]
                future_pose_v = torch.squeeze(future_points)[2 * args.num_poses:3 * args.num_poses]
                if args.visu:
                    fig_obj, sc_trj = visualize_inference(args, fig_obj, sc_trj, image, future_points,
                                                          df_stacked, speed_limit=0, current_speed=0)
                    print(f"X_POSE = {future_pose_x}, ******* Y_POSE = {future_pose_y} ",
                          f" *******, V_POSE = {future_pose_v}")

        else:
            # This is the Inference Mode
            # LOAD MODEL
            if args.algo == "BC":
                model_infer = BehavioralCloning(args=args, input_c=args.dim_input_feature,
                                                output_size=args.num_poses*args.num_featurespose).to(device)
            elif args.algo == "GAN":
                model_infer = Generator(args=args, input_c=5, output_size=15).to(device)
            elif args.algo == "GAIL":
                raise NotImplementedError

            if args.model_name[-5:] == ".ckpt":
                checkpoint = torch.load(os.path.join(args.model_path, args.model_name))
                model_infer.load_state_dict(checkpoint["model_state_dict"])
            else:
                model_infer.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name)))

            for _ in range(len(dataset)):

                # Pass input to the model
                df_stacked, stacked_images, _ = next(iterator)
                with torch.no_grad():
                    action, _, _ = model_infer(image=stacked_images.to(device),
                                               nparray=df_stacked.to(device),
                                               )

                # Get the action or the output and visualize it
                image_frame = torch.squeeze(stacked_images)[0]
                speed_limit = torch.squeeze(df_stacked)[0][4]
                current_speed = torch.squeeze(df_stacked)[0][0]
                if args.track:
                    if int(speed_limit) > 0:
                        inference_metrics(current_speed=current_speed, speed_limit=speed_limit)
                if args.visu:
                    fig_obj, sc_trj = visualize_inference(args, fig_obj, sc_trj, image_frame, action, df_stacked,
                                                          speed_limit=speed_limit,
                                                          current_speed=current_speed)
                    future_pose_x = torch.squeeze(action)[0:args.num_poses].cpu().numpy()
                    future_pose_y = torch.squeeze(action)[args.num_poses:2 * args.num_poses].cpu().numpy()
                    future_pose_v = torch.squeeze(action)[2 * args.num_poses:3 * args.num_poses].cpu().numpy()
                    print(f"X_P5OSE = {future_pose_x}, ******* Y_POSE = {future_pose_y} *******, V_POSE = {future_pose_v}")

    if args.infer_type == 'Hybrid':
        env_simpilot = SimPilotEnv(args=args,
                                   exec_name=args.exec_path,
                                   )
        env = env_simpilot.load_env_unity()
        env_sumo = env_simpilot.load_env_sumo()
        env_visualize = env_simpilot.load_env_visualization()
        env_string = env_simpilot.load_env_string_channel()

        obs_idx_map = dict()
        for key in env.behavior_specs:
            print(f"Behavior: {key}")
            spec = env.behavior_specs[key]
            print(f"\tAction Spec: discrete_size={spec.action_spec.discrete_size}; continuous_size={spec.action_spec.continuous_size}")
            print("\tObservation Specs:")
            for (i, _) in enumerate(spec.observation_specs):
                obs_spec = spec.observation_specs[i]
                name = ("EgoObservation" if "VectorSensor_size" in obs_spec.name else obs_spec.name)
                obs_idx_map[name] = i
                print(f"\t\tName={obs_spec.name} | Shape={obs_spec.shape} |"
                      f"Type={obs_spec.observation_type.name}")
        behavior_name = list(env.behavior_specs)[0]

        dataloader_inference = DataLoader(dataset, batch_size=1, shuffle=False)
        iterator = iter(dataloader_inference)
        if args.replay_data:
            for _ in np.arange(0, args.num_eps):
                step = 0
                env.reset()
                # simpilot_report = env_string.parameters["EpisodeReport"]
                decision_steps, terminal_steps = env.get_steps(behavior_name=behavior_name)
                tracked_agent = -1
                done = False
                lane_change_sent = False
                lane_target = None
                for _ in range(len(dataset)):

                    ego_obs = decision_steps.obs[obs_idx_map["EgoObservation"]][0]
                    position_x = ego_obs[2]
                    position_y = ego_obs[3]
                    velocity_x = ego_obs[4]
                    continuous_lane_id = ego_obs[11]
                    lane_relative_t = ego_obs[12]
                    angle_to_lane = ego_obs[13]
                    ego_collision_type = ego_obs[15]

                    total_distance += velocity_x * TIME_PER_STEP

                    if tracked_agent == -1 and len(decision_steps) >= 1:
                        tracked_agent = decision_steps.agent_id[0]

                    df_stacked, stacked_images, future_points = next(iterator)
                    movable_obj_sorted = df_stacked[0, 0, args.dim_input_feature:].numpy().reshape((NUM_MOVE_OBJS, len(move_obj_columns_hybrid)))
                    lane_id_map = torch.squeeze(stacked_images)[0].numpy()
                    future_pose_x = torch.squeeze(future_points)[0:args.num_poses]
                    future_pose_y = torch.squeeze(future_points)[args.num_poses:2 * args.num_poses]
                    future_pose_v = torch.squeeze(future_points)[2 * args.num_poses:3 * args.num_poses]
                    if args.visu and (step % args.vis_rate == 0):
                        fig_dy_obj = visualize_dy_objects(args,
                                                          fig_dy_obj,
                                                          lane_id_map,
                                                          movable_obj_sorted,
                                                          future_points=future_points
                                                         )
                        

                    lane_id_image = lane_id_map
                    lanes = pose_to_lane(args, img_height=lane_id_image.shape[0],
                                         img_width=lane_id_image.shape[1], lane_id_map=lane_id_image,
                                         future_points=future_points)

                    ego_info = df_stacked[0, 0, 0:args.dim_input_feature]
                    lanes_command = lanes_to_commands(args, lanes, ego_info)
                    travel_assist_command, lane_change_sent, lane_target = \
                        lanes_to_travel_assist(args,
                                               lanes,
                                               lanes_command,
                                               ego_info,
                                               lane_change_sent,
                                               lane_target
                                               )

                    print(f"Lanes = {lanes} *** "
                          f"commands = {lanes_command} *** "
                          f"travel_assist_command = {travel_assist_command} *** "
                          f"lane_change_sent = {lane_change_sent}")

                    if args.sumo:
                        if args.controller in ("TravelAssist", "TravelAssistUnsafe"):
                            future_pose_v = torch.squeeze(future_points)[10:15].cpu().numpy()
                            speed_action = future_pose_v[-1]
                            speed_action /= MAX_SPEED_TRAVEL_ASSIST

                            actions = spec.action_spec.empty_action(n_agents=decision_steps.agent_id.size)
                            actions.add_discrete(np.expand_dims([travel_assist_command], axis=0))
                            actions.add_continuous(np.expand_dims([speed_action], axis=0))
                            env.set_actions(behavior_name=behavior_name, action=actions)

                    if step % args.print_rate == 0:
                        print(f" *********** Step = {step} "
                              f" *********** Target Speed = {speed_action} "
                             )

                    try:
                        env.step()
                        if args.track:
                            inference_metrics(velocity_x, speed_limit)

                    except UnityCommunicatorStoppedException:
                        exit(" ********************* Exit: UnityCommunicatorStoppedException"
                            " *********************")
                    decision_steps, terminal_steps = env.get_steps(behavior_name=behavior_name)
                    step += 1

        env.close()

    exit(" ********************* Exit: Done with Inference *********************")

def run_model(args, model_infer):
    """_summary_

    Args:
        args (_type_): _description_
        model_infer (_type_): _description_
    """
    fig_obj, sc_trj, fig_dy_obj, fig_dy_obj_res = None, None, None, None
    if args.model_name[-5:] == ".ckpt":
        checkpoint = torch.load(os.path.join(args.model_path, args.model_name))
        model_infer.load_state_dict(checkpoint["state_dict"])
    else:
        model_infer.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name)))
    model_infer.eval()
    env_simpilot = SimPilotEnv(args=args,
                               exec_name=args.exec_path,
                               )
    env = env_simpilot.load_env_unity()
    env_sumo = env_simpilot.load_env_sumo()
    env_visualize = env_simpilot.load_env_visualization()
    env_string = env_simpilot.load_env_string_channel()

    crashed = False
    x_old, y_old = 0, 0
    current_yaw_deg = 0
    start_time = time.time()
    data_frame_episode = []
    reset = True
    lane_change_sent = False
    previous_lane = None

    # stack_time = args.stack_time # time in ms (5000 ms: default)
    stack_steps_total = int(args.stack_time / args.sim_steptime)
    # should be 5000 / 20 = 250

    stack_steps = int(stack_steps_total / (args.num_framestack - 1))
    # should be 250 / 6-1 = 50

    stacked_indices = np.arange(0, stack_steps_total + stack_steps, stack_steps)

    # ************************** Adding expert to inference
    driver = RuleBasedDriver()
    driver.set_dist_lane_change(5)

    # Dict to get observations by name
    obs_idx_map = dict()
    for key in env.behavior_specs:
        print(f"Behavior: {key}")
        spec = env.behavior_specs[key]
        print(f"\tAction Spec: discrete_size={spec.action_spec.discrete_size}; continuous_size={spec.action_spec.continuous_size}")
        print("\tObservation Specs:")
        for (i, _) in enumerate(spec.observation_specs):
            obs_spec = spec.observation_specs[i]
            name = ("EgoObservation" if "VectorSensor_size" in obs_spec.name else obs_spec.name)
            obs_idx_map[name] = i
            print(f"\t\tName={obs_spec.name} | Shape={obs_spec.shape} |"
                    f"Type={obs_spec.observation_type.name}")
    behavior_name = list(env.behavior_specs)[0]

    collection_time = int(time.time())

    for eps in np.arange(0, args.num_eps):
        if args.evaluate:
            df = pd.DataFrame(None, columns=HEADERS_TO_SAVE_INFERENCE)
        step = 0
        init_x, init_y = None, None
        init_time = time.time()
        completed_loop = False
        num_lane_change = 0
        total_distance = 0
        total_num_lane_changes = 0
        past_dfstack = torch.zeros(stack_steps_total + 1,
                                   args.dim_input_feature +
                                   NUM_MOVE_OBJS * len(move_obj_columns))
        past_imgstack = torch.zeros(stack_steps_total + 1, args.img_height, args.img_width, 1)
        # fig_scatter = plt.figure(figsize=(8, 8))
        # ax_scatter = fig_scatter.add_subplot(111)
        # ax_scatter.grid(True)
        # plt.show(block=False)

        # if args.randomization_env:
        #     if args.randomization_laneid == 0:
        #         laneid_wanted = np.random.randint(3, 6)
        #     else:
        #         laneid_wanted = args.randomization_laneid
        #     spawnpath = args.spawnpoints_path
        #     df_rand = pd.read_csv(spawnpath, delimiter=',')
        #     if laneid_wanted == 0:
        #         df_filtered = df_rand
        #     else:
        #         df_filtered = df_rand[df_rand['continuous_lane_id'] == laneid_wanted].sample(frac=1)
        #     sampled_row = df_filtered.sample(n=1)
        #     env.reset()
        #     env_simpilot.agent_channel.set_init_transform(x=float(sampled_row.px.values),
        #                                                   y=float(sampled_row.py.values),
        #                                                   yaw=float(sampled_row.yaw.values))
        env.reset()
        # simpilot_report = env_string.parameters["EpisodeReport"]
        decision_steps, terminal_steps = env.get_steps(behavior_name=behavior_name)

        tracked_agent = -1
        done = False
        lane_change_sent = False
        lane_target = None
        # ************* FRAMESTACKING AT INIT
        lane_id_map = decision_steps.obs[obs_idx_map["LaneIdSensor"]][0]
        if lane_id_map.shape[2] == 3:
            lane_id_map = convert_image_to_lane_ids(lane_id_map)
        image = torch.FloatTensor(lane_id_map)
        past_imgstack[0] = image

        # Retrieve ego observations
        ego_obs = decision_steps.obs[obs_idx_map["EgoObservation"]][0]
        position_x = ego_obs[2]
        position_y = ego_obs[3]
        velocity_x = ego_obs[4]
        continuous_lane_id = ego_obs[11]
        lane_relative_t = ego_obs[12]
        angle_to_lane = ego_obs[13]
        vehicle_switching_lane = ego_obs[14]
        ego_collision_type = ego_obs[15]
        left_lane_available = ego_obs[26]
        right_lane_available = ego_obs[27]
        allowed_speed = ego_obs[28]
        acc_target_speed = ego_obs[29]
        travel_assist_lane_change_state = ego_obs[30]

        # Retrieve movable object observations
        movable_obj = decision_steps.obs[obs_idx_map["MovableObjects"]][0]
        # Sorting movable objects
        movable_obj_sorted = movable_obj[np.argsort(-np.sqrt(movable_obj[:, 1] ** 2 + movable_obj[:, 2] ** 2))]
        # total_distance += velocity_x * TIME_PER_STEP
        init_x = position_x
        init_y = position_y

        # Retrieve static lane observations
        static_lanes = decision_steps.obs[obs_idx_map["StaticLanes"]][0].astype(np.float16)
        speed_limit = static_lanes[0][1]

        df_data = torch.FloatTensor(
                                    np.hstack([
                                    velocity_x,
                                    continuous_lane_id,
                                    vehicle_switching_lane,
                                    left_lane_available,
                                    right_lane_available,
                                    speed_limit,
                                    movable_obj_sorted[:, list(move_obj_columns.values())].flatten()
                                    ]))

        past_dfstack[0] = df_data
        lane_action_ta = 0
        while done is False:
            total_distance += velocity_x * TIME_PER_STEP

            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]
            df_stack = torch.stack([past_dfstack[i] for i in stacked_indices]).unsqueeze(0)
            stacked_images = torch.stack([past_imgstack[i] for i in stacked_indices]).squeeze().unsqueeze(0)
           
            with torch.no_grad():
                if args.algo == "BC":
                    if args.travelassist_pred:
                        if args.single_head:
                            predicted_pose, speed_action_command, lane_change_command_logit = \
                                model_infer(image=stacked_images.to(device),
                                            nparray=df_stack.to(device))
                        else:
                            predicted_pose, predicted_velocity, lane_change_command_logit, pred_car_matrix = \
                                model_infer(image=stacked_images.to(device),
                                            nparray=df_stack.to(device))

                    else:
                        if args.single_head:
                            predicted_pose = model_infer(image=stacked_images.to(device),
                                                         nparray=df_stack.to(device))
                        else:
                            predicted_pose, predicted_velocity = \
                                model_infer(image=stacked_images.to(device),
                                            nparray=df_stack.to(device))

                if args.algo == "GAN":
                    predicted_pose = model_infer(image=stacked_images.to(device),
                                                 nparray=df_stack.to(device),
                                                 )

            if args.bezier:
                # control_points = predicted_pose[:, :BEZIER_DIM]
                control_points = predicted_pose
                zero_tensor = torch.zeros(1, 2)
                control_points = torch.cat((zero_tensor.cuda(), control_points), 1)
                curve_points = quartic_bezier_curve(control_points, args.poly_points)
                curve_points = curve_points.detach().cpu().numpy()
                future_pose_x = curve_points[:, list(np.linspace(0,
                                args.poly_points,
                                args.num_poses+1)[1:].astype(np.int16)-1), 0]
                future_pose_y = curve_points[:, list(np.linspace(0,
                                args.poly_points,
                                args.num_poses+1)[1:].astype(np.int16)-1), 1]
                # future_pose_v = predicted_pose[:, BEZIER_DIM:].cpu().numpy()
                future_pose_v = predicted_velocity.cpu().numpy()
                future_pose_v = np.squeeze(future_pose_v)
            else:
                future_pose_x = torch.squeeze(predicted_pose)[0:args.num_poses].cpu().numpy()
                future_pose_y = torch.squeeze(predicted_pose)[args.num_poses:2 * args.num_poses].cpu().numpy()
                future_pose_v = predicted_velocity.cpu().numpy()
                future_pose_v = np.squeeze(future_pose_v)

            future_pose_yaw = np.zeros_like(future_pose_v)
            future_pose_yaw = np.arctan(future_pose_y / future_pose_x)

            if args.controller == "TeleportController" and not args.sumo:
                # ***** FIRST few steps are trash! For teleporter
                local_pose = [0, 0, 0]
                actions_trj = spec.action_spec.empty_action(n_agents=1)
                actions_trj.add_continuous(np.expand_dims(local_pose, axis=0))
                env.set_actions(behavior_name=behavior_name, action=actions_trj)

            if args.controller == "SumoController" and args.sumo:
                interest_point_pose = 1
                env_sumo.setSpeed('EgoCar_0', future_pose_v[interest_point_pose])

            if args.controller in ("TravelAssist", "TravelAssistUnsafe") and args.sumo:
                if args.travelassist_command:
                    speed_action = speed_action_command.item()
                    lane_action_ta = torch.argmax(lane_change_command_logit, axis=1).item()

                else:
                    future_pose_x = np.abs(future_pose_x)
                    future_pose_v = np.abs(future_pose_v)
                    speed_action = future_pose_v[0]
                    speed_action /= MAX_SPEED_TRAVEL_ASSIST

                    # lane_change_logit shape is [batch_size = 1, future_num poses = 5, num_classes = 4]
                    # argmaxing on dim = 2 means we are going to take the highest logit value based on each class (l,r,c,t)
                    # thus returning [1, 5] tensor. Now we take action based on the first entry of the
                    # five future "actions", hence why the [0][0].TA_Multi_transformer_Bezier_CarNetResidual_BC_encoder_custom_act_ReLU_opt_Singleopt_2023_06_23-02_00_epoch=199.ckpt
                    lane_action_ta = lane_change_command_logit.argmax(2)[0][0].item()
                    if lane_action_ta == TRANSITION_TRAVEL_ASSIST:
                        lane_action_ta = CURRENT_TRAVEL_ASSIST
                    if args.print_flag:
                        print(
                            f"Lane_assist_command = {lane_action_ta} *** "
                            f"Lane_change_sent = {lane_change_sent}")

                if args.adaptive_cruise_control and (args.controller in ("TravelAssist", "TravelAssistUnsafe")):
                    speed_action = (speed_limit + 1.4) / MAX_SPEED_TRAVEL_ASSIST
                    # speed_action = (speed_limit + 0) / MAX_SPEED_TRAVEL_ASSIST

                # ***************** Adding expert - rule-based for speed adjustment
                if args.rule_based:
                    # driver.set_dist_lane_change(np.random.uniform(5, 10))
                    driver.set_safe_dist_front(np.random.uniform(20, 30))
                    objs = driver.get_near_objs(movable_obj, continuous_lane_id)
                    speed_action = driver.keep_current_lane(objs,
                                                            velocity_x,
                                                            acceleration_x,
                                                            speed_limit)
                    speed_action /= MAX_SPEED_TRAVEL_ASSIST

                actions = spec.action_spec.empty_action(n_agents=decision_steps.agent_id.size)
                actions.add_discrete(np.expand_dims([lane_action_ta], axis=0))
                actions.add_continuous(np.expand_dims([speed_action],
                                                      axis=0))
                env.set_actions(behavior_name=behavior_name, action=actions)

                if lane_action_ta != 0 and ta_map_new[travel_assist_lane_change_state] == "None":
                    total_num_lane_changes += 1
                    # print(f"Lane action taken is {lane_action_ta} "
                    #       f"Controller status is {ta_map_new[travel_assist_lane_change_state]}")
                    # print(f"Lane action taken is {lane_action_ta} "
                    #       f"Controller status is {ta_map_new[travel_assist_lane_change_state]}")

                if step % args.print_rate == 0:
                    print(
                        f"Episode = {eps}",
                        f"Step = {step}",
                        f"Target Speed = {(speed_action * MAX_SPEED_TRAVEL_ASSIST):.2f} *****",
                        f"Speed Limit = {speed_limit:.2f} **** ",
                        f"Current speed = {velocity_x:.2f} **** ",
                        f"Total_distance = {total_distance:.1f} **** ",
                        f"Total_num_lane_changes = {total_num_lane_changes}",)

            future_control_points = env_simpilot.pose_to_control(future_pose_x,
                                                                 future_pose_y,
                                                                 future_pose_v,
                                                                 future_pose_yaw)
            # image_frame = torch.squeeze(stacked_images)[0]
            if args.visu and (step % args.vis_rate == 0):
                fig_dy_obj = visualize_dy_objects(args,
                                                  fig_dy_obj,
                                                  lane_id_map,
                                                  movable_obj,
                                                  future_points=np.concatenate((future_pose_x,
                                                                                future_pose_y),
                                                                                axis=None)
                                                    )
            if args.bezier:
                env_visualize.visualize_points(future_control_points)

            try:
                env.step()
            except UnityCommunicatorStoppedException:
                exit(" ********************* Exit: UnityCommunicatorStoppedException *********************")

            decision_steps, terminal_steps = env.get_steps(behavior_name=behavior_name)

            if args.track:
                inference_metrics(velocity_x, speed_limit)
            if step % 50 == 0 and False:
                ax_scatter.scatter(position_x, position_y, color='blue', linewidths=0.01)
                plt.pause(0.0000001)
            # 1 loop is finished
            if step > 1000 and \
                np.sqrt((position_x - init_x) ** 2 + (position_y - init_y) ** 2) < 10 and \
                    not completed_loop:
                completed_loop = True
                done = True
                if args.track:
                    wandb.log({"Steps to finish 1 loop": step,
                               "Time to finish 1 loop": time.time() - init_time,
                               "Number of Lane Changes in 1 loop": num_lane_change})

            # ************************** UPDATE FRAMESTACK
            lane_id_map = decision_steps.obs[obs_idx_map["LaneIdSensor"]][0]
            if lane_id_map.shape[2] == 3:
                lane_id_map = convert_image_to_lane_ids(lane_id_map)
            image = torch.FloatTensor(lane_id_map)
            past_imgstack = torch.roll(past_imgstack, shifts=1, dims=0)
            past_imgstack[0] = image

            # Retrieve ego observations
            ego_obs = decision_steps.obs[obs_idx_map["EgoObservation"]][0]
            timestamp = ego_obs[0]
            position_x = ego_obs[2]
            position_y = ego_obs[3]
            velocity_x = ego_obs[4]
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
            left_lane_available = ego_obs[26]
            right_lane_available = ego_obs[27]
            allowed_speed = ego_obs[28]
            travel_assist_lane_change_state = ego_obs[30]

            # Retrieve static lane observations
            static_lanes = decision_steps.obs[obs_idx_map["StaticLanes"]][0].astype(np.float16)
            speed_limit = static_lanes[0][1]

            # Retrieve movable object observations
            movable_obj = decision_steps.obs[obs_idx_map["MovableObjects"]][0]
            movable_obj_sorted = movable_obj[np.argsort(-np.sqrt(movable_obj[:, 1] ** 2 + movable_obj[:, 2] ** 2))]

            df_data = torch.FloatTensor(
                    np.hstack([
                        velocity_x,
                        continuous_lane_id,
                        vehicle_switching_lane,
                        left_lane_available,
                        right_lane_available,
                        speed_limit,
                        movable_obj_sorted[:, list(move_obj_columns.values())].flatten()
                        ]))

            past_dfstack = torch.roll(past_dfstack, shifts=1, dims=0)
            past_dfstack[0] = df_data

            step += 1
            if tracked_agent in terminal_steps or ego_collision_type in EGO_COLLISION:
                done = True
                crashed = True

            if args.evaluate:
                df.loc[len(df.index)] = [
                    args.initials,
                    collection_time,
                    args.milestone,
                    args.task,
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
                    static_lanes.flatten(),
                    speed_limit,
                    'Sumo' if args.sumo else 'Human',
                    ego_collision_type,
                    left_lane_available,
                    right_lane_available,
                    allowed_speed,
                    movable_obj,
                    speed_action,
                    int(lane_action_ta),
                    travel_assist_lane_change_state,
                    completed_loop
                    ]

        if args.evaluate:

            env.reset()
            simpilot_report = env_string.parameters["EpisodeReport"]
            with open(os.path.join(args.rawdata_path,
                                      "{}/datafiles/{}_{}_{}_{}_{}".format(args.model_name, args.initials,
                                                                           args.milestone,
                                                                           args.task,
                                                                           collection_time,
                                                                           eps)), 'wb') as handle:
                pickle.dump(simpilot_report, handle, protocol=pickle.HIGHEST_PROTOCOL)

            df.to_pickle(os.path.join(args.rawdata_path,
                                      "{}/datafiles/{}_{}_{}_{}_{}.pkl".format(args.model_name,
                                                                               args.initials,
                                                                               args.milestone,
                                                                               args.task,
                                                                               collection_time,
                                                                               eps)))
        if ego_collision_type != 0:
            wandb.log({"epoch/Distance_travelled_before_accident": total_distance})
        else:
            wandb.log({"epoch/Distance_travelled_finishing_loop": total_distance})

        if args.track:
            log_dict = {
                "epoch_step": eps + 1,
                "epoch/Time_to_finish_epoch": time.time() - init_time,
                "epoch/Steps_to_finish_epoch": step,
                "epoch/Number_of_Lane_Changes_in_one_epoch": total_num_lane_changes,
                }
            wandb.log(log_dict)

    env.close()

    if args.evaluate:
        evaluate_data(args)
