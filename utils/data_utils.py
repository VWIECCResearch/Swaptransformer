"""
#################################
# Python API: Data collector Utils
#################################
"""

#########################################################
# import libraries
# import re
import os
import time
import shutil
import wandb
import tarfile
import torch
import pickle
import paramiko
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from config import NUM_MOVE_OBJ, MOVE_OBJ_COLUMNS, MOVE_OBJ_COLUMNS_HYBRID

#########################################################
# General Parameters

input_obs_index_dict = dict(
    timestamp=0,
    ego_id=1,
    positionX=2,
    positionY=3,
    velocityX=4,
    velocityY=5,
    accelerationX=6,
    accelerationY=7,
    orientation=8,
    headingX=9,
    headingY=10,
    continuous_lane_id=11,
    lane_relative_t=12,
    angle_to_lane=13,
    vehicle_switching_lane=14,
    ego_collision_type=15,
    controller_state=16,
    nav_turn_command_n=17,
    nav_point_position_nX=18,
    nav_point_position_nY=19,
    nav_point_distance_n=20,
    nav_point_angle_n=21,
    nav_turn_command_n1=22,
    nav_point_position_n1X=23,
    nav_point_position_n1Y=24,
    nav_point_distance_n1=25,
)

input_movable_obj_index_dict = dict(
    ID=0,
    PositionX=1,
    PositionY=2,
    VelocityX=3,
    VelocityY=4,
    Orientation=5,
    ContinuousLaneId=6,
    BoundingboxLength=7,
    BoundingboxWidth=8,
    ObjectType=9,
    LaneRelativeT=10,
    SignalState=11,
)

input_static_lanes_dict = dict(ID=0, SpeedLimit=1, LaneType=2)

input_data_index = dict(
    ego_obs=input_obs_index_dict,
    static_lanes=input_static_lanes_dict,
    movable_obj=input_movable_obj_index_dict
)

target_index_dict = dict(
    timestamp=0,
    ego_id=1,
    positionX=2,
    positionY=3,
    velocityX=4,
    velocityY=5,
    accelerationX=6,
    accelerationY=7,
    orientation=8,
    headingX=9,
    headingY=10,
    continuous_lane_id=11,
    lane_relative_t=12,
    angle_to_lane=13,
    vehicle_switching_lane=14,
    ego_collision_type=15,
    controller_state=16,
    nav_turn_command_n=17,
    nav_point_position_nX=18,
    nav_point_position_nY=19,
    nav_point_distance_n=20,
    nav_point_angle_n=21,
    nav_turn_command_n1=22,
    nav_point_position_n1X=23,
    nav_point_position_n1Y=24,
    nav_point_distance_n1=25,
)

DECIMAL_ROUND = 3
APOLLO_IP = "10.144.221.194"
POLYFIT_THRESHOLD = 0.001
CURRENT_TRAVEL_ASSIST = 0
LEFT_TRAVEL_ASSIST = 1
RIGHT_TRAVEL_ASSIST = 2
NOT_CHANGING_LANE = 2.0
TRANSITION_TRAVEL_ASSIST = 3
EGO_COLLISION = [256.0, 512.0, 262400.0, 263168.0, 262144.0]
CRASHED_DISPOSAL_STEPS = 250
TA_LANE_CHANGE_STATE = {"NONE": 0,
                        "INSTANTIATED": 1,
                        "READY_CHANGE": 2,
                        "STARTED_CHANGE": 3}

saved_csv_time = time.strftime("%Y_%m_%d-%H_%M")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################
# Function definition

def global_merge(args):
    """
    Extract all tar files, move images from processed folder to existing appollo images folder,
    concatenate pickle files into one large data file, and move processed tar files to archive
    folder.
    """
    dest = args.dest
    large_df_path = args.large_df_path
    img_folder = args.img_folder
    archive = args.archive_path

    # Extract all tar files and move to archive folder
    for file_name in glob(os.path.join(dest, "*.tar.gz")):
        tar = tarfile.open(file_name)
        tar.extractall(dest)
        shutil.move(file_name, archive)
        print(f"Extracted {file_name}")

    # Move new images from processed folder to existing appollo images folder
    for filename in os.listdir(os.path.join(dest, "processed/images/")):
        if filename.endswith(".png"):
            shutil.copy2(os.path.join(os.path.join(dest, "processed/images/"),
                                      filename),
                         img_folder)

    # Concatenate pickle files into one large data file
    if large_df_path is None:
        data = pd.DataFrame()
    else:
        data = pd.read_pickle(large_df_path)

    for file in glob(os.path.join(dest, "processed/*.pkl")):
        new_data = pd.read_pickle(file)
        data = pd.concat([data, new_data])
        print(f"Concatenating with {file}")

    data.reset_index(drop=True, inplace=True)
    data.drop('index', axis=1, inplace=True)
    data.reset_index(drop=False, inplace=True)
    if large_df_path is None:
        data.to_pickle(dest + "df.pkl")
    else:
        data.to_pickle(large_df_path)

    # Remove processed image folder
    shutil.rmtree(os.path.join(dest, "processed/"))


def local_merge(args):
    """
    Move images from processed folder to existing appollo images folder,
    concatenate pickle files into one large data file, and move processed tar files to archive
    folder.
    """
    img_folder = args.img_folder
    processed_path = args.processeddata_path
    destination = args.dest
    large_df_path = args.large_df_path

    if os.path.exists(img_folder) == False:
        os.mkdir(img_folder)

    for dest in os.listdir(processed_path): 
        # Move new images from processed folder to existing appollo images folder
        processed_file = os.path.join(processed_path, dest)
        for filename in os.listdir(os.path.join(processed_file, "processed/images/")):
            if filename.endswith(".png"):
                shutil.copy2(os.path.join(os.path.join(processed_file, "processed/images/"),
                                        filename),
                            img_folder)

    # Concatenate pickle files into one large data file
    if large_df_path is None:
        data = pd.DataFrame()
    else:
        data = pd.read_pickle(large_df_path)

    for dest in os.listdir(processed_path):
        processed_file = os.path.join(processed_path, dest)
        for file in glob(os.path.join(processed_file, "processed/*.pkl")):
            new_data = pd.read_pickle(file)
            data = pd.concat([data, new_data])
            print(f"Concatenating with {file}")

    data.reset_index(drop=True, inplace=True)
    data.drop('index', axis=1, inplace=True)
    data.reset_index(drop=False, inplace=True)
    if large_df_path is None:
        data.to_pickle(destination + "df.pkl")
    else:
        data.to_pickle(large_df_path)


def pre_process_data(args, ):
    """_summary_

    Args:
        args (_type_): _description_
    """
    file_list = []
    # Pickles are stored in datafiles
    dir_path = os.path.join(args.rawdata_path, "datafiles")
    save_path = args.processeddata_path
    dir_list = os.listdir(dir_path)

    # Concatenate each dataset after pre processing
    all_processed_data_df = (pd.DataFrame())

    for file in dir_list:
        file_join_path = os.path.join(dir_path, file)
        if os.path.isfile(file_join_path) and file.split('.')[-1] == 'pkl':
            file_list.append(file_join_path)

    move_images(os.path.join(os.path.dirname(dir_path[:-1]),
                             "images"), save_path)

    print(" ************* START Processing ALL DataFrames ************* ")

    if args.multiprocess:
        p = multiprocessing.Pool(args.num_processes)
        for file in file_list:
            # Pre-processing each .pkl file asynchronously
            p.apply_async(pre_process_file, [file, args])
        p.close()
        p.join()
    else:
        for file in tqdm(file_list):
            pre_process_file(file, args)

    print(" **************** Start merging processed episodes **************** ")
    for file in tqdm(file_list):
        processed_file_path = os.path.join(args.processeddata_path, file.split("/")[-1])
        if os.path.exists(processed_file_path):
            # Some episodes might be disposed
            processed_file = pd.read_pickle(processed_file_path)
            all_processed_data_df = pd.concat([all_processed_data_df, processed_file])
            del processed_file

    print(" **************** Removing processed episodes **************** ")
    for file in tqdm(file_list):
        processed_file_path = os.path.join(args.processeddata_path, file.split("/")[-1])
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)

    # To just reset index values
    all_processed_data_df.reset_index(drop=True, inplace=True)
    # To name a new column for index in the dataframe
    all_processed_data_df.reset_index(drop=False, inplace=True)

    all_processed_data_df.to_pickle(os.path.join(save_path,
                                                 "processed_{}_{}_{}_{}.pkl".format(args.initials,
                                                                                    args.milestone,
                                                                                    args.task,
                                                                                    saved_csv_time)
                                                 )
                                    )
    # Compress processed folder
    if args.compress:
        compress_folder(save_path,
                        args.compresseddata_path,
                        args.compress_name)
        if args.apollo:
            send_data(os.path.join(args.compresseddata_path,
                                   args.compress_name),
                      args)


def pre_process_file(file, args):
    """_summary_

    Args:
        file (_type_): _description_
        args (_type_): _description_
    """
    SIM_STEP_TIME = args.sim_steptime  # step time interval for between each step = 20ms
    POSE_STEP_TIME = args.pose_steptime  # step time between poses = 500 ms
    NUM_FUTURE_POSE = args.num_poses
    NUM_STEPS_POSE = int(POSE_STEP_TIME / SIM_STEP_TIME)
    TOTAL_NUM_STEPS_POSES = int(NUM_STEPS_POSE * NUM_FUTURE_POSE)
    # *****************************************
    # Artificially adding those lane changes
    # 0: Current lane, 1: LEFT_TRAVEL_ASSIST, 2: RIGHT_TRAVEL_ASSIST, 3: TRANSITION

    # Checking if controller rejected any lane change commands
    # episode_raw_data_df = pd.read_csv(file, sep=';')
    raw_data_df = pd.read_pickle(file)
    # image_folder = os.path.join(dir_path, dir_list[-1]) if os.path.isdir(os.path.join(dir_path, dir_list[-1])) else None
    # observation_file_name = file.split("/")[-1][:-4]  # Works for both
    # .csv and .pkl

    raw_data_df.drop(['control_points',
                      'traj_pose_x',
                      'traj_pose_y',
                      'traj_pose_v'],
                     axis=1,
                     inplace=True)

    raw_data_df["future_x_local_array"] = np.nan
    raw_data_df["future_y_local_array"] = np.nan
    raw_data_df["future_v_global_array"] = np.nan
    raw_data_df["future_points"] = np.nan
    raw_data_df["future_ta_lane_change_array"] = np.nan
    raw_data_df["car_matrix"] = np.nan

    raw_data_df["future_x_local_array"] = raw_data_df["future_x_local_array"].astype(
        "object")
    raw_data_df["future_y_local_array"] = raw_data_df["future_y_local_array"].astype(
        "object")
    raw_data_df["future_v_global_array"] = raw_data_df["future_v_global_array"].astype(
        "object")
    raw_data_df["future_points"] = raw_data_df["future_points"].astype(
        "object")
    raw_data_df["future_ta_lane_change_array"] = \
        raw_data_df["future_ta_lane_change_array"].astype("object")
    raw_data_df["car_matrix"] = raw_data_df["car_matrix"].astype(
        "object")

    controller_reject = False
    raw_data = raw_data_df[raw_data_df["lane_change_command"] != 0]
    lst_commands = raw_data["continuous_lane_id"].to_numpy()

    for j in range(len(lst_commands) - 1):
        if lst_commands[j] == lst_commands[j + 1]:
            controller_reject = True
            break
    if controller_reject:
        print("Episode disposed")
        return

    raw_data_df["lane_change_command_modified"] = raw_data_df["lane_change_command"].copy()
    raw_data_df.reset_index(drop=True, inplace=True)
    if args.add_lane_changes != 0:

        lane_change_indices = \
            raw_data_df["lane_change_command"].to_numpy().nonzero()

        for ind in lane_change_indices[0]:
            for shift_ind in np.arange(1, args.add_lane_changes + 1):

                if ind - shift_ind >= 0 and \
                    raw_data_df.iloc[ind - shift_ind]["eps"] == raw_data_df.iloc[ind]["eps"]:

                    raw_data_df.at[ind - shift_ind, "lane_change_command_modified"] = \
                        raw_data_df.at[ind, "lane_change_command_modified"]

            shift_ind = 1
            while True:
                if ind + shift_ind < len(raw_data_df) and \
                    raw_data_df.iloc[ind + shift_ind]["eps"] == raw_data_df.iloc[ind]["eps"] and \
                        raw_data_df.iloc[ind + shift_ind]["travel_assist_lane_change_state"] in \
                            (TA_LANE_CHANGE_STATE["INSTANTIATED"],
                                TA_LANE_CHANGE_STATE["READY_CHANGE"]):
                    raw_data_df.at[ind + shift_ind, "lane_change_command_modified"] = \
                        raw_data_df.at[ind, "lane_change_command_modified"]
                    shift_ind += 1
                else:
                    break

        raw_data_df["lane_change_command_modified"] = \
            np.where(raw_data_df["travel_assist_lane_change_state"] == \
                TA_LANE_CHANGE_STATE["STARTED_CHANGE"],
                TRANSITION_TRAVEL_ASSIST,
                raw_data_df["lane_change_command_modified"])

        raw_data_df["lane_change_command"] = \
            raw_data_df["lane_change_command"].astype(int)
        raw_data_df["lane_change_command_modified"] = \
            raw_data_df["lane_change_command_modified"].astype(int)
    # *****************************************
    raw_data_df['sorted_movable_obj'] = raw_data_df['movable_obj'].apply(lambda r: \
        r[np.argsort(-np.sqrt(r[:, 1] ** 2 + r[:, 2] ** 2))])

    raw_data_df['movable_obj_EucDist'] = raw_data_df['movable_obj'].apply(lambda r: \
        -np.sort(-np.sqrt(r[:, 1] ** 2 + r[:, 2] ** 2)))

    for row in range(len(raw_data_df)):
        if (row + TOTAL_NUM_STEPS_POSES) < len(raw_data_df):
            global_pose_x, global_pose_y = raw_data_df.at[row,
                                                            'position_x'], raw_data_df.at[row, 'position_y']

            future_x_global = raw_data_df.loc[range(row + NUM_STEPS_POSE,
                                                    row + NUM_STEPS_POSE + TOTAL_NUM_STEPS_POSES,
                                                    NUM_STEPS_POSE), 'position_x']
            future_y_global = raw_data_df.loc[range(row + NUM_STEPS_POSE,
                                                    row + NUM_STEPS_POSE + TOTAL_NUM_STEPS_POSES,
                                                    NUM_STEPS_POSE), 'position_y']

            future_x_relative = future_x_global - global_pose_x
            future_y_relative = future_y_global - global_pose_y

            # IF WE CONSIDER ALL FUTURE POSE ORIENTATIONT FOR CALCULATION
            # future_orientation_global = all_raw_data_df.loc[range(row+NUM_STEPS_POSE,
            #                                                       row+NUM_STEPS_POSE+TOTAL_NUM_STEPS_POSES,
            # NUM_STEPS_POSE), 'orientation']

            # IF WE CONSIDER ONLY THE CURRENT POSE ORIENTATIONT FOR
            # CALCULATION
            current_orientation_global = raw_data_df.loc[row,
                                                            'orientation']

            future_x_local = np.cos(current_orientation_global) * future_x_relative + np.sin(
                current_orientation_global) * future_y_relative
            future_y_local = np.cos(current_orientation_global) * future_y_relative - np.sin(
                current_orientation_global) * future_x_relative
            future_v_global = raw_data_df.loc[range(row + NUM_STEPS_POSE,
                                                    row + NUM_STEPS_POSE + TOTAL_NUM_STEPS_POSES,
                                                    NUM_STEPS_POSE), 'ego_speed']

            future_ta_lane_change_array = raw_data_df.loc[range(row,
                                                                row + TOTAL_NUM_STEPS_POSES,
                                                                NUM_STEPS_POSE),
                                                            'lane_change_command_modified']

            future_x_local_array = future_x_local.to_numpy()
            future_y_local_array = future_y_local.to_numpy()
            future_v_global_array = future_v_global.to_numpy()
            future_ta_lane_change_array = future_ta_lane_change_array.to_numpy()

            future_x_local_array = np.round(future_x_local_array, DECIMAL_ROUND)
            future_y_local_array = np.round(future_y_local_array, DECIMAL_ROUND)

            x_new, y_new = compute_bezier(np.insert(future_x_local_array, 0, 0), np.insert(future_y_local_array, 0, 0)
                                         , args.poly_points, args.time_based)
            future_points = np.concatenate((x_new, y_new))

            if args.car_network:
                cur_movable_object = raw_data_df.loc[row, 'sorted_movable_obj'][:, list(MOVE_OBJ_COLUMNS.values())]
                x = np.insert(cur_movable_object[:, 0], 0, 0)
                y = np.insert(cur_movable_object[:, 1], 0, 0)
                car_positions = np.stack([x, y], 1)
                car_matrix = calculate_distance_matrix(car_positions)
            else:
                car_matrix = np.zeros(1)

            raw_data_df.at[row, 'future_x_local_array'] = future_x_local_array
            raw_data_df.at[row, 'future_y_local_array'] = future_y_local_array
            raw_data_df.at[row, 'future_v_global_array'] = np.round(
                future_v_global_array)
            raw_data_df.at[row, 'future_points'] = future_points
            raw_data_df.at[row, 'future_ta_lane_change_array'] = future_ta_lane_change_array
            raw_data_df.at[row, 'car_matrix'] = car_matrix

            # Episode disposal
            if raw_data_df.loc[row, 'collision_type'] == 262144:
                # Collision has happened
                for ind in range(max(row - CRASHED_DISPOSAL_STEPS + 1, 0), row + 1):
                    raw_data_df.at[ind, 'eps'] = np.nan
                break

    rows_to_delete, _ = np.where(raw_data_df.isna())
    raw_data_df.drop(rows_to_delete.tolist(), axis=0, inplace=True)
    processed_file_path = os.path.join(args.processeddata_path, file.split("/")[-1])
    raw_data_df.to_pickle(processed_file_path)
    file_name = file.split("/")[-1]
    print(f"Total number of records processed {len(raw_data_df)} for file {file_name}")
    del raw_data_df

def move_images(source, destination):
    """_summary_
        Moving images to the processed folder in a single file
    Args:
        source (_type_): _description_
        destination (_type_): _description_
    """
    if os.path.exists(destination) is False:
        os.mkdir(destination)

    destination = os.path.join(destination, "images")

    if os.path.exists(destination) is False:
        os.mkdir(destination)
    print(" ************* MOVING ALL IMAGES ************* ")
    for subfolder in tqdm(os.listdir(source)):
        subfolder_path = os.path.join(source, subfolder)

        if os.path.isdir(subfolder_path):

            image_list = (
                glob(subfolder_path + "/*.jpg")
                + glob(subfolder_path + "/*.jpeg")
                + glob(subfolder_path + "/*.png")
            )

            if image_list:
                for image_path in image_list:

                    image_name = image_path.split("/")[-1]

                    new_image_path = os.path.join(destination, image_name)

                    shutil.copy2(image_path, new_image_path)


def compress_folder(source, destination, archive_file_name):
    """_summary_

    Args:
        source (_type_): _description_
        destination (_type_): _description_
        archive_file_name (_type_): _description_
    """
    if source[-1] == "/":
        source = source[:-1]

    with tarfile.open(os.path.join(destination, archive_file_name), mode='w:gz') as tar:
        tar.add(source, arcname=os.path.basename(source))
    print(f'Successfully compressed {source} to {archive_file_name} in {destination}')


def send_data(data_path, args):
    """_summary_

    Args:
        data_path (_type_): _description_
        args (_type_): _description_
    """
    device_host = APOLLO_IP
    device_user = args.apollo_user
    device_pass = args.apollo_pass

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(device_host, username=device_user, password=device_pass)

    sftp = client.open_sftp()
    sftp.put(data_path, os.path.join(args.dest, args.compress_name))
    sftp.close()


def compute_bezier(x_local, y_local, poly_points, time_based):
    """_summary_

    Args:
        x_local (_type_): _description_
        y_local (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_val, y_val = x_local, y_local
    x_val = interpolate_array(x_val, poly_points)
    y_val = interpolate_array(y_val, poly_points)
    t_val = np.linspace(0, 2.5, poly_points)
    num_of_degrees = 17

    if sum(x_val) > POLYFIT_THRESHOLD:
        lowest_mse = -1
        top_deg = 0

        for degree in range(1, num_of_degrees):

            if time_based:
                x_fit = np.polyfit(t_val, x_val, deg=degree)
                func_x = np.poly1d(x_fit)
                x_new = func_x(t_val)
                y_fit = np.polyfit(t_val, y_val, deg=degree)
                func_y = np.poly1d(y_fit)
                y_new = func_y(t_val)
            else:
                z_fit = np.polyfit(x_val, y_val, deg=degree)
                func = np.poly1d(z_fit)
                x_new = np.linspace(x_val[0], x_val[-1], poly_points)
                y_new = func(x_new)
                xsqrd = (x_val - x_new) ** 2
                ysqrd = (y_val - y_new) ** 2
                total_mse = np.sqrt(xsqrd + ysqrd).mean()
                if total_mse < lowest_mse or lowest_mse == -1:
                    lowest_mse = total_mse
                    top_deg = degree

        z_fit = np.polyfit(x_val, y_val, deg=top_deg)
        func = np.poly1d(z_fit)
        x_new = np.linspace(x_val[0], x_val[-1], poly_points)
        y_new = func(x_new)

    else:
        x_new = np.zeros(poly_points)
        y_new = np.zeros(poly_points)

    return x_new, y_new


def interpolate_array(arr, num_points):
    """_summary_

    Args:
        arr (_type_): _description_
        num_points (_type_): _description_

    Returns:
        _type_: _description_
    """
    indices = np.linspace(0, len(arr) - 1, num_points)
    interpolated_values = np.interp(indices, range(len(arr)), arr)
    return interpolated_values


def calculate_distance_matrix(car_positions : np.array):
    """_summary_

    Args:
        car_positions (np.array): _description_

    Returns:
        _type_: _description_
    """
    # Make sure ego is inserted at position 0
    num_vehicles = car_positions.shape[0]
    distance_matrix = -1 * np.ones((num_vehicles, num_vehicles))
    num_vehicles = len(np.where(np.sum(car_positions[1:], 1) != 0)[0]) + 1

    for i in range(num_vehicles):
        for j in range(num_vehicles):
            if np.all(car_positions[i] != 0) or np.all(car_positions[j] != 0):
                distance_matrix[i, j] = np.linalg.norm(car_positions[i] - car_positions[j])
            if i == j:
                distance_matrix[i, j] = 0
    return distance_matrix


def cubic_bezier_curve(control_points, num_points):
    """_summary_

    Args:
        control_points (_type_): _description_
        num_points (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Reshape control_points to [BATCH, 4, 2]
    control_points = control_points.view(-1, 4, 2)

    # Create a tensor of shape [batch_size, 5, 1] containing values from 0 to 1
    t = torch.linspace(0, 1, num_points).view(1, num_points, 1).cuda()

    # Calculate the coefficients for the cubic Bezier curve
    t_1 = 1 - t
    coeff_1 = t_1**2 * t_1
    coeff_2 = 3 * t_1**2 * t
    coeff_3 = 3 * t_1 * t**2
    coeff_4 = t**3

    # calculate the points on the curve
    points = coeff_1 * control_points[:, 0, :].unsqueeze(1) + \
            coeff_2 * control_points[:, 1, :].unsqueeze(1) + \
            coeff_3 * control_points[:, 2, :].unsqueeze(1) + \
            coeff_4 * control_points[:, 3, :].unsqueeze(1)

    return points


def quartic_bezier_curve(control_points, num_points):
    """_summary_

    Args:
        control_points (_type_): _description_
        num_points (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Reshape control_points to [batch_size, 5, 2]
    control_points = control_points.view(-1, 5, 2)

    # Create a tensor of shape [batch_size, num_points, 1] containing values
    # from 0 to 1
    t = torch.linspace(0, 1, num_points).view(-1, num_points, 1).cuda()

    # Calculate the coefficients for the quartic Bezier curve
    t_1 = 1 - t
    coeff_1 = t_1**4
    coeff_2 = 4 * t_1**3 * t
    coeff_3 = 6 * t_1**2 * t**2
    coeff_4 = 4 * t_1 * t**3
    coeff_5 = t**4

    # Calculate the points on the curve
    points = coeff_1 * control_points[:, 0, :].unsqueeze(1) + \
             coeff_2 * control_points[:, 1, :].unsqueeze(1) + \
             coeff_3 * control_points[:, 2, :].unsqueeze(1) + \
             coeff_4 * control_points[:, 3, :].unsqueeze(1) + \
             coeff_5 * control_points[:, 4, :].unsqueeze(1)

    return points


class PickleDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
            self,
            file_path,
            image_folder,
            column_names,
            transform,
            predict_columns,
            num_framestack,
            dim_input_feature,
            args):

        print(" ********** READING DATASET ********** ")
        self.data = pd.read_pickle(file_path)
        self.data = self.data[column_names + predict_columns]
        self.len = len(self.data)
        # 0 means current lane, (1, 2) means (left or right), 3 means transition
        self.indices_c = np.array(self.data[self.data['lane_change_command_modified'] == 0].index, dtype=np.int32)
        self.indices_lr = np.array(self.data[self.data['lane_change_command_modified'].isin([1, 2])].index, dtype=np.int32)
        self.indices_t = np.array(self.data[self.data['lane_change_command_modified'] == TRANSITION_TRAVEL_ASSIST].index, dtype=np.int32)

        self.data_step = np.array(self.data["step"], dtype=np.int32)
        self.data_image_name = np.array(self.data["image_name"], )
        self.data_ego_speed = np.array(self.data["ego_speed"], dtype=np.float32)
        self.data_continuous_lane_id = np.array(self.data["continuous_lane_id"], dtype=np.int8)
        self.data_vehicle_switching_lane = np.array(self.data["vehicle_switching_lane"], dtype=np.int8)
        self.data_left_lane_available = np.array(self.data["left_lane_available"], dtype=np.int8)
        self.data_right_lane_available = np.array(self.data["right_lane_available"], dtype=np.int8)
        self.data_speed_limit = np.array(self.data["speed_limit"], dtype=np.float32)
        self.data_sorted_movable_obj = np.array(self.data["sorted_movable_obj"], )
        self.data_future_ta_lane_change_array = np.array(self.data["future_ta_lane_change_array"], )
        self.data_future_v_global_array = np.array(self.data["future_v_global_array"], )
        self.data_future_x_local_array = np.array(self.data["future_x_local_array"], )
        self.data_future_y_local_array = np.array(self.data["future_y_local_array"], )
        self.data_future_points = np.array(self.data["future_points"], )
        self.data_car_matrix = np.array(self.data["car_matrix"], )

        self.image_folder = image_folder
        self.num_framestack = num_framestack
        self.stack_time = args.stack_time # time in ms (5000 ms: default)
        self.stack_steps_total = int(self.stack_time / args.sim_steptime)
        # should be 5000 / 20 = 250

        self.stack_steps = int(self.stack_steps_total / (num_framestack - 1))
        # should be 250 / 6-1 = 50

        self.dim_input_feature = dim_input_feature
        self.args = args
        self.move_obj_columns = MOVE_OBJ_COLUMNS
        if args.proc == "INFERENCE" and args.infer_type == 'Hybrid':
            self.move_obj_columns = MOVE_OBJ_COLUMNS_HYBRID
        self.num_move_obj = NUM_MOVE_OBJ

        self.move_obj_columns_vals = np.fromiter(self.move_obj_columns.values(), dtype=np.int8)
        del self.data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx % 3 == 0:
            idx = np.random.choice(self.indices_lr)
        elif idx % 3 == 1:
            idx = np.random.choice(self.indices_c)
        else:
            idx = np.random.choice(self.indices_t)

        # current_step = self.data.iloc[idx]['step']
        current_step = self.data_step[idx]
        # idx and step will be the same as current_step if there is only one
        # episode

        df_stacked = torch.zeros(
            self.num_framestack,
            self.dim_input_feature + self.num_move_obj * len(self.move_obj_columns)) # .to(device)
        stacked_images = torch.zeros(self.num_framestack, self.args.img_height, self.args.img_width)
        for idx_frstack, idx_frstack_step in zip(range(0, self.num_framestack),
                                                 range(0, self.stack_steps_total + self.stack_steps,
                                                       self.stack_steps)):

            if current_step - idx_frstack_step < 0:
                # Do nothing because it's already zeros
                continue
            else:

                image_path = os.path.join(
                    self.image_folder, self.data_image_name[idx - idx_frstack_step])
                image = Image.open(image_path).convert('L')

                image = torch.FloatTensor(np.asarray(image))
                # self.stack.append(image)
                # self.stack.appendleft(image)
                stacked_images[idx_frstack] = image

                # Dynamic Lane ID for movable objects
                # sorted_movable_obj = self.data_sorted_movable_obj[idx - idx_frstack]
                # sorted_movable_obj = sorted_movable_obj[:, 6] - self.data_continuous_lane_id[idx - idx_frstack_step]
                
                df_stacked[idx_frstack] = torch.FloatTensor(
                    np.hstack([
                        self.data_ego_speed[idx - idx_frstack_step],
                        self.data_continuous_lane_id[idx - idx_frstack_step],
                        self.data_vehicle_switching_lane[idx - idx_frstack_step],
                        self.data_left_lane_available[idx - idx_frstack_step],
                        self.data_right_lane_available[idx - idx_frstack_step],
                        self.data_speed_limit[idx - idx_frstack_step],
                        self.data_sorted_movable_obj[idx - idx_frstack_step][:, self.move_obj_columns_vals].flatten()
                    ]))

        future_ta_lane_changes = \
            torch.FloatTensor([self.data_future_ta_lane_change_array[idx]]).squeeze()

        future_v_global_tensor = torch.FloatTensor(
            self.data_future_v_global_array[idx])

        if self.args.bezier:
            future_x_local_tensor = torch.FloatTensor(np.insert(
                self.data_future_x_local_array[idx], 0, 0))
            future_y_local_tensor = torch.FloatTensor(np.insert(
                self.data_future_y_local_array[idx], 0, 0))
            future_points = torch.FloatTensor(self.data_future_points[idx])
        else:
            future_x_local_tensor = torch.FloatTensor(
                self.data_future_x_local_array[idx])
            future_y_local_tensor = torch.FloatTensor(
                self.data_future_y_local_array[idx])
            future_points = torch.cat([future_x_local_tensor, future_y_local_tensor]) # .to(device)
        car_matrix = torch.FloatTensor(self.data_car_matrix[idx])
        return df_stacked, stacked_images, future_points, \
            future_v_global_tensor, future_ta_lane_changes, car_matrix


def meter_to_pixel(args, img_height=None, img_width=None, point=None):
    """_summary_

    Args:
        args (_type_): _description_
        point (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ONE_PIXEL = float(2 * args.bev_size / img_height)
    ONE_METER = float(1. / ONE_PIXEL)
    pixel = None
    if point is not None and img_height is not None and img_width is not None:
        ego_loc_pixel = (int(img_height / 2 + 10 * ONE_METER - 1), int(img_width / 2))
        h_pose_pix = np.minimum(np.rint(ego_loc_pixel[0] - point[0] * ONE_METER).astype(np.int32),
                                img_height)
        w_pose_pix = np.rint(ego_loc_pixel[1] + point[1] * ONE_METER).astype(np.int32)
        w_pose_pix = min(w_pose_pix, img_width-1)
        w_pose_pix = max(w_pose_pix, 0)
        pixel = (h_pose_pix, w_pose_pix)
    return pixel


def pose_to_pixel(args, img_height=None, img_width=None, future_points=None):
    """_summary_

    Args:
        args (_type_): _description_
        future_points (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    pixels = []
    if future_points is not None:
        if isinstance(future_points, np.ndarray):
            future_pose_x = np.squeeze(future_points)[0: args.num_poses]
            future_pose_y = -np.squeeze(future_points)[args.num_poses: 2*args.num_poses]
        else:
            future_pose_x = torch.squeeze(future_points)[0: args.num_poses].cpu().numpy()
            future_pose_y = -torch.squeeze(future_points)[args.num_poses: 2*args.num_poses].cpu().numpy()
        for point_x, point_y in zip(future_pose_x, future_pose_y):
            pixel = meter_to_pixel(args,
                                   img_height=img_height,
                                   img_width=img_width,
                                   point=(point_x, point_y))
            pixels.append(pixel)
    return pixels


def pose_to_lane(args, img_height=None, img_width=None, lane_id_map=None, future_points=None):
    """_summary_

    Args:
        args (_type_): _description_
        lane_id_map (_type_, optional): _description_. Defaults to None.
        future_points (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    lanes = []
    if lane_id_map is not None and future_points is not None:
        pixels = pose_to_pixel(args,
                               img_height=img_height,
                               img_width=img_width,
                               future_points=future_points)
        for pixel in pixels:
            lane_value = np.squeeze(lane_id_map)[pixel[0], pixel[1]]
            lanes.append(lane_value)
    return lanes


def lanes_to_commands(args, lanes=None, ego_info=None):
    """_summary_

    Args:
        args (_type_): _description_
        lanes (_type_, optional): _description_. Defaults to None.
        ego_info (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    commands = []
    if lanes is not None and ego_info is not None:

        continuous_lane_id = ego_info[1]
        for lane in lanes:
            if lane == continuous_lane_id:
                command = 'C'
            elif lane == continuous_lane_id - 1:
                command = 'L'
            elif lane == continuous_lane_id + 1:
                command = 'R'
            else:
                command = None
            commands.append(command)
    return commands


def lanes_to_travel_assist(
        args,
        lanes,
        lanes_command=None,
        ego_info=None,
        lane_change_sent=False,
        lane_target=None):
    """_summary_

    Args:
        args (_type_): _description_
        lanes (_type_): _description_
        lanes_command (_type_, optional): _description_. Defaults to None.
        ego_info (_type_, optional): _description_. Defaults to None.
        lane_change_sent (bool, optional): _description_. Defaults to False.
        lane_target (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    lane_travel_assist = CURRENT_TRAVEL_ASSIST

    if lanes_command is not None and ego_info is not None:
        continuous_lane_id = ego_info[1]
        vehicle_switching_lane = ego_info[2]
        left_lane_available = ego_info[3]
        right_lane_available = ego_info[4]
        if lanes_command[-1] == 'C':
            lane_travel_assist = CURRENT_TRAVEL_ASSIST
        if vehicle_switching_lane == NOT_CHANGING_LANE and lane_change_sent is False:
            if lanes_command[-1] == 'L' and left_lane_available:
                lane_travel_assist = LEFT_TRAVEL_ASSIST
                lane_change_sent = True
                lane_target = lanes[-1]
            if lanes_command[-1] == 'R' and right_lane_available:
                lane_travel_assist = RIGHT_TRAVEL_ASSIST
                lane_change_sent = True
                lane_target = lanes[-1]
        if lane_change_sent is True:
            if continuous_lane_id == lane_target:
                lane_change_sent = False

    return lane_travel_assist, lane_change_sent, lane_target


def evaluate_data(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    print(' ************** EVALUATION STARTS NOW ************** ')
    evaluation_data_dir = os.path.join(args.rawdata_path , args.model_name, 'datafiles')
    pkl_files = [x for x in os.listdir(evaluation_data_dir) if 'pkl' == x.split('.')[-1]]
    pkl_reports = [x for x in os.listdir(evaluation_data_dir) if 'pkl' != x.split('.')[-1]]
    running_seconds = 0
    running_average_above = 0
    running_average_below = 0
    running_collision = 0
    running_left_changes = 0
    running_right_changes = 0
    running_below_1 = 0
    running_below_5_above_1 = 0
    running_above_5 = 0
    running_avg_speed_diff = 0
    running_not_completed = 0
    running_passing_left = 0
    running_passing_right = 0
    avg_finish_loop = 0
    for df_file in pkl_files:
        # raw_path = files
        df_raw = pd.read_pickle(os.path.join(evaluation_data_dir , df_file))
        # To just reset index values
        df_raw.reset_index(drop=True, inplace=True)
        # To name a new column for index in the dataframe
        df_raw.reset_index(drop=False, inplace=True)

        seconds_to_complete = len(df_raw) * 20 / 1000
        ego_speed = df_raw['ego_speed']
        speed_limit = df_raw['speed_limit']
        if args.proc != 'EXPERT':
            completed_loop = df_raw['completed_loop'][len(df_raw.index) - 1]
        else:
            completed_loop = 1
        avg_speed_diff = abs(ego_speed - speed_limit).mean()
        collision_occured = 1 if any(collision in np.unique(df_raw["collision_type"]) for collision in EGO_COLLISION) else 0

        driver_above_speed_limit = ego_speed > speed_limit
        driver_below_speed_limit = ego_speed < speed_limit
        driver_equal_speed_limit = (ego_speed == speed_limit)
        average_below = abs(df_raw[driver_below_speed_limit]['ego_speed'] - df_raw[driver_below_speed_limit]['speed_limit']).mean()
        average_above = abs(df_raw[driver_above_speed_limit]['ego_speed'] - df_raw[driver_above_speed_limit]['speed_limit']).mean()
        # Counting lane changes
        if np.isnan(average_above):
            average_above = 0
        if np.isnan(average_below):
            average_below = 0
        # Counting lane changes
        num_left_change = 0
        num_right_change = 0
        staying_curr_lane = 0
        cur_lane = 0
        for count, value in enumerate(list(df_raw['continuous_lane_id'])):
            if count == 0 :
                cur_lane = value
            if value != cur_lane:
                if value < cur_lane:
                    num_left_change += 1
                    cur_lane = value
                else:
                    num_right_change += 1
                    cur_lane = value
            else:
                staying_curr_lane += 1
        #0, 1 and 5, and anything above 5. Create folder based on modelname, read all the pickles, aggreate data also add wandb
        less_1 = abs(df_raw[driver_above_speed_limit]['ego_speed'] - df_raw[driver_above_speed_limit]['speed_limit']) < 1
        less_5 = abs(df_raw[driver_above_speed_limit]['ego_speed'] - df_raw[driver_above_speed_limit]['speed_limit']) < 5
        greater_1 = abs(df_raw[driver_above_speed_limit]['ego_speed'] - df_raw[driver_above_speed_limit]['speed_limit']) > 1
        greater_5 = abs(df_raw[driver_above_speed_limit]['ego_speed'] - df_raw[driver_above_speed_limit]['speed_limit']) > 5

        num_step_going_below_1 = sum(np.array(less_1))
        num_step_below_5_and_above_1 = sum(np.array(less_5) == np.array(greater_1))
        num_steps_above_5 = sum(np.array(greater_5))

        print(f"Finished in {seconds_to_complete} seconds, num_steps above speed limit {sum(driver_above_speed_limit)}"
              f", num_steps below speed limit {sum(driver_below_speed_limit)},"
              f" average speed above {average_above}, average speed below {average_below}")
        if not collision_occured and completed_loop:
            running_seconds += seconds_to_complete
        running_average_above += average_above
        running_average_below += average_below
        running_collision += collision_occured
        running_not_completed += (1 - completed_loop)
        running_left_changes += num_left_change
        running_right_changes += num_right_change
        running_below_1 += num_step_going_below_1
        running_below_5_above_1 += num_step_below_5_and_above_1
        running_above_5 += num_steps_above_5
        running_avg_speed_diff += avg_speed_diff

    for env_report in pkl_reports:
        cur_report = os.path.join(evaluation_data_dir, env_report)
        with open(cur_report, 'rb') as handle:
            b = pickle.load(handle)
        passing_left = b['PassingReport']['PassingsLeft']
        passing_right = b['PassingReport']['PassingsRight']
        running_passing_left += passing_left
        running_passing_right += passing_right
    num_episodes = len(pkl_files)
    if num_episodes != running_collision + running_not_completed:
        if num_episodes == running_collision:
            avg_finish_loop = 0
        else:
            avg_finish_loop = running_seconds / (num_episodes - running_collision)
    if args.track:
        # if num_episodes != running_collision + running_not_completed:
        #     avg_finish_loop = running_seconds / (num_episodes - running_collision - running_not_completed)
        wandb.log({
            "Avg. steps above speed limit (0-1 mps)": running_below_1 / num_episodes,
            "Avg. steps above speed limit (1-5 mps)": running_below_5_above_1 / num_episodes,
            "Avg. steps above speed limit (5- mps)": running_above_5 / num_episodes,
            "Avg. speed diff (speed limit and ego speed)": running_avg_speed_diff / num_episodes,
            "Avg. num of of left changes": running_left_changes / num_episodes,
            "Avg. num of right changes": running_right_changes / num_episodes,
            "Avg. time to finish loop (s)": avg_finish_loop,
            "Avg. speed above speed limit": running_average_above / num_episodes,
            "Avg. speed below speed limit": running_average_below / num_episodes,
            "Avg. num of Passing from Left": running_passing_left / num_episodes,
            "Avg. num of Passing from Right": running_passing_right / num_episodes,
            "Total number of collision ": running_collision,
                })

        wandb.finish()
    print(f"Average num of seconds = {avg_finish_loop}, "
          f"Average num above speed = {running_average_above / num_episodes}, "
          f"Average num below speed = {running_average_below / num_episodes}, "
          f"Total number of collision = {running_collision}")
