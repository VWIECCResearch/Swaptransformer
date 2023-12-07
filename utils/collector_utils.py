"""
#################################
# Python API: Data collector Utils
#################################
"""

#########################################################
# import libraries
import os
import csv
import cv2
import numpy as np

#########################################################
# General Parameters

SELECTION_EGO_OBS = [
    "timestamp",
    "ego_id",
    "position",
    "velocity",
    "acceleration",
    "orientation",
    "heading",
    "continuous_lane_id",
    "lane_relative_t",
    "angle_to_lane",
    "vehicle_switching_lane",
    "ego_collision_type",
    "controller_state",
    "nav_turn_command_n",
    "nav_point_position_n",
    "nav_point_distance_n",
    "nav_point_angle_n",
    "nav_turn_command_n1",
    "nav_point_position_n1",
    "nav_point_distance_n1",
]

EGO_OBS_MAP = {
    "timestamp": 1,
    "ego_id": 1,
    "position": 2,
    "velocity": 2,
    "acceleration": 2,
    "orientation": 1,
    "heading": 2,
    "continuous_lane_id": 1,
    "lane_relative_t": 1,
    "angle_to_lane": 1,
    "vehicle_switching_lane": 1,
    "ego_collision_type": 1,
    "controller_state": 1,
    "nav_turn_command_n": 1,
    "nav_point_position_n": 2,
    "nav_point_distance_n": 1,
    "nav_point_angle_n": 1,
    "nav_turn_command_n1": 1,
    "nav_point_position_n1": 2,
    "nav_point_distance_n1": 1,
}


HEADERS_TO_SAVE = [
    'initials',
    'time',
    'milestone',
    'task',
    'eps',
    'step',
    'pythonTime',
    'ego_speed',
    'position_x',
    'position_y',
    'timestamp',
    'heading_x',
    'heading_y',
    'acceleration_x',
    'acceleration_y',
    'orientation',
    'continuous_lane_id',
    'lane_relative_t',
    'angle_to_lane',
    'controller_state',
    'vehicle_switching_lane',
    'traj_pose_x',
    'traj_pose_y',
    'traj_pose_v',
    'control_points',
    'static_lanes',
    'image_name',
    'speed_limit',
    'expert_type',
    'collision_type',
    'left_lane_available',
    'right_lane_available',
    'allowed_speed',
    'movable_obj',
    'speed_action',
    'lane_change_command',
    'travel_assist_lane_change_state'
]

HEADERS_TO_LOAD = ['index',
                   'step',
                   'ego_speed',
                   'lane_relative_t',
                   'continuous_lane_id',
                   'angle_to_lane',
                   'heading_x', 'heading_y',
                   'image_name',
                   'speed_limit',
                   'static_lanes',
                   'left_lane_available',
                   'right_lane_available',
                   'movable_obj',
                   'sorted_movable_obj',
                   'movable_obj_EucDist',
                   'vehicle_switching_lane',
                   'travel_assist_lane_change_state',
                   'car_matrix',
                   'future_points'
                   ]

HEADERS_TO_PREDICT = ['future_v_global_array',
                      'future_x_local_array',
                      'future_y_local_array',
                      'speed_action',
                      'lane_change_command',
                      'lane_change_command_modified',
                      'future_ta_lane_change_array'
                      ]

HEADERS_TO_SAVE_INFERENCE = [
                            'initials',
                            'time',
                            'milestone',
                            'task',
                            'eps',
                            'step',
                            'pythonTime',
                            'ego_speed',
                            'position_x',
                            'position_y',
                            'timestamp',
                            'heading_x',
                            'heading_y',
                            'acceleration_x',
                            'acceleration_y',
                            'orientation',
                            'continuous_lane_id',
                            'lane_relative_t',
                            'angle_to_lane',
                            'controller_state',
                            'vehicle_switching_lane',
                            'static_lanes',
                            'speed_limit',
                            'expert_type',
                            'collision_type',
                            'left_lane_available',
                            'right_lane_available',
                            'allowed_speed',
                            'movable_obj',
                            'speed_action',
                            'lane_change_command',
                            'travel_assist_lane_change_state',
                            'completed_loop'
                            ]


#########################################################
# Function definition


class CSVWriter:
    """_summary_
    """

    def __init__(self, filename, delimiter=";", header=None) -> None:
        self.filename = filename
        self.delimiter = delimiter
        self.header = header

    def write_row(self, row):
        """_summary_

        Args:
            row (_type_): _description_
        """
        with open(self.filename, "a", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            if self.header:
                writer.writerow(self.header)
                self.header = None
            writer.writerow(row)


def create_folder(name, index=None):
    """_summary_

    Args:
        name (_type_): _description_
        index (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if index is None:
        i = 1
    else:
        i = index
    while True:
        folder_name = f"{name}_{i}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        i += 1
    return folder_name, i


def selection_size(vect_map, selection):
    """_summary_

    Args:
        vect_map (_type_): _description_
        selection (_type_): _description_

    Returns:
        _type_: _description_
    """
    size = 0
    for s in selection:
        size += vect_map[s]
    return size


def select_from_array(vector, vect_map, selection):
    """_summary_

    Args:
        vector (_type_): _description_
        vect_map (_type_): _description_
        selection (_type_): _description_

    Returns:
        _type_: _description_
    """
    selected = {}
    ind = 0
    for s, v in vect_map.items():
        if s in selection:
            selected[s] = vector[ind: ind + v]
        ind += v
    return selected


def convert_image_to_lane_ids(img: np.ndarray):
    """_summary_

    Args:
        img (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    # Used to convert a PNG-encoded image back to a single integer value per pixel
    # Assume that the integer value has been encoded into the RGB-components
    img[:, :, 0] = img[:, :, 0] * 65536
    img[:, :, 1] = img[:, :, 1] * 256
    img = 255 * np.sum(img, axis=2, keepdims=True)
    img = img.astype(np.int32)
    return img


def make_video(path):
    """_summary_

    Args:
        path (_type_): _description_
    """
    video_name = "video.mp4"
    images = []

    os.chdir(path)
    for image in os.listdir(path):
        images.append(image)

    images = sorted(images)
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 30.00, (width, height))

    for _, image in enumerate(images):
        video.write(cv2.imread(image))

    video.release()


def args_to_wandbnanme(args, run_date_time):
    """_summary_

    Args:
        args (_type_): _description_
        run_date_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    if args.single_head:
        single_name = "Single"
    else:
        single_name = "Multi"
    if args.bezier:
        bezier_name = "Bezier"
    else:
        bezier_name = "NonBezier"
    if args.travelassist_pred:
        travel_name = "TA"
    else:
        travel_name = "NonTA"
    if args.residual:
        residual_name = "Residual"
    else:
        residual_name = "NonResidual"
    if args.multi_opt:
        opt_name = "Multiopt"
    else:
        opt_name = "Singleopt"
    if args.car_network:
        car_net = "_CarNet"
    else:
        car_net = ""

    wandb_name = f"Dev_{args.initials}_" + args.proc + '_' + travel_name + '_' + single_name + \
        '_' + args.base_model + car_net + '_' + bezier_name + '_' + residual_name + f"_{args.algo}_encoder_{args.encoder}_" \
        f"act_{args.activation}_opt_{opt_name}_{run_date_time}"
    return wandb_name
