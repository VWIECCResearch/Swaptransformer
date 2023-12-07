"""
#################################
# Python API: TRAJECTORY Utility Function
#################################
"""

#########################################################
# import libraries
import math
import numpy as np
# import cv2.cv2 as cv2
try:
    from pynput.keyboard import Key, Listener
except:
    print("pynput can't be imported")
from config import Config_TRJ, Config_Expert
# from mlagents_envs.environment import UnityEnvironment

#########################################################
# General Parameters
MAX_ACC = Config_TRJ.get("MAX_ACC")
MIN_ACC = Config_TRJ.get("MIN_ACC")
MAX_KAPPA = Config_TRJ.get("MAX_KAPPA")
MIN_KAPPA = Config_TRJ.get("MIN_KAPPA")
NUM_FUTURE_TRJ = Config_TRJ.get("NUMBER_POINTS")
TRJ_LENGTH_TIME = Config_TRJ.get("TRJ_LENGTH_TIME")
NUM_EGO_ELEMENTS = Config_TRJ.get("NUM_EGO_ELEMENTS")
TRJ_TIME_INTERVAL = Config_TRJ.get("TRJ_TIME_INTERVAL")
NUM_CONTROL_ELEMENTS = Config_TRJ.get("NUM_CONTROL_ELEMENTS")

STEER_RANGE = Config_Expert.get("STEER_RANGE")


#########################################################
# Function definition

def calculate_curv(x, y):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    dx = 0.5 * (x[2] - x[0])
    dy = 0.5 * (y[2] - y[0])
    dl = np.sqrt(dx**2 + dy**2)
    # dl = 1
    x_dv = dx / dl
    y_dv = dy / dl

    x_ddv = (x[2] - 2 * x[1] + x[0]) / (dl**2)
    y_ddv = (y[2] - 2 * y[1] + y[0]) / (dl**2)
    kappa = (x_dv * y_ddv - y_dv * x_ddv) / (np.sqrt((x_dv**2 + y_dv**2)**3))
    if kappa > MAX_KAPPA:
        kappa = MAX_KAPPA
    if kappa < MIN_KAPPA:
        kappa = MIN_KAPPA
    if math.isnan(kappa):
        kappa = 0
    return kappa


def calculate_head(x, y):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    head_angle_rad = math.atan2(y[1] - y[0], x[1] - x[0])
    head_angle_deg = np.rad2deg(head_angle_rad)
    # print(head_angle_deg)
    return head_angle_deg, head_angle_rad


def calculate_acceleration(x, y, v):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        v (_type_): _description_

    Returns:
        _type_: _description_
    """
    delta_dist = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    acceleration = (v[1]**2 - v[0]**2) / (2 * delta_dist)
    if math.isnan(acceleration):
        acceleration = 0
    if acceleration > MAX_ACC:
        acceleration = MAX_ACC
    if acceleration < MIN_ACC:
        acceleration = MIN_ACC
    return acceleration


def calculate_s_ref(x, y, prev_s_ref):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        prev_s_ref (_type_): _description_

    Returns:
        _type_: _description_
    """
    s_ref = prev_s_ref + ((np.diff(x))**2 + np.diff(y)**2)**0.5
    return s_ref


def calculate_t_ref(x, y, v, prev_t_ref):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        v (_type_): _description_
        prev_t_ref (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_ref = prev_t_ref + (((np.diff(x))**2 + np.diff(y)**2)**0.5) / np.abs(v)
    if math.isnan(t_ref) or math.isinf(t_ref):
        t_ref = 0
    return t_ref


def data_ready_to_send_draft_old(
        x_ref, y_ref, vx_ref, x_old, y_old, s_old, t_old, v_old, yaw_old):
    """_summary_

    Args:
        x_ref (_type_): _description_
        y_ref (_type_): _description_
        vx_ref (_type_): _description_
        x_old (_type_): _description_
        y_old (_type_): _description_
        s_old (_type_): _description_
        t_old (_type_): _description_
        v_old (_type_): _description_
        yaw_old (_type_): _description_

    Returns:
        _type_: _description_
    """
    acc = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    heading_angle = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    curvature = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    s_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    t_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    # yaw_rate = np.zeros([num_future_trj], dtype=np.float16)

    for index in range(0, NUM_FUTURE_TRJ):
        acc[index] = calculate_acceleration(
            x_ref[index:index + 2], y_ref[index:index + 2], vx_ref[index:index + 2])
        _, heading_angle[index] = calculate_head(
            x_ref[index:index + 2], y_ref[index:index + 2])
        if index == 0:
            curvature[index] = calculate_curv(np.concatenate((np.array([x_old]), x_ref[index:index + 2])),
                                              np.concatenate((np.array([y_old]), y_ref[index:index + 2])))

            s_ref[index] = 0
            t_ref[index] = 0

        else:
            curvature[index] = calculate_curv(
                x_ref[index - 1:index + 2], y_ref[index - 1:index + 2])

            s_ref[index] = calculate_s_ref(x=x_ref[index - 1:index + 1],
                                           y=y_ref[index - 1:index + 1],
                                           prev_s_ref=s_ref[index - 1])

            t_ref[index] = calculate_t_ref(x=x_ref[index - 1:index + 1],
                                           y=y_ref[index - 1:index + 1],
                                           v=vx_ref[index],
                                           prev_t_ref=t_ref[index - 1])

    wheel_angle = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    wheel_angle_rate = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    yaw_rate = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)

    mu_s = np.ones([NUM_FUTURE_TRJ], dtype=np.float16)
    grade_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    bank_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    delta_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)

    return np.concatenate((s_ref, x_ref[:-2], y_ref[:-2], vx_ref[:-2], acc, heading_angle, curvature, wheel_angle,
                           wheel_angle_rate, t_ref), axis=0).reshape(NUM_CONTROL_ELEMENTS, NUM_FUTURE_TRJ),\
        np.concatenate((x_ref[:-2], y_ref[:-2], vx_ref[:-2] * np.cos(heading_angle), vx_ref[:-2] * np.sin(heading_angle),
                        acc * np.cos(heading_angle), acc * np.sin(heading_angle), heading_angle, yaw_rate),
                       axis=0).reshape(NUM_EGO_ELEMENTS, NUM_FUTURE_TRJ)


def data_ready_to_send(x_ref, y_ref, vx_ref):
    """_summary_

    Args:
        x_ref (_type_): _description_
        y_ref (_type_): _description_
        vx_ref (_type_): _description_

    Returns:
        _type_: _description_
    """
    delta_x = 1.5 * np.diff(x_ref[-2:])
    delta_y = 1.5 * np.diff(y_ref[-2:])
    delta_v = 1.5 * np.diff(vx_ref[-2:])

    x_ref = np.concatenate((x_ref[:], x_ref[-2:] + delta_x))
    y_ref = np.concatenate((y_ref[:], y_ref[-2:] + delta_y))
    vx_ref = np.concatenate((vx_ref[:], vx_ref[-1:], vx_ref[-1:]))

    # Adjust of having 24 points at the beginning and removing the first point
    # after all calculations

    # ***************************** Uncomment to have 24 points
    # num_future_trj_24_points = num_future_trj + 1
    # *****************************

    acc = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    heading_angle = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    curvature = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    s_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    t_ref = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)

    for index in range(0, NUM_FUTURE_TRJ):
        acc[index] = calculate_acceleration(
            x_ref[index:index + 2], y_ref[index:index + 2], vx_ref[index:index + 2])
        _, heading_angle[index] = calculate_head(
            x_ref[index:index + 2], y_ref[index:index + 2])
        # print("heading_angle[index]:", heading_angle[index])
        curvature[index] = calculate_curv(
            x_ref[index:index + 3], y_ref[index:index + 3])
        if index == 0:
            s_ref[index] = 0
            t_ref[index] = 0

        else:
            s_ref[index] = calculate_s_ref(x=x_ref[index - 1:index + 1],
                                           y=y_ref[index - 1:index + 1],
                                           prev_s_ref=s_ref[index - 1])

            t_ref[index] = calculate_t_ref(x=x_ref[index - 1:index + 1],
                                           y=y_ref[index - 1:index + 1],
                                           v=vx_ref[index],
                                           prev_t_ref=t_ref[index - 1])

    # Adjust of having 24 points at the beginning and removing
    # the first point after all calculations
    # ***************************** Uncomment to have 24 points

    # x_ref = x_ref[1:]
    # y_ref = y_ref[1:]
    # vx_ref = vx_ref[1:]
    # acc = acc[1:]
    # heading_angle = heading_angle[1:]
    # curvature = curvature[1:]
    # curvature[-1] = curvature[-2] = np.average(curvature[-4:-2])
    # s_ref = s_ref[1:]
    # t_ref = t_ref[1:]

    # *****************************
    t_ref = np.linspace(0.0, TRJ_LENGTH_TIME, NUM_FUTURE_TRJ)
    wheel_angle = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    wheel_angle_rate = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)
    yaw_rate = np.zeros([NUM_FUTURE_TRJ], dtype=np.float16)

    return np.concatenate((s_ref, x_ref[:-2], y_ref[:-2], vx_ref[:-2], acc, heading_angle, curvature, wheel_angle,
                           wheel_angle_rate, t_ref), axis=0).reshape(NUM_CONTROL_ELEMENTS, NUM_FUTURE_TRJ),\
        np.concatenate((x_ref[:-2], y_ref[:-2], vx_ref[:-2] * np.cos(heading_angle), vx_ref[:-2] * np.sin(heading_angle),
                        acc * np.cos(heading_angle), acc * np.sin(heading_angle), heading_angle, yaw_rate),
                       axis=0).reshape(NUM_EGO_ELEMENTS, NUM_FUTURE_TRJ)


def human_input_to_trajectory(ego_speed, target_speed, ego_yaw):
    """_summary_

    Args:
        ego_speed (_type_): _description_
        target_speed (_type_): _description_
        ego_yaw (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_trj, y_trj, v_trj, acc_trj, yaw_trj, delta_s_trj = np.zeros(NUM_FUTURE_TRJ, dtype=np.float16), \
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float16), \
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float16), \
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float16), \
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float32), \
        np.zeros(NUM_FUTURE_TRJ - 1, dtype=np.float16)

    yaw_trj = np.linspace(0, ego_yaw, NUM_FUTURE_TRJ)

    # acc_min, acc_max = MIN_ACC, MAX_ACC
    v_trj[0] = ego_speed
    for t_index in range(1, NUM_FUTURE_TRJ):
        acc_trj[t_index -
                1] = np.clip((target_speed -
                              v_trj[t_index -
                                    1]), a_min=MIN_ACC, a_max=MAX_ACC)
        v_trj[t_index] = v_trj[t_index - 1] + \
            acc_trj[t_index - 1] * TRJ_TIME_INTERVAL
        x_trj[t_index] = x_trj[t_index - 1] + TRJ_TIME_INTERVAL * v_trj[t_index] + \
            (1 / 2) * (TRJ_TIME_INTERVAL ** 2) * acc_trj[t_index - 1]

        y_trj[t_index] = (x_trj[t_index] - x_trj[t_index - 1]) * \
            np.tan(yaw_trj[t_index]) + y_trj[t_index - 1]

    return x_trj, y_trj, v_trj


def map_key_to_yaw(key_steer, current_yaw, key_acc, target_speed):
    """_summary_

    Args:
        key_steer (_type_): _description_
        current_yaw (_type_): _description_
        key_acc (_type_): _description_
        target_speed (_type_): _description_

    Returns:
        _type_: _description_
    """
    yaw_bins = np.linspace(-STEER_RANGE, STEER_RANGE,
                           int(2 * STEER_RANGE / Config_Expert.get("STEER_STEP")) + 1)
    current_index = int(np.where(current_yaw == yaw_bins)[0])

    if key_steer is None:
        current_index = current_index
    elif key_steer == Key.right:
        current_index = current_index - 1
    elif key_steer == Key.left:
        current_index = current_index + 1
    else:
        current_index = current_index

    if current_index < 0:
        current_index = 0
    if current_index >= yaw_bins.shape[0]:
        current_index = yaw_bins.shape[0] - 1

    new_yaw_deg = yaw_bins[current_index]

    if key_acc is Key.up:
        target_speed = target_speed + Config_Expert.get("SPEED_STEP")
    elif key_acc is Key.down:
        target_speed = target_speed - Config_Expert.get("SPEED_STEP")
    else:
        target_speed = target_speed
    if target_speed < 0:
        target_speed = 0
    if target_speed >= 25:
        target_speed = 25
    return new_yaw_deg, np.deg2rad(new_yaw_deg), target_speed


def constant_velocity_trj(ego_speed, target_speed,
                          ego_yaw, target_radius, curve_turn=False):
    """_summary_

    Args:
        ego_speed (_type_): _description_
        target_speed (_type_): _description_
        ego_yaw (_type_): _description_
        target_radius (_type_): _description_
        curve_turn (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    x_trj, y_trj, v_trj, acc_trj, yaw_trj, delta_s_trj = np.zeros(NUM_FUTURE_TRJ, dtype=np.float16),\
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float16),\
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float16), \
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float16), \
        np.zeros(NUM_FUTURE_TRJ, dtype=np.float32), \
        np.zeros(NUM_FUTURE_TRJ - 1, dtype=np.float16)

    v_trj[0] = ego_speed
    yaw_trj[0] = 0

    for t_index in range(1, NUM_FUTURE_TRJ):
        acc_trj[t_index -
                1] = np.clip((target_speed -
                              v_trj[t_index -
                                    1]), a_min=MIN_ACC, a_max=MAX_ACC)
        v_trj[t_index] = v_trj[t_index - 1] + \
            acc_trj[t_index - 1] * TRJ_TIME_INTERVAL

        if curve_turn:
            delta_s_trj[t_index - 1] = v_trj[t_index] * TRJ_TIME_INTERVAL
            yaw_trj[t_index] = yaw_trj[t_index - 1] + \
                float(delta_s_trj[t_index - 1] / target_radius)
            # print("Ego Yaw[t_index]: ", yaw_trj[t_index], " ******** ",
            # "target_radius:", target_radius, " ******** "
            #       "delta_s_trj[t_index-1]: ", delta_s_trj[t_index-1], " ******** ")
            x_trj[t_index] = x_trj[t_index - 1] + TRJ_TIME_INTERVAL * \
                v_trj[t_index] * np.cos(yaw_trj[t_index])
            y_trj[t_index] = y_trj[t_index - 1] + TRJ_TIME_INTERVAL * \
                v_trj[t_index] * np.sin(yaw_trj[t_index])
        else:
            x_trj[t_index] = x_trj[t_index - 1] + \
                TRJ_TIME_INTERVAL * v_trj[t_index]

    acc_trj[-1] = acc_trj[-2]
    prev_yaw_traj = yaw_trj[2]
    return x_trj, y_trj, v_trj, prev_yaw_traj
