"""
#################################
# Python API: Trajectory Interface for Simulation
#################################
"""

#########################################################
# import libraries
import numpy as np
from config import Config_TRJ

#########################################################
# General Parameters
# Configurable parameters for rule based driver
TIME_PER_STEP = 0.02
EPSILON = 0.0001

# Rule-based parameters
REACT_TIME = 3 # Time of reaction between the ego and vehicle in front in (s)
SPEED_DIFF = 1.5 # Speed difference that you can ignore (m/s)
SAFE_DIST_FRONT = 8 # Safe distance with the vehicle in front of ego (m)
SAFE_DIST_REAR = 6 # Safe distance with the vehicle behind ego (m)
SPEED_SECTION = 5
EMERGENCY_DIST = 2 # Emergency brake activation threshold (m)
LANE_CHANGE_TIME_LMT = 10 # Seconds
LANE_CHANGE_STEP_LMT = LANE_CHANGE_TIME_LMT / TIME_PER_STEP # Steps
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

ta_map = {0: "None",
          1: "Instantiated",
          2: "Ready to change Lane",
          3: "Started Movement",
          4: "None",
          5: "None",
          6: "None"}
# 4, 5, and 6 are for newer versions of simpilot (12.0.0 and after)


class RuleBasedDriver():
    """_summary_
    """
    def __init__(self):
        # Ego has to be at this distance * SAFE_DIST_FRONT or less close to the vehicle in front
        self.dist_lane_change = np.random.uniform(5, 12)
        self.safe_dist_front = SAFE_DIST_FRONT # Safe distance with the vehicle in front of ego (m)

    def change_lane(self, movable_objs, v, acc, target_speed,
                    lane_id, left_lane_available, right_lane_available):
        """_summary_

        Args:
            key (_type_): _description_

        Returns:
            (int, int): (change_lane, target_speed)
        """
        left_rear, left_front, cur_rear, cur_front, right_rear, right_front = movable_objs
        if cur_front is not None:
            v_front = cur_front[2]
            dist_front = cur_front[0]

            # Emergency brake
            if dist_front < EMERGENCY_DIST:
                return (0, 0)
            # Speed if we stay in the current lane
            v_stay_lane = self.keep_current_lane(movable_objs, v, acc, target_speed)

            if left_lane_available and \
            dist_front >= self.safe_dist_front and \
            abs(dist_front / (v - v_front + EPSILON)) >= REACT_TIME and \
            (v > v_stay_lane or (v < target_speed and dist_front <= self.dist_lane_change * self.safe_dist_front)):

                left_lane_change = self.is_lane_change_safe(left_front,
                                                            left_rear,
                                                            v,
                                                            acc,
                                                            target_speed,
                                                            v_front)
                if left_lane_change:
                    speed = self.get_speed_lane_change(movable_objs, v, acc, target_speed, lane_id - 1)
                    return (1, speed)
            if right_lane_available and \
            dist_front >= self.safe_dist_front and \
            abs(dist_front / (v - v_front + EPSILON)) >= REACT_TIME and \
            (v > v_stay_lane or (v < target_speed and dist_front <= self.dist_lane_change * self.safe_dist_front)):

                right_lane_change = self.is_lane_change_safe(right_front,
                                                            right_rear,
                                                            v,
                                                            acc,
                                                            target_speed,
                                                            v_front)
                if right_lane_change:
                    speed = self.get_speed_lane_change(movable_objs, v, acc, target_speed, lane_id + 1)
                    return (2, speed)

            return (0, v_stay_lane)
        else:

            if right_lane_available:
                right_lane_change = self.is_lane_change_safe(right_front,
                                                            right_rear,
                                                            target_speed,
                                                            acc,
                                                            target_speed,
                                                            target_speed)
                if right_lane_change:
                    speed = self.get_speed_lane_change(movable_objs, v, acc, target_speed, lane_id + 1)
                    return (2, speed)

            return (0, min(target_speed, v + SPEED_DIFF))

    def get_near_objs(self, movable_obj, lane_id):
        """_summary_

        Args:
            movable_obj (_type_): _description_
            lane_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Each vehicle (x, y, v_x, v_y, orientation)
        left_rear = None
        left_front = None

        cur_front = None
        cur_rear = None

        right_front = None
        right_rear = None

        for i in range(len(movable_obj)):
            obj = movable_obj[i]
            obj_id = obj[0]
            obj_position_x = obj[1]
            obj_position_y = obj[2]
            obj_velocity_x = obj[3]
            obj_velocity_y = obj[4]
            obj_orientation = obj[5]
            obj_lane_id = obj[6]
            obj_box_length = obj[7]
            obj_box_width = obj[8]
            obj_type = obj[9]
            obj_dist_to_center_of_lane = obj[10]

            rear_axle_dist = obj_position_x

            if obj_position_x > 0:
                # Movable obj in front of ego
                obj_position_x -= (obj_box_length / 2) # Distance to the center of the movable obj
                obj_position_x -= 3.4 # From rear axle of ego to the front bumper
            else:
                # Movable obj behind ego
                obj_position_x += (obj_box_length / 2) # Distance to the center of the movable obj
                obj_position_x += 0.7 # From rear axle of ego to the rear bumper
            # For when ego is parallel to other movable objects in different lanes
            if rear_axle_dist * obj_position_x < 0:
                if rear_axle_dist < 0:
                    obj_position_x = -0.001
                else:
                    obj_position_x = 0.001

            if obj_lane_id == lane_id - 1:
                # Left lane
                if obj_position_x > 0:
                    #Front
                    if left_front == None or left_front[0] > obj_position_x:
                        left_front = (obj_position_x,
                                    obj_position_y,
                                    obj_velocity_x,
                                    obj_orientation)
                if obj_position_x < 0:
                    # Rear
                    if left_rear == None or abs(left_rear[0]) > abs(obj_position_x):
                        left_rear = (abs(obj_position_x),
                                    obj_position_y,
                                    obj_velocity_x,
                                    obj_orientation)
            if obj_lane_id == lane_id:
                # Current lane
                if obj_position_x > 0:
                    # Front
                    if cur_front == None or cur_front[0] > obj_position_x:
                        cur_front = (obj_position_x,
                                    obj_position_y,
                                    obj_velocity_x,
                                    obj_orientation)
                if obj_position_x < 0:
                    # Rear
                    if cur_rear == None or abs(cur_rear[0]) > abs(obj_position_x):
                        cur_rear = (abs(obj_position_x),
                                    obj_position_y,
                                    obj_velocity_x,
                                    obj_orientation)
            if obj_lane_id == lane_id + 1:
                # Right lane
                if obj_position_x > 0:
                    # Front
                    if right_front == None or right_front[0] > obj_position_x:
                        right_front = (obj_position_x,
                                    obj_position_y,
                                    obj_velocity_x,
                                    obj_orientation)
                if obj_position_x < 0:
                    # Rear
                    if right_rear == None or abs(right_rear[0]) > abs(obj_position_x):
                        right_rear = (abs(obj_position_x),
                                    obj_position_y,
                                    obj_velocity_x,
                                    obj_orientation)
        return (left_rear,
                left_front,
                cur_rear,
                cur_front,
                right_rear,
                right_front)

    def is_lane_change_safe(self, front, rear, v, acc, target_speed, v_front_ego):
        """_summary_

        Args:
            front (_type_): _description_
            rear (_type_): _description_
            v (_type_): _description_
            acc (_type_): _description_
            target_speed (_type_): _description_
            v_front_ego (_type_): _description_

        Returns:
            _type_: _description_
        """
        if front is None and rear is None:
            return True
        if front is None:
            # Only rear vehicle
            v_rear = rear[2]
            dist_rear = rear[0]

            if (v > v_rear or dist_rear / (v_rear - v + EPSILON) >= REACT_TIME) and dist_rear >= SAFE_DIST_REAR:
                return True
            return False
        elif rear is None:
            # Only front vehicle
            v_front = front[2]
            dist_front = front[0]

            if v_front > v and \
            (dist_front / (v_front - v + EPSILON) >= REACT_TIME) and \
            dist_front >= self.safe_dist_front and \
            v_front > v_front_ego:
                return True
            else:
                return False
        else:
            # Front and rear vehicles in the new lane
            v_rear = rear[2]
            dist_rear = rear[0]
            v_front = front[2]
            dist_front = front[0]
            if dist_front >= self.safe_dist_front and dist_rear >= SAFE_DIST_REAR:
                if (v_front > v and \
                    dist_front / (v - v_front + EPSILON) >= REACT_TIME and \
                        v_front > v_front_ego) and \
                (v >= v_rear or dist_rear / (v_rear - v + EPSILON) >= REACT_TIME):
                    return True
                else:
                    return False
            else:
                return False

    def keep_current_lane(self, movable_objs, v, acc, target_speed):
        """_summary_

        Args:
            key (_type_): _description_

        Returns:
            int: target_speed
        """
        left_rear, left_front, cur_rear, cur_front, right_rear, right_front = movable_objs
        if cur_front is not None:
            return self.match_speed(cur_front, v, acc, target_speed)
        else:
            return min(target_speed, v + SPEED_DIFF)

    def get_speed_lane_change(self, movable_objs, v, acc, target_speed, lane_dest):
        """_summary_

        Args:
            key (_type_): _description_

        Returns:
            int: target_speed
        """
        left_rear, left_front, cur_rear, cur_front, right_rear, right_front = movable_objs
        if lane_dest == 1:
            dest_front = left_front
        else:
            dest_front = right_front
        if cur_front is None and dest_front is None:
            return min(target_speed, v + SPEED_DIFF)
        elif dest_front is not None:
            return self.match_speed(dest_front, v, acc, target_speed)
        elif cur_front is not None:
            return self.match_speed(cur_front, v, acc, target_speed)
        else:
            return min(self.match_speed(cur_front, v, acc, target_speed), self.match_speed(dest_front, v, acc, target_speed))

    def match_speed(self, vehicle, v, acc, target_speed):
        """_summary_

        Args:
            vehicle (_type_): _description_
            v (_type_): _description_
            acc (_type_): _description_
            target_speed (_type_): _description_

        Returns:
            _type_: _description_
        """
        dist_front = vehicle[0]
        v_front = vehicle[2]
        if dist_front < self.safe_dist_front:
            return max(v / 2, v - SPEED_DIFF)
        if (v_front < v - SPEED_DIFF and \
           (dist_front / (v - v_front + EPSILON)) < REACT_TIME):
            return max(v - SPEED_DIFF, min(v_front, min(target_speed, v + SPEED_DIFF)))
        speed_inc = dist_front / self.safe_dist_front - 1
        # For smoother slow downs
        return min(min(target_speed, v + SPEED_DIFF), v_front + speed_inc * SPEED_SECTION)

    def reset_dist_lane_change(self):
        """_summary_
        """
        self.dist_lane_change = np.random.uniform(5, 15)

    def set_dist_lane_change(self, value):
        """_summary_

        Args:
            value (_type_): _description_
        """
        self.dist_lane_change = value

    def set_safe_dist_front(self, value):
        """_summary_

        Args:
            value (_type_): _description_
        """
        self.safe_dist_front = value
