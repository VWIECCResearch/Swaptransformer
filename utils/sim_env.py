"""
#################################
# Python API: Environment creation for SimPilot
#################################
"""

#########################################################
# import libraries
import os
import json
import uuid
# import gym
import numpy as np
import xml.etree.ElementTree as ET
# from gym_unity.envs import UnityToGymWrapper
from typing import List
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage

#########################################################
# General Parameters


#########################################################
# Function and class definition


class StringChannel(SideChannel):
    """_summary_

    Args:
        SideChannel (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__(uuid.UUID("fc3a3bc1-a01f-46bc-91f7-80f67f2bdc5a"))
        self.parameters = dict()

    def on_message_received(self, msg: IncomingMessage) -> None:
        # Message follows the format TYPE:CONTENT
        message = msg.read_string()
        msg_type, content = message.split(":", 1)  # only do a single split (path may contain ":" too)
        if msg_type == "OpenDrivePath":
            self.parameters[msg_type] = content
        elif msg_type == "AvailableScenes":
            self.parameters[msg_type] = content
        elif msg_type == "ReasonForEpisodeEnd":
            self.parameters[msg_type] = content
        elif msg_type == "EpisodeReport":
            report = json.loads(content)
            self.parameters["EpisodeReport"] = report
    
    # Message follows the format TYPE:CONTENT
    # e.g. msg_type = "SetScene:<Scene Name>"
    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)


class AgentSideChannel(SideChannel):
    """_summary_

    Args:
        SideChannel (_type_): _description_
    """
    RANDOMIZATION_DOMAIN_TRAFFIC = 0
    RANDOMIZATION_DOMAIN_OBSTACLES = 1
    RANDOMIZATION_DOMAIN_RoadNet = 2

    RANDOMIZATION_COMPLEXITY = 0
    RANDOMIZATION_DENSITY = 1

    __SET_TRANSFORM = 0
    __SET_DECISION_PERIOD = 1
    __SET_MAX_STEP = 2
    __SET_START_SPEED = 3
    __SET_RANDOMIZATION_OPTION = 4

    def __init__(self) -> None:
        super().__init__(uuid.UUID("ead33186-aa9b-11ec-b909-0242ac120002"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass

    def set_init_transform(self, x: float, y: float, yaw: float) -> None:
        """
        Sets the ego's initial position and yaw for upcoming episodes.
        If the position should be set right away, a following episode reset is needed.
        !This feature only works as intended if exactly one AICar is present in the scene!
        """
        data = [x, y, yaw]
        msg = OutgoingMessage()
        msg.write_int32(self.__SET_TRANSFORM)
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

    def set_decision_period(self, decision_period: int) -> None:
        """
        Sets the decision period parameter for all agents.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__SET_DECISION_PERIOD)
        msg.write_int32(decision_period)
        super().queue_message_to_send(msg)

    def set_max_step(self, max_step: int) -> None:
        """
        Sets the episode length (Max Steps) for all agents.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__SET_MAX_STEP)
        msg.write_int32(max_step)
        super().queue_message_to_send(msg)

    def set_start_speed(self, speed: float) -> None:
        """
        Sets the start speed for supported controllers (SUMO controllers).
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__SET_START_SPEED)
        msg.write_float32(speed)
        super().queue_message_to_send(msg)

    def set_randomization_option(self, domain: int, setting: int, target_level: int):
        """_summary_

        Args:
            domain (int): _description_
            setting (int): _description_
            target_level (int): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__SET_RANDOMIZATION_OPTION)
        msg.write_int32(domain)
        msg.write_int32(setting)
        msg.write_int32(target_level)
        super().queue_message_to_send(msg)


class ControlPoint_release_10_1:
    """_summary_
    """

    def __init__(self, x, y, yaw, vx):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx


class ControlPoint:
    """_summary_
    """

    def __init__(self, x: float, y: float, yaw: float, additional_text: str):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.additional_text = additional_text
        # May also contain \n to enforce a new line in the visualization


class PointVisualizationSideChannel(SideChannel):
    """_summary_

    Args:
        SideChannel (_type_): _description_
    """
    __VISUALIZE_POINTS = 0
    __CLEAR_VISUALIZATION = 1

    def __init__(self) -> None:
        super().__init__(uuid.UUID("e91218ad-b36a-4698-baa3-19ba8495d5e6"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass

    def visualize_points_release_10_1(
            self, points: List[ControlPoint]) -> None:
        """
        Send control points that are visualized in the Unity debug view (e.g. in the cameras
        that can be selected by repeatedly pressing "C").
        If points are already being visualized, they are automatically removed before visualizing new ones.
        !This feature only works as intended if exactly one AICar is present in the scene!
        """
        data = []
        for point in points:
            data.append(point.x)
            data.append(point.y)
            data.append(point.yaw)
            data.append(point.vx)

        msg = OutgoingMessage()
        msg.write_int32(self.__VISUALIZE_POINTS)
        msg.write_int32(len(points))
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

    def visualize_points(self, points: List[ControlPoint]) -> None:
        """
        Send control points that are visualized in the Unity debug view (e.g. in the cameras
        that can be selected by repeatedly pressing "C").
        If points are already being visualized, they are automatically removed before visualizing new ones.
        !This feature only works as intended if exactly one AICar is present in the scene!
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__VISUALIZE_POINTS)
        msg.write_int32(len(points))

        for point in points:
            msg.write_float32_list([point.x, point.y, point.yaw])
            msg.write_string(str(point.additional_text))

        super().queue_message_to_send(msg)

    def clear_visualization(self):
        """
        Clear the visualization in the view. Removes all previously visualized points.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__CLEAR_VISUALIZATION)
        super().queue_message_to_send(msg)


class SumoSideChannel(SideChannel):
    """_summary_

    Args:
        SideChannel (_type_): _description_
    """
    __DOMAIN_VEHICLE = 0xc4
    __Domain_ROUTE = 0xc6

    __CMD_ADD = 0x85
    __CMD_CHANGELANE = 0x13
    __CMD_SLOWDOWN = 0x14
    __CMD_CHANGETARGET = 0x31
    __CMD_SPEED = 0x40
    __CMD_SEED_FACTOR = 0x5e
    __CMD_ACCELERATION = 0x72
    __CMD_ROUTE_ID = 0x53
    __CMD_ROUTE = 0x57
    __CMD_MOVE_TO = 0x5c
    __CMD_MOVE_TO_XY = 0xb4
    __CMD_SPEEDMODE = 0xb3
    __CMD_MAXSPEED = 0x41
    __CMD_LANECHANGEMODE = 0xb6
    __CMD_REMOVE = 0x81
    __CMD_MINGAP = 0x4c
    __CMD_SET_SPEED = 0x40

    __CMD_ADD_ROUTE = 0x80

    def __init__(self) -> None:
        super().__init__(uuid.UUID("dd6594bf-6d98-4796-97bd-5670d8856927"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass

    def add(self,
            vehID,
            routeID,
            typeID='DEFAULT_VEHTYPE',
            depart='now',
            departLane='first',
            departPos='base',
            departSpeed='0',
            arrivalLane='current',
            arrivalPos='max',
            arrivalSpeed='current',
            fromTaz='',
            toTaz='',
            line='',
            personCapacity=0,
            personNumber=0
            ):
        """_summary_

        Args:
            vehID (_type_): _description_
            routeID (_type_): _description_
            typeID (str, optional): _description_. Defaults to 'DEFAULT_VEHTYPE'.
            depart (str, optional): _description_. Defaults to 'now'.
            departLane (str, optional): _description_. Defaults to 'first'.
            departPos (str, optional): _description_. Defaults to 'base'.
            departSpeed (str, optional): _description_. Defaults to '0'.
            arrivalLane (str, optional): _description_. Defaults to 'current'.
            arrivalPos (str, optional): _description_. Defaults to 'max'.
            arrivalSpeed (str, optional): _description_. Defaults to 'current'.
            fromTaz (str, optional): _description_. Defaults to ''.
            toTaz (str, optional): _description_. Defaults to ''.
            line (str, optional): _description_. Defaults to ''.
            personCapacity (int, optional): _description_. Defaults to 0.
            personNumber (int, optional): _description_. Defaults to 0.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_ADD)
        msg.write_string(vehID)
        msg.write_string(routeID)
        msg.write_string(typeID)
        msg.write_string(depart)
        msg.write_string(departLane)
        msg.write_string(departPos)
        msg.write_string(departSpeed)
        msg.write_string(arrivalLane)
        msg.write_string(arrivalPos)
        msg.write_string(arrivalSpeed)
        msg.write_string(fromTaz)
        msg.write_string(toTaz)
        msg.write_string(line)
        msg.write_int32(personCapacity)
        msg.write_int32(personNumber)
        super().queue_message_to_send(msg)

    def changeLane(self, vehID, laneIndex, duration):
        """_summary_

        Args:
            vehID (_type_): _description_
            laneIndex (_type_): _description_
            duration (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_CHANGELANE)
        msg.write_string(vehID)
        msg.write_int32(laneIndex)
        msg.write_float32(duration)
        msg.write_bool(False)
        super().queue_message_to_send(msg)

    def changeLaneRelative(self, vehID, indexOffset, duration):
        """_summary_

        Args:
            vehID (_type_): _description_
            indexOffset (_type_): _description_
            duration (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_CHANGELANE)
        msg.write_string(vehID)
        msg.write_int32(indexOffset)
        msg.write_float32(duration)
        msg.write_bool(True)
        super().queue_message_to_send(msg)

    def slowDown(self, vehID, speed, duration):
        """_summary_

        Args:
            vehID (_type_): _description_
            speed (_type_): _description_
            duration (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_SLOWDOWN)
        msg.write_string(vehID)
        msg.write_float32(speed)
        msg.write_float32(duration)
        super().queue_message_to_send(msg)

    def changeTarget(self, vehID, edgeID):
        """_summary_

        Args:
            vehID (_type_): _description_
            edgeID (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_CHANGETARGET)
        msg.write_string(vehID)
        msg.write_string(edgeID)
        super().queue_message_to_send(msg)

    def setSpeed(self, vehID, speed):
        """_summary_

        Args:
            vehID (_type_): _description_
            speed (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_SET_SPEED)
        msg.write_string(vehID)
        msg.write_float32(speed)
        super().queue_message_to_send(msg)

    def setAcceleration(self, vehID, acceleration, duration):
        """_summary_

        Args:
            vehID (_type_): _description_
            acceleration (_type_): _description_
            duration (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_ACCELERATION)
        msg.write_string(vehID)
        msg.write_float32(acceleration)
        msg.write_float32(duration)
        super().queue_message_to_send(msg)

    def setRouteID(self, vehID, routeID):
        """_summary_

        Args:
            vehID (_type_): _description_
            routeID (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_ROUTE_ID)
        msg.write_string(vehID)
        msg.write_string(routeID)
        super().queue_message_to_send(msg)

    def setRoute(self, vehID, edgeList):
        """_summary_

        Args:
            vehID (_type_): _description_
            edgeList (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_ROUTE)
        msg.write_string(vehID)
        for edge in edgeList:
            msg.write_string(edge)
        super().queue_message_to_send(msg)

    def moveTo(self, vehID, laneID, pos, reason=0):
        """_summary_

        Args:
            vehID (_type_): _description_
            laneID (_type_): _description_
            pos (_type_): _description_
            reason (int, optional): _description_. Defaults to 0.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_MOVE_TO)
        msg.write_string(vehID)
        msg.write_string(laneID)
        msg.write_float32(pos)
        msg.write_int32(reason)
        super().queue_message_to_send(msg)

    def moveToXY(self,
                 vehID,
                 edgeID,
                 lane,
                 x,
                 y,
                 angle=-1073741824.0,
                 keepRoute=1):
        """_summary_

        Args:
            vehID (_type_): _description_
            edgeID (_type_): _description_
            lane (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            angle (float, optional): _description_. Defaults to -1073741824.0.
            keepRoute (int, optional): _description_. Defaults to 1.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_MOVE_TO_XY)
        msg.write_string(vehID)
        msg.write_string(edgeID)
        msg.write_int32(lane)
        msg.write_float32(x)
        msg.write_float32(y)
        msg.write_float32(angle)
        msg.write_int32(keepRoute)
        super().queue_message_to_send(msg)

    def setSpeedMode(self, vehID, sm):
        """_summary_

        Args:
            vehID (_type_): _description_
            sm (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_SPEEDMODE)
        msg.write_string(vehID)
        msg.write_int32(sm)
        super().queue_message_to_send(msg)

    def setMaxSpeed(self, typeID, speed):
        """_summary_

        Args:
            typeID (_type_): _description_
            speed (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_MAXSPEED)
        msg.write_string(typeID)
        msg.write_float32(speed)
        super().queue_message_to_send(msg)

    def setSpeedFactor(self, typeID, factor):
        """_summary_

        Args:
            typeID (_type_): _description_
            factor (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_SEED_FACTOR)
        msg.write_string(typeID)
        msg.write_float32(factor)
        super().queue_message_to_send(msg)

    def setLaneChangeMode(self, vehID, lcm):
        """_summary_

        Args:
            vehID (_type_): _description_
            lcm (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_LANECHANGEMODE)
        msg.write_string(vehID)
        msg.write_int32(lcm)
        super().queue_message_to_send(msg)

    def remove(self, vehID, reason=3):
        """_summary_

        Args:
            vehID (_type_): _description_
            reason (int, optional): _description_. Defaults to 3.
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_REMOVE)
        msg.write_string(vehID)
        msg.write_int32(reason)
        super().queue_message_to_send(msg)

    def setMinGap(self, typeID, minGap):
        """_summary_

        Args:
            typeID (_type_): _description_
            minGap (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__DOMAIN_VEHICLE)
        msg.write_int32(self.__CMD_MINGAP)
        msg.write_string(typeID)
        msg.write_float32(minGap)
        super().queue_message_to_send(msg)

    def addRoute(self, routeID, edges):
        """_summary_

        Args:
            routeID (_type_): _description_
            edges (_type_): _description_
        """
        msg = OutgoingMessage()
        msg.write_int32(self.__Domain_ROUTE)
        msg.write_int32(self.__CMD_ADD_ROUTE)
        msg.write_string(routeID)
        for edge in edges:
            msg.write_string(edge)
        super().queue_message_to_send(msg)


class SimPilotEnv:
    """_summary_
    """

    def __init__(self, args, exec_name="", no_graphic=False, worker_id=0):
        self.exec_name = exec_name
        self.work_dir = os.path.dirname(self.exec_name)
        os.chdir(self.work_dir)
        self.config_jsonfile(work_dir=self.work_dir, args=args)

        # MLAgent's own side channel for adjusting environment parameters
        self.engine_channel = EngineConfigurationChannel()
        self.string_channel = StringChannel()  # Side channel for string communication
        # Side channel to set agent related fields
        self.agent_channel = AgentSideChannel()
        # Side channel to visualize control points near the agent
        self.visualization_channel = PointVisualizationSideChannel()
        self.sumo_channel = SumoSideChannel()
        self.worker_id = worker_id
        self.no_graphic = no_graphic
        if args.editor:
            print('*' * 40)
            print("Please hit the PLAY button in Unity!")
            print('*' * 40)
            self.env_unity = UnityEnvironment(file_name=None,
                                              #   seed=1,
                                              side_channels=[self.engine_channel,
                                                             self.string_channel,
                                                             self.agent_channel,
                                                             self.sumo_channel,
                                                             self.visualization_channel
                                                             ],
                                              additional_args=['-force-vulkan'] if args.vulkan else [''],
                                              no_graphics=self.no_graphic,
                                              worker_id=self.worker_id
                                              )
        else:
            self.env_unity = UnityEnvironment(file_name=exec_name,
                                              #   seed=1,
                                              side_channels=[self.engine_channel,
                                                             self.string_channel,
                                                             self.agent_channel,
                                                             self.sumo_channel,
                                                             self.visualization_channel
                                                             ],
                                              additional_args=['-force-vulkan'] if args.vulkan else [''],
                                              no_graphics=self.no_graphic,
                                              worker_id=self.worker_id
                                              )

        # self.env_gym = UnityToGymWrapper(self.env_unity, )

        self.env_unity.reset()
        behavior_name = list(self.env_unity.behavior_specs)[0]
        spec = self.env_unity.behavior_specs[behavior_name]
        vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
        print("Is there a visual observation ?", vis_obs)
        print("Observed data:")
        for count, obs_spec in enumerate(spec.observation_specs):
            print(
                f"Observation Data_{count} is {obs_spec.name} with Shape: {obs_spec.shape}")

        print("Action space: ", spec.action_spec)
        if spec.action_spec.is_continuous():
            print("The action is continuous")

        if spec.action_spec.is_discrete():
            print("The action is discrete")

    def config_jsonfile(self, work_dir, args):
        """_summary_

        Args:
            work_dir (_type_): _description_
            args (_type_): _description_
        """
        json_file = work_dir + '/projectsettings.json'
        # load the JSON file
        with open(json_file, "r", encoding='utf-8') as json_fn:
            data = json.load(json_fn)


        data['generalConfig']['BEVResolutionHeight'] = args.img_height
        data['generalConfig']['BEVResolutionWidth'] = args.img_width
        data['generalConfig']['BEVSize'] = args.bev_size
        data['generalConfig']['Controller'] = args.controller
        data['generalConfig']['MaxSteps'] = args.maxsteps
        data['generalConfig']['BEVOffsetX'] = args.bevoffsetx
        data['generalConfig']['BEVOffsetY'] = args.bevoffsety

        if args.sumo and args.controller in ("SumoController",
                                             "TravelAssist",
                                             "TravelAssistUnsafe"):
            data['sumoConfig']['EnableSimulation'] = True
            # data['sumoConfig']['NumberOfVehicles'] = 20
        else:
            data['sumoConfig']['EnableSimulation'] = False
        if args.human:
            data['sumoConfig']['EnableSimulation'] = False

        with open(json_file, "w", encoding='utf-8') as json_fn:
            json.dump(data, json_fn, indent=2)
        traffic_level_density = data['sumoConfig']['TrafficComplexityLevel']
        json_file = work_dir + '/advanced_settings.json'

        # Now we edit advanced_settings.json
        with open(json_file, "r", encoding='utf-8') as json_fn:
            data = json.load(json_fn)

        if args.rand_num_vehicles and args.randomization_env:
            data['TrafficPresets']['Density'][traffic_level_density - 1]['VehiclesPerRoadKm'] = \
                int(np.random.uniform(
                    low=args.rand_num_vehicles[0], high=args.rand_num_vehicles[1]))

        with open(json_file, "w", encoding='utf-8') as json_fn:
            json.dump(data, json_fn, indent=2)


    def load_env_unity(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.env_unity

    # def load_env_gym(self):
    #     """_summary_

    #     Returns:
    #         _type_: _description_
    #     """
    #     return self.env_gym

    def load_env_sumo(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.sumo_channel

    def load_env_engine(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.engine_channel

    def load_env_string(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.string_channel

    def load_env_agent(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.agent_channel

    def load_env_visualization(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.visualization_channel

    def load_env_string_channel(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.string_channel

    # def reset_env(self):
    #     """_summary_
    #     """
    #     self.env_gym.reset()

    def test_straight_path(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def test_circular_path(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def pose_to_control_release_10_1(
            self,
            future_pose_x,
            future_pose_y,
            future_pose_v,
            future_pose_yaw):
        """_summary_

        Args:
            future_pose_x (_type_): _description_
            future_pose_y (_type_): _description_
            future_pose_v (_type_): _description_
            future_pose_yaw (_type_): _description_

        Returns:
            _type_: _description_
        """
        control_points = []
        for x, y, v, yaw in zip(
                future_pose_x, future_pose_y, future_pose_v, future_pose_yaw):
            control_points.append(ControlPoint(x, y, yaw, v))
        return control_points

    def pose_to_control(
            self,
            future_pose_x,
            future_pose_y,
            future_pose_v,
            future_pose_yaw):
        """_summary_

        Args:
            future_pose_x (_type_): _description_
            future_pose_y (_type_): _description_
            future_pose_v (_type_): _description_
            future_pose_yaw (_type_): _description_

        Returns:
            _type_: _description_
        """
        control_points = []
        for x, y, v, yaw in zip(
                np.squeeze(future_pose_x),
                np.squeeze(future_pose_y),
                np.squeeze(future_pose_v),
                np.squeeze(future_pose_yaw)):
            control_points.append(ControlPoint(x, y, yaw, str(np.round(v * 3.6, 1))))
        return control_points

    def configure_vtype(self, args):
        """_summary_

        Args:
            args (_type_): _description_
        """
        tree = ET.parse(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/scenedata/vTypes.add_origin.xml')
        root = tree.getroot()

        for vtype in root.findall('vType'):
            id = vtype.get('id')
            vtype.set('color', 'red')
            if id == 'egoType':
                vtype.set('speedFactor', '1')
            else:
                # newspeed = str(round(np.random.normal(1, .2), 2))
                newspeed = str(round(np.random.uniform(0.2,.5), 2))
                vtype.set('speedFactor', newspeed)

            vClass = vtype.get('vClass')
            color = vtype.get('color')
            accel = vtype.get('accel')
            carFollowModel = vtype.get('carFollowModel')
            speedFactor = vtype.get('speedFactor')
            tau = vtype.get('tau')
            emergencyDecel = vtype.get('emergencyDecel')
            lcKeepRight = vtype.get('lcKeepRight')
            lcTurnAlignmentDistance = vtype.get('lcTurnAlignmentDistance')
            lcStrategic = vtype.get('lcStrategic')
            lcSublane = vtype.get('lcSublane')
            lcPushy = vtype.get('lcPushy')
            lcImpatience = vtype.get('lcImpatience')
            lcTimeToImpatience = vtype.get('lcTimeToImpatience')
            lcAssertive = vtype.get('lcAssertive')
            jmStoplineGap = vtype.get('jmStoplineGap')
            jmTimegapMinor = vtype.get('jmTimegapMinor')

            if args.print_flag:
                print(id,
                    vClass,
                    color,
                    accel,
                    carFollowModel,
                    speedFactor,
                    tau,
                    emergencyDecel,
                    lcKeepRight,
                    lcTurnAlignmentDistance,
                    lcStrategic,
                    lcSublane,
                    lcPushy,
                    lcImpatience,
                    lcTimeToImpatience,
                    lcAssertive,
                    jmStoplineGap,
                    jmTimegapMinor)

        tree.write(os.path.join(self.work_dir +
                                '/Assets/Scenes/BuildIn/Highway3LanesOneway/' +
                                'SceneData/TrafficSimulation/Sumo/Simulations/' +
                                'Base/vTypes.add.xml'))

    def hard_reset(self, args):
        """_summary_

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.env_unity.close()
        self.config_jsonfile(work_dir=self.work_dir, args=args)
        if args.editor:
            print('*' * 40)
            print("Please hit the PLAY button in Unity!")
            print('*' * 40)
            self.env_unity = UnityEnvironment(file_name=None,
                                              #   seed=1,
                                              side_channels=[self.engine_channel,
                                                             self.string_channel,
                                                             self.agent_channel,
                                                             self.sumo_channel,
                                                             self.visualization_channel
                                                             ],
                                              additional_args=['-force-vulkan'] if args.vulkan else [''],
                                              no_graphics=self.no_graphic,
                                              worker_id=self.worker_id
                                              )
        else:
            self.env_unity = UnityEnvironment(file_name=self.exec_name,
                                              #   seed=1,
                                              side_channels=[self.engine_channel,
                                                             self.string_channel,
                                                             self.agent_channel,
                                                             self.sumo_channel,
                                                             self.visualization_channel
                                                             ],
                                              additional_args=['-force-vulkan'] if args.vulkan else [''],
                                              no_graphics=self.no_graphic,
                                              worker_id=self.worker_id
                                              )
        env = self.load_env_unity()
        env_sumo = self.load_env_sumo()
        return env, env_sumo
