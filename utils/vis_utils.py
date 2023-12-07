"""
#################################
# Python API: Visualization
#################################
"""

#########################################################
# import libraries

# import warnings
from copy import deepcopy
import torch
import wandb
import numpy as np
import torchvision.transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


from config import Config_TRJ
#########################################################
# General Parameters
NUMBER_POINTS_TRJ = Config_TRJ.get("NUMBER_POINTS")


invTrans = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                           torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                            std=[1., 1., 1.]),
                                           ])


def visualize(control_message, fig_obj, sc_trj,
              frame, min_acc, max_vel, ego_pos):
    """_summary_

    Args:
        control_message (_type_): _description_
        fig_obj (_type_): _description_
        sc_trj (_type_): _description_
        frame (_type_): _description_
        min_acc (_type_): _description_
        max_vel (_type_): _description_
        ego_pos (_type_): _description_

    Returns:
        _type_: _description_
    """

    x = control_message[1, :]
    y = control_message[2, :]
    vel = control_message[3, 0]
    acc = control_message[4, 0]
    yaw = control_message[5, 0]
    curvature_cal = control_message[6, 0]
    curvature_cal_set = control_message[6, :]
    time_stamp = control_message[9, :]

    if fig_obj is None:
        fig_obj = plt.figure(figsize=(10, 8))

        ax_trj = fig_obj.add_subplot(321)
        ax_trj.grid(True)
        ax_trj.set_xlabel("X - Ref", size=12, fontweight='bold', color='black')
        ax_trj.set_ylabel("Y - Ref", size=12, fontweight='bold', color='black')
        sc_trj = ax_trj.scatter(x, y, color='blue', linewidths=0.01)
        ax_trj.plot(x[0], y[0], marker=(3, 0, -90 + np.rad2deg(yaw)),
                    markersize=15, linestyle='None', color='green')
        ax_trj.annotate("", xy=(x[int(NUMBER_POINTS_TRJ / 3) - 1], y[int(NUMBER_POINTS_TRJ / 3) - 1]),
                        xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="->", lw=2))

        ax_trj.plot(
            ego_pos[0],
            ego_pos[1],
            marker="o",
            markersize=15,
            linestyle='None',
            color='red')

        ax_trj.set_xlim([np.min(x) - 0.2, np.max(x) + 0.2])
        ax_trj.set_ylim([np.min(y) - 0.2, np.max(y) + 0.2])

        ax_can = fig_obj.add_subplot(322)
        ax_can.bar(np.arange(1, 3), [vel, acc], align='center', alpha=1, width=0.2,
                   tick_label=["Speed", "Calculated Acceleration"], color="green")
        ax_can.set_ylabel("m/s", size=12, labelpad=10, fontweight='bold')
        ax_can.grid()

        ax_frame = fig_obj.add_subplot(323)
        # ax_frame.imshow(invTrans(frame).permute(1, 2, 0))
        ax_frame.imshow(frame)
        ax_frame.set_yticklabels([])
        ax_frame.set_xticklabels([])

        ax_head_cal = fig_obj.add_subplot(324, projection='polar')
        ax_head_cal.bar(
            yaw,
            height=2,
            width=0.3,
            bottom=0.0,
            alpha=0.7,
            color='r')

        ax_curv = fig_obj.add_subplot(325)
        ax_curv.set_xlabel("Trajectory", size=12, fontweight="bold", )
        ax_curv.set_ylabel("Curvature", size=12, fontweight="bold", )
        ax_curv.plot(
            1,
            curvature_cal,
            color="blue",
            linestyle="-",
            label="Calculated Curvature",
            linewidth=2)
        ax_curv.grid()
        ax_curv.legend()

        ax_curv_set = fig_obj.add_subplot(326)
        ax_curv_set.set_xlabel("Trajectory", size=12, fontweight="bold", )
        ax_curv_set.set_ylabel("Curvature Set", size=12, fontweight="bold", )
        ax_curv_set.plot(np.arange(1, NUMBER_POINTS_TRJ + 1), curvature_cal_set, color="blue", linestyle="-",
                         label="Calculated Curvature", linewidth=2)
        ax_curv_set.grid()
        ax_curv_set.legend()

        plt.show(block=False)

    else:
        sc_trj.set_offsets(np.c_[x[1:], y[1:]])
        fig_obj.axes[0].lines[0].set_xdata(x[0])
        fig_obj.axes[0].lines[0].set_ydata(y[0])
        fig_obj.axes[0].lines[0].set_marker((3, 0, np.rad2deg(yaw) - 90))

        fig_obj.axes[0].lines[1].set_xdata(ego_pos[0])
        fig_obj.axes[0].lines[1].set_ydata(ego_pos[1])

        fig_obj.axes[0].texts[0].set_x(x[0])
        fig_obj.axes[0].texts[0].set_y(y[0])
        fig_obj.axes[0].texts[0].xy = (
            x[int(NUMBER_POINTS_TRJ / 3) - 1], y[int(NUMBER_POINTS_TRJ / 3) - 1])
        fig_obj.axes[0].set_xlim([np.min(x) - 30, np.max(x) + 30])
        fig_obj.axes[0].set_ylim([np.min(y) - 30, np.max(y) + 30])

        bars_can = [rect for rect in fig_obj.axes[1].patches]
        bars_can[0].set_height(vel)
        bars_can[1].set_height(acc)

        max_vel = max([vel, max_vel, acc])
        min_acc = min([min_acc, acc])
        fig_obj.axes[1].set_ylim(min_acc - 1, max_vel + 5)

        fig_obj.axes[2].images[0].set_data(frame)

        pies = [pie for pie in fig_obj.axes[3].patches]
        pies[0].set_x(np.squeeze(yaw))

        new_data_x_curve = fig_obj.axes[4].lines[0].get_xdata()[-1] + 1
        new_x_curve = np.append(
            fig_obj.axes[4].lines[0].get_xdata(),
            new_data_x_curve)
        new_y_curve = np.append(
            fig_obj.axes[4].lines[0].get_ydata(),
            curvature_cal)
        fig_obj.axes[4].lines[0].set_xdata(new_x_curve)
        fig_obj.axes[4].lines[0].set_ydata(new_y_curve)
        fig_obj.axes[4].set_xlim(left=0, right=max(new_x_curve) + 20)
        fig_obj.axes[4].set_ylim(
            bottom=min(new_y_curve) - 0.001,
            top=max(new_y_curve) + 0.001)

        fig_obj.axes[5].lines[0].set_ydata(curvature_cal_set)
        fig_obj.axes[5].set_ylim(
            bottom=min(curvature_cal_set) - 0.001,
            top=max(curvature_cal_set) + 0.001)

    plt.pause(0.0000001)
    return fig_obj, sc_trj, min_acc, max_vel,


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to
    # have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombuffer("RGBA", (w, h), buf.tostring())
    # return Image.fromstring("RGBA", (w, h), buf.tostring())


def visualize_road(fig_obj, frame_drivable, frame_lane, frame_road):
    """_summary_

    Args:
        fig_obj (_type_): _description_
        frame_drivable (_type_): _description_
        frame_lane (_type_): _description_
        frame_road (_type_): _description_

    Returns:
        _type_: _description_
    """
    if fig_obj is None:
        fig_obj = plt.figure(figsize=(5, 8))

        ax_frame_drivable = fig_obj.add_subplot(311)
        # ax_frame_drivable.grid(True)

        ax_frame_drivable.imshow(frame_drivable)
        ax_frame_drivable.set_yticklabels([])
        ax_frame_drivable.set_xticklabels([])

        ax_frame_lane = fig_obj.add_subplot(312)
        ax_frame_lane.set_title("LaneID")
        ax_frame_lane.imshow(frame_lane)
        ax_frame_lane.set_yticklabels([])
        ax_frame_lane.set_xticklabels([])

        ax_frame_road = fig_obj.add_subplot(313)
        ax_frame_road.set_title("RoadID")
        ax_frame_road.imshow(frame_road)
        ax_frame_road.set_yticklabels([])
        ax_frame_road.set_xticklabels([])

    else:

        fig_obj.axes[0].images[0].set_data(frame_drivable)
        fig_obj.axes[1].images[0].set_data(frame_lane)
        fig_obj.axes[2].images[0].set_data(frame_road)

    plt.pause(0.0000001)
    return fig_obj


def visualize_inference(args, fig_obj, sc_trj, frame_lane, data, df_stacked,
                        speed_limit=0, current_speed=0):
    """_summary_

    Args:
        fig_obj (_type_): _description_
        sc_trj (_type_): _description_
        frame_lane (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """

    if isinstance(data, list):
        if len(data) > 2:

            future_pose_x = torch.squeeze(data[2])[0:5].cpu().numpy()
            future_pose_y = -torch.squeeze(data[2])[5:10].cpu().numpy()
            future_pose_v = torch.squeeze(data[2])[10:15].cpu().numpy()

    else:

        future_pose_x = torch.squeeze(data)[0:5].cpu().numpy()
        future_pose_y = -torch.squeeze(data)[5:10].cpu().numpy()
        future_pose_v = torch.squeeze(data)[10:15].cpu().numpy()

    speed_limit_df = df_stacked[0, -1, -1].cpu().numpy()
    frame_lane = map_pose_to_pixel(args,
                                   frame_lane,
                                   [future_pose_x, future_pose_y, future_pose_v])

    if fig_obj is None:
        fig_obj = plt.figure(figsize=(15, 8))

        ax_frame_lane = fig_obj.add_subplot(131)

        ax_frame_lane.imshow(frame_lane, cmap='gist_stern')
        ax_frame_lane.set_yticklabels([])
        ax_frame_lane.set_xticklabels([])

        ax_trj = fig_obj.add_subplot(132)
        ax_trj.grid(True)
        ax_trj.set_xlabel("X - Ref", size=12, fontweight='bold', color='black')
        ax_trj.set_ylabel("Y - Ref", size=12, fontweight='bold', color='black')
        sc_trj = ax_trj.scatter(
            future_pose_y,
            future_pose_x,
            color='blue',
            linewidths=5)
        fig_obj.axes[1].set_xlim([-20, 20])
        fig_obj.axes[1].set_ylim([-30, 100])

        ax_speed_both = fig_obj.add_subplot(133)
        ax_speed_both.set_xlim(1, 6)
        ax_speed_both.set_ylim(0, 8)
        ax_speed_both.text(3, 6, f'Speed Limit = {speed_limit:.2f} km/h', style='italic',
                           bbox={'facecolor': 'red', 'alpha': 0.5, 'pad':10})
        ax_speed_both.text(3, 2, f'Current Speed = {current_speed:.2f} km/h', style='italic',
                           bbox={'facecolor': 'green', 'alpha': 0.5, 'pad':10})


    else:

        fig_obj.axes[0].images[0].set_data(frame_lane)
        sc_trj.set_offsets(np.c_[future_pose_y, future_pose_x])

        fig_obj.axes[1].set_xlim([-20, 20])
        fig_obj.axes[1].set_ylim([-30, 100])

        fig_obj.axes[2].texts[0].set_text(f'Speed Limit = {speed_limit:.2f} km/h')
        fig_obj.axes[2].texts[1].set_text(f'Current Speed = {current_speed:.2f} km/h')

    plt.pause(0.0000001)
    return fig_obj, sc_trj


def map_pose_to_pixel(args, frame, pose):
    """_summary_

    Args:
        frame (_type_): _description_
        pose (_type_): _description_

    Returns:
        _type_: _description_
    """
    BEV_EGO_ID = 7.0
    BEV_OBJS_ID = 20.0
    ONE_PIXEL = float(2 * args.bev_size / frame.shape[0])
    ONE_METER = float(1. / ONE_PIXEL)
    future_pose_x, future_pose_y, _ = pose[0], pose[1], pose[2]

    ego_loc_pixel = (
        int(frame.shape[0] / 2 + 10 * ONE_METER - 1), int(frame.shape[1] / 2))

    pixel_backward_ego = round(0.748 / ONE_PIXEL)
    pixel_forward_ego = round(3.465 / ONE_PIXEL)
    pixel_left_ego = round(1.02 / ONE_PIXEL)
    pixel_right_ego = pixel_left_ego
    frame[ego_loc_pixel[0]-pixel_forward_ego:ego_loc_pixel[0]+pixel_backward_ego,
          ego_loc_pixel[1]-pixel_left_ego:ego_loc_pixel[1]+pixel_right_ego] = BEV_EGO_ID

    h_pose_pix = np.rint(ego_loc_pixel[0] - future_pose_x * ONE_METER).astype(np.int32)
    w_pose_pix = np.rint(ego_loc_pixel[1] + future_pose_y * ONE_METER).astype(np.int32)

    h_pose_pix = np.delete(h_pose_pix, h_pose_pix < 0)
    w_pose_pix = w_pose_pix[0:h_pose_pix.shape[0]]
    w_pose_pix = np.mod(w_pose_pix, frame.shape[1])

    # SET POSE PIXELS on THE FRAME
    frame[h_pose_pix, w_pose_pix] = BEV_OBJS_ID
    return frame


def map_dy_obj_to_pixel(args, frame, dy_objects, future_points=None):
    """_summary_

    Args:
        frame (_type_): _description_
        dy_objects (_type_): _description_

    Returns:
        _type_: _description_
    """
    BEV_EGO_ID = 7.0
    BEV_POSE_ID = 20.0
    BEV_OBJS_ID_HEX = "#F0FFFF"
    ONE_PIXEL = float(2 * args.bev_size / frame.shape[0])
    ONE_METER = float(1. / ONE_PIXEL)

    ego_loc_pixel = (
        int(frame.shape[0] / 2 + 10 * ONE_METER - 1), int(frame.shape[1] / 2))
    pixel_backward_ego = round(0.748 / ONE_PIXEL)
    pixel_forward_ego = round(3.465 / ONE_PIXEL)
    pixel_left_ego = round(1.02 / ONE_PIXEL)
    pixel_right_ego = pixel_left_ego
    frame[ego_loc_pixel[0] -
          pixel_forward_ego:ego_loc_pixel[0] +
          pixel_backward_ego, ego_loc_pixel[1] -
          pixel_left_ego:ego_loc_pixel[1] +
          pixel_right_ego] = BEV_EGO_ID

    # *************************** SET FUTURE POSE
    if future_points is not None:
        if isinstance(future_points, np.ndarray):
            future_pose_x = np.squeeze(future_points)[0:5]
            future_pose_y = -np.squeeze(future_points)[5:10]
        else:
            future_pose_x = torch.squeeze(future_points)[0:5].cpu().numpy()
            future_pose_y = -torch.squeeze(future_points)[5:10].cpu().numpy()
        h_pose_pix = np.rint(ego_loc_pixel[0] - future_pose_x * ONE_METER).astype(np.int32)
        w_pose_pix = np.rint(ego_loc_pixel[1] + future_pose_y * ONE_METER).astype(np.int32)

        h_pose_pix = np.delete(h_pose_pix, h_pose_pix < 0)
        w_pose_pix = w_pose_pix[0:h_pose_pix.shape[0]]
        w_pose_pix = np.mod(w_pose_pix, frame.shape[1])

        # SET POSE PIXELS on THE FRAME
        frame[h_pose_pix, w_pose_pix] = BEV_POSE_ID

    # *************************** SET DYNAMIC OBJECTS
    num_available_dy_obj = np.count_nonzero(dy_objects[:, 0])
    img_bev_obj = Image.fromarray(np.squeeze(frame))
    draw_bev_obj = ImageDraw.Draw(img_bev_obj)
    for _, dy_obj in enumerate(dy_objects[0:num_available_dy_obj]):
        h_pose_pix_dy = round(ego_loc_pixel[0] - dy_obj[1] * ONE_METER)
        w_pose_pix_dy = round(ego_loc_pixel[1] - dy_obj[2] * ONE_METER)

        if h_pose_pix_dy >= frame.shape[0] or h_pose_pix_dy < 0:
            continue

        theta = dy_obj[5]
        width = dy_obj[8] * ONE_METER
        height = dy_obj[7] * ONE_METER

        rect = np.array([(0, 0),
                         (round(width) - 1, 0),
                         (round(width) - 1, round(height) - 1),
                         (0, round(height) - 1),
                         (0, 0)
                         ])
        rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
        offset = np.array([w_pose_pix_dy - round(width / 2),
                           h_pose_pix_dy - round(height / 2)])
        transformed_rect = np.round(np.dot(rect, rotation_mat) + offset)
        draw_bev_obj.polygon([tuple(point) for point in transformed_rect], fill=BEV_OBJS_ID_HEX)
    converted_img = np.asarray(img_bev_obj)
    return converted_img


def visualize_dy_objects(
        args,
        fig_dy_obj,
        frame,
        dy_objects,
        future_points=None):
    """_summary_

    Args:
        fig_dy_obj (_type_): _description_
        frame (_type_): _description_
        dy_objects (_type_): _description_

    Returns:
        _type_: _description_
    """
    frame_modified = deepcopy(frame)
    frame_bev = map_dy_obj_to_pixel(args,
                                    frame_modified,
                                    dy_objects,
                                    future_points)
    if fig_dy_obj is None:
        fig_dy_obj = plt.figure(figsize=(7, 8))
        ax_dy_obj_bev = fig_dy_obj.add_subplot(111)
        ax_dy_obj_bev.imshow(frame_bev, cmap='gist_stern')
        ax_dy_obj_bev.set_yticklabels([])
        ax_dy_obj_bev.set_xticklabels([])

    else:
        fig_dy_obj.axes[0].images[0].set_data(frame_bev)

    plt.pause(0.0000001)
    return fig_dy_obj


def plotpoly_trainer(predicted_pose, combined_groundtruth, bezier,
                     predicted_poses_mean, current_poses_mean):
    """_summary_

    Args:
        predicted_pose (_type_): _description_
        combined_groundtruth (_type_): _description_
    """
    with torch.no_grad():
        if bezier:
            predicted_sample = torch.mean(predicted_pose.cuda() + predicted_poses_mean.cuda(), 0).cpu().detach().numpy()
            groundtruth_sample = torch.mean(combined_groundtruth.cuda() + current_poses_mean.cuda(),0).cpu().detach().numpy()
            predicted_poses_mean = torch.tensor(predicted_sample).cuda()
            current_poses_mean = torch.tensor(groundtruth_sample).cuda()
        else:
            predicted_sample = torch.mean(predicted_pose, 0).cpu().detach().numpy()
            groundtruth_sample = torch.mean(combined_groundtruth, 0).cpu().detach().numpy()

        x1, y1 = predicted_sample
        x2, y2 = groundtruth_sample
        plt.figure(figsize=(8, 6))
        plt.plot(x2, y2, 'b-', label='Ground Truth PolyLine')
        plt.plot(x1, y1, 'r--', label='Predicted Bezier')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ground Truth vs Predicted')
        plt.legend(loc='upper left')
        if bezier:
            wandb.log({'Bezier Figure': plt})
        else:
            wandb.log({'Pose Figure' : plt})

    return predicted_poses_mean, current_poses_mean
