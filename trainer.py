"""
#################################
# Python API: Imitation Learning Trainer
#################################
"""

#########################################################
# import libraries
import os
import sys
import time
import torch
import wandb
import argparse

import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as tlr

from tqdm import tqdm
from collections import deque
from config import Config_WANDB
from config import Config_GAIL, NUM_MOVE_OBJ
from torchvision import transforms
from utils.data_utils import PickleDataset, cubic_bezier_curve, quartic_bezier_curve
from torch.utils.data import Dataset, DataLoader
from model.network import BehavioralCloning, Generator, Discriminator
from utils.collector_utils import HEADERS_TO_LOAD, HEADERS_TO_PREDICT, args_to_wandbnanme
from utils.vis_utils import plotpoly_trainer

#########################################################
# General Parameters
BEZIER_DIM = 4 * 2
# 4 represents number of control points in quartic Bezier
# NUM_FUTURE_TRJ = Config_TRJ.get("NUMBER_POINTS")
# NUM_EGO_ELEMENTS = Config_TRJ.get("NUM_EGO_ELEMENTS")
# TRJ_TIME_INTERVAL = Config_TRJ.get("TRJ_TIME_INTERVAL")
# NUM_CONTROL_ELEMENTS = Config_TRJ.get("NUM_CONTROL_ELEMENTS")
transform = transforms.Compose([transforms.ToTensor()])
current_file_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log2 = np.log(2.)
invlog2 = 1. / log2

#########################################################
# Function definition

def weighted_loss(pred, target):
    """_summary_

    Args:
        pred (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """
    weighted = torch.e** (1 / (100 - torch.arange(100))).to(device) - 1
    distance = torch.sum((pred - target)** 2, dim=1)
    weighted_loss_compute = (distance * weighted).mean()
    return weighted_loss_compute


def validation_starts(val_dataloader, model, args, epoch):
    """_summary_

    Args:
        val_dataloader (_type_): _description_
        model (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    running_mse_pose = 0
    running_mse_v_action = 0
    running_ce_lane_action = 0
    running_mse_car_network = 0
    running_loss_total = 0

    loss_ce_obj = nn.CrossEntropyLoss()
    loss_mse_obj = nn.MSELoss()

    for data in tqdm(val_dataloader):

        df_stacked, stacked_images, groundtruth_pose, \
                    future_v_global_tensor, groundtruth_pose_ta, gt_car_matrix = data
        predicted_pose, predicted_velocity, lane_change_command_logit, predicted_car_matrix = \
            model(image=stacked_images.to(device),
                  nparray=df_stacked.to(device))

        batch_size = df_stacked.shape[0]
        groundtruth_x = groundtruth_pose[:, :args.poly_points]
        groundtruth_y = groundtruth_pose[:, args.poly_points:2 * args.poly_points]
        control_points = predicted_pose

        zero_tensor = torch.zeros(batch_size, 2)
        control_points = torch.cat((zero_tensor.to(device), control_points), 1)
        control_points = quartic_bezier_curve(control_points, args.poly_points)
        combined_groundtruth = torch.stack((groundtruth_x, groundtruth_y), 1)
        predicted_pose = control_points.permute(0, 2, 1)

        loss_lane_action_ce = loss_ce_obj(lane_change_command_logit.transpose(2, 1),
                                          groundtruth_pose_ta.long().to(device))
        loss_v_action_mse = loss_mse_obj(predicted_velocity,
                                        future_v_global_tensor.to(device))
        loss_pose_mse = weighted_loss(predicted_pose.to(device),
                                      combined_groundtruth.to(device))
        if args.car_network:
            batch_size = gt_car_matrix.shape[0]
            gt_car_matrix = gt_car_matrix.to(device)
            mask = (gt_car_matrix == -1).all(dim=2)
            gt_car_matrix = gt_car_matrix[~mask]
            predicted_car_matrix = predicted_car_matrix.reshape(batch_size, NUM_MOVE_OBJ + 1, NUM_MOVE_OBJ + 1)[~mask]
            loss_car_network_mse = loss_mse_obj(predicted_car_matrix, gt_car_matrix)
        else:
            loss_car_network_mse = 0

        total_loss = loss_v_action_mse + loss_pose_mse + loss_lane_action_ce + loss_car_network_mse

        running_mse_pose += loss_pose_mse
        running_mse_v_action += loss_v_action_mse
        running_ce_lane_action += loss_lane_action_ce
        running_mse_car_network += loss_car_network_mse
        running_loss_total += total_loss

    average_mse_pose = running_mse_pose.item() / len(val_dataloader)
    average_mse_v_action = running_mse_v_action.item() / len(val_dataloader)
    average_ce_lane_action = running_ce_lane_action.item() / len(val_dataloader)
    average_mse_car_netowrk = running_mse_car_network.item() / len(val_dataloader)
    average_loss_total = running_loss_total.item() / len(val_dataloader)

    if args.track:
        log_dict = {'Loss/val_Total loss': average_loss_total,
                    'Loss/val_Loss v MSE': average_mse_v_action,
                    'Loss/val_Loss lane CE': average_ce_lane_action,
                    'Loss/val_Loss car network MSE': average_mse_car_netowrk,
                    'Loss/val_Loss pose MSE': average_mse_pose}
        wandb.log(log_dict)

    if args.print_flag:
        print(f' ++++ Validation *** Epoch = {epoch} '
              f' *** Average BZ (X,Y) MSE = {average_mse_pose:.2f} '
              f' *** Average (V) MSE = {average_mse_v_action:.2f} '
              f' *** Average Lane CE = {average_ce_lane_action:.2f} '
              f' *** Average Car Network MSE = {average_mse_car_netowrk:.2f} '
              f' *** Average Total loss = {average_loss_total:.2f}')


def train(args):
    """_summary_

    Args:
        args (_type_): _description_
    """

    run_date_time = time.strftime("%Y_%m_%d-%H_%M")
    # BEZIER_OUT_DIM = 4 * (args.num_featurespose - 1)
    # 4 represents number of control points in quartic Bezier
    # In Cubic, it would be 3 x 2
    # In Non_Bezier_dim the dimension would be 5 x 3 = 15
    NON_BEZIER_DIM = args.num_poses * args.num_featurespose
    dataset = PickleDataset(file_path=args.training_df_path,
                            image_folder=args.training_image_path,
                            column_names=HEADERS_TO_LOAD,
                            transform=transform,
                            predict_columns=HEADERS_TO_PREDICT,
                            num_framestack=args.num_framestack,
                            dim_input_feature=args.dim_input_feature,
                            args=args
                            )

    dataset_val = PickleDataset(file_path=args.validation_df_path,
                                image_folder=args.validation_image_path,
                                column_names=HEADERS_TO_LOAD,
                                transform=transform,
                                predict_columns=HEADERS_TO_PREDICT,
                                num_framestack=args.num_framestack,
                                dim_input_feature=args.dim_input_feature,
                                args=args
                                )

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers)

    val_dataloader = DataLoader(dataset_val,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0)

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

    loss = 0
    mse_loss = 0
    loss_ta = 0
    generator_loss = 0
    disc_loss = 0
    step = 0

    if args.algo == "BC":
        if args.single_head:
            bc_model = BehavioralCloning(
                args=args,
                input_c=args.num_framestack,
                output_size=(BEZIER_DIM + args.num_poses) if args.bezier else NON_BEZIER_DIM).to(device)
        else:
            bc_model = BehavioralCloning(
                args=args,
                input_c=args.num_framestack,
                output_size=BEZIER_DIM if args.bezier else NON_BEZIER_DIM).to(device)

        if args.reset_training:
            bc_model.load_state_dict(torch.load(args.saved_model_path))
        betas = (0.5, 0.999)
        optimizer_bz = torch.optim.Adam(list(bc_model.torso.parameters()) +
                                        list(bc_model.encoder_image.parameters()) +
                                        list(bc_model.adjuster.parameters()) +
                                        list(bc_model.encoder_nparray_fc.parameters()) +
                                        list(bc_model.encoder_bypass_fc.parameters()) +
                                        list(bc_model.pos_generation.parameters()),
                                        betas=betas,
                                        lr=args.lr_bc,
                                        weight_decay=1e-5 if args.L2 else 0)

        optimizer_ta_speed = torch.optim.Adam(list(bc_model.torso.parameters()) +
                                              list(bc_model.encoder_nparray_fc.parameters()) +
                                              list(bc_model.velocity_generation.parameters()),
                                              betas=betas,
                                              lr=args.lr_bc,
                                              weight_decay=1e-5 if args.L2 else 0)

        optimizer_ta_lane = torch.optim.Adam(list(bc_model.torso.parameters()) +
                                             list(bc_model.encoder_nparray_fc.parameters()) +
                                             list(bc_model.lane_fc.parameters()),
                                             betas=betas,
                                             lr=args.lr_bc,
                                             weight_decay=1e-5 if args.L2 else 0)
        if args.car_network:
            optimizer_car_network = torch.optim.Adam(list(bc_model.torso.parameters()) +
                                                     list(bc_model.matrix_distance_network.parameters()),
                                                     betas=betas,
                                                     lr=args.lr_bc,
                                                     weight_decay=1e-5 if args.L2 else 0)
        else:
            optimizer_car_network = None
        if args.scheduler != 0:
            scheduler_bz = tlr.ReduceLROnPlateau(optimizer_bz, 'min', patience=args.scheduler, factor=args.lr_factor)
            scheduler_lane = tlr.ReduceLROnPlateau(optimizer_ta_lane, 'min', patience=args.scheduler, factor=args.lr_factor)
            scheduler_speed = tlr.ReduceLROnPlateau(optimizer_ta_speed, 'min', patience=args.scheduler, factor=args.lr_factor)
            if args.car_network:
                scheduler_car_network = tlr.ReduceLROnPlateau(optimizer_car_network, 'min', patience=args.scheduler, factor=args.lr_factor)
            else:
                scheduler_car_network = None

        optimizer_total = torch.optim.Adam(bc_model.parameters(),
                                           betas=betas,
                                           lr=args.lr_bc,
                                           weight_decay=1e-5 if args.L2 else 0)
        if args.scheduler != 0:
            scheduler_total = tlr.ReduceLROnPlateau(optimizer_total, 'min', patience=args.scheduler)


        loss_ce_obj = nn.CrossEntropyLoss()
        loss_mse_obj = nn.MSELoss()

        current_poses_mean = torch.zeros(args.batch_size, 2, 100).cuda()
        predicted_poses_mean = torch.zeros(args.batch_size, 2, 100).cuda()
        ## training starts here ###
        for epoch in tqdm(range(args.num_epoch)):
            for data in tqdm(dataloader):

                df_stacked, stacked_images, groundtruth_pose, \
                    future_v_global_tensor, groundtruth_pose_ta, car_matrix = data

                predicted_pose, predicted_velocity, lane_change_command_logit, pred_car_matrix = \
                    bc_model(image=stacked_images.to(device),
                             nparray=df_stacked.to(device))

                loss_lane_action = loss_ce_obj(lane_change_command_logit.transpose(2, 1),
                                               groundtruth_pose_ta.long().to(device))
                loss_v_action = loss_mse_obj(predicted_velocity,
                                             future_v_global_tensor.to(device))
                if args.car_network:
                    batch_size = car_matrix.shape[0]
                    car_matrix = car_matrix.to(device)
                    mask = (car_matrix == -1).all(dim=2)
                    masked_out_gt_car_matrix = car_matrix[~mask]
                    masked_out_pred_car_matrix = pred_car_matrix.reshape(batch_size, NUM_MOVE_OBJ + 1, NUM_MOVE_OBJ + 1)[~mask]
                    loss_car_network = loss_mse_obj(masked_out_pred_car_matrix, masked_out_gt_car_matrix)

                else:
                    loss_car_network = 0

                batch_size = df_stacked.shape[0]
                groundtruth_x = groundtruth_pose[:, :args.poly_points]
                groundtruth_y = groundtruth_pose[:, args.poly_points:2 * args.poly_points]
                control_points = predicted_pose

                zero_tensor = torch.zeros(batch_size, 2)
                control_points = torch.cat((zero_tensor.to(device), control_points), 1)
                control_points = quartic_bezier_curve(control_points, args.poly_points)
                combined_groundtruth = torch.stack((groundtruth_x, groundtruth_y), 1)
                predicted_pose = control_points.permute(0, 2, 1)

                loss_pose_mse = weighted_loss(predicted_pose.to(device),
                                              combined_groundtruth.to(device))

                total_loss = loss_v_action + loss_pose_mse + loss_lane_action + loss_car_network
                if args.multi_opt:
                    if args.car_network:
                        optimizer_car_network.zero_grad()
                    optimizer_bz.zero_grad()
                    optimizer_ta_lane.zero_grad()
                    optimizer_ta_speed.zero_grad()

                    if args.car_network:
                        loss_car_network.backward(retain_graph=True)
                    loss_pose_mse.backward(retain_graph=True)
                    loss_lane_action.backward(retain_graph=True)
                    loss_v_action.backward()

                    if args.car_network:
                        optimizer_car_network.step()
                    optimizer_bz.step()
                    optimizer_ta_lane.step()
                    optimizer_ta_speed.step()

                    if args.scheduler != 0:
                        if args.car_network:
                            scheduler_car_network.step(loss_car_network)
                        scheduler_bz.step(loss_pose_mse)
                        scheduler_speed.step(loss_v_action)
                        scheduler_lane.step(loss_lane_action)
                        if args.track:
                            wandb.log({"Speed lr": optimizer_ta_speed.param_groups[0]['lr'],
                                       "Pose lr": optimizer_bz.param_groups[0]['lr'],
                                       "Lane lr": optimizer_ta_lane.param_groups[0]['lr']})
                            if args.car_network:
                                wandb.log({"Car lr": optimizer_car_network.param_groups[0]['lr']})
                else:
                    optimizer_total.zero_grad()
                    total_loss.backward()
                    optimizer_total.step()

                    if args.scheduler != 0:
                        scheduler_total.step(total_loss)
                        if args.track:
                            wandb.log({"Optimizer lr": optimizer_total.param_groups[0]['lr']})

                if args.track:
                    log_dict = {'Loss/Total loss': total_loss,
                                'Loss/Loss v MSE': loss_v_action,
                                'Loss/Loss lane CE': loss_lane_action,
                                'Loss/Loss car network MSE': loss_car_network,
                                'Loss/Loss pose MSE': loss_pose_mse}
                    wandb.log(log_dict)

                    # if args.bezier:
                    if step % 500 == 0:
                        predicted_poses_mean, current_poses_mean = \
                            plotpoly_trainer(predicted_pose, combined_groundtruth, args.bezier,
                                             predicted_poses_mean, current_poses_mean)

                if args.print_flag and False:
                    print(f' *** Epoch = {epoch} *** DataLoader Step = {dataloader_step} '
                          f' *** BZ (X,Y) MSE = {loss_pose_mse.item():.2f} '
                          f' *** (V) MSE = {loss_v_action.item():.2f} '
                          f' *** Car Network MSE = {loss_car_network.item():.2f} '
                          f' *** Lane CE = {loss_lane_action.item():.2f}')

                if args.save_model and False:
                    if dataloader_step % args.model_saverate == (args.model_saverate - 1):
                        bc_model.save_model(run_date_time=run_date_time, epoch=epoch,
                                            step=dataloader_step)

                # print(f' *** Epoch = {epoch} *** DataLoader Step = {dataloader_step} ')
                step += 1
                # End of Epoch
            print(f' *** End of epoch = {epoch} *** ')

            if epoch % args.val_starting_epoch == (args.val_starting_epoch - 1):
                bc_model.eval()
                print('******** validation starts ********')
                with torch.no_grad():
                    validation_starts(val_dataloader=val_dataloader,
                                      model=bc_model,
                                      args=args,
                                      epoch=epoch)
                bc_model.train()
            if args.track:
                log_dict = {"epoch_step": epoch + 1,}
                wandb.log(log_dict)

            if args.save_model and (epoch % args.model_saverate == (args.model_saverate - 1)):
                bc_model.save_model(run_date_time=run_date_time, epoch=epoch)

    elif args.algo == "GAN":
        losses_mse = deque(maxlen=20)
        losses_disc = deque(maxlen=20)
        losses_gen = deque(maxlen=20)

        gen_model = Generator(
            args=args,
            input_c=args.num_framestack,
            output_size=15).to(device).float()

        dis_model = Discriminator(
            args=args, input_c=args.num_framestack).to(device).float()

        # Weight Initialization
        # gen_model.apply(weights_init)
        # dis_model.apply(weights_init)

        lr_gen = args.lr_gen
        lr_dis = args.lr_dis

        betas=(0.5, 0.999)

        optimizer_gen = torch.optim.Adam(gen_model.parameters(),
                                         lr=lr_gen,
                                         betas=betas,
                                         weight_decay=1e-5 if args.L2 else 0)
        optimizer_dis = torch.optim.Adam(dis_model.parameters(),
                                         lr=lr_dis,
                                         betas=betas,
                                         weight_decay=1e-5 if args.L2 else 0)

        loss_bce_obj = nn.BCELoss()
        loss_mse_obj = nn.MSELoss()

        ## training starts here ###
        for epoch in tqdm(range(args.num_epoch)):
            for data_first, data_second in zip(dataloader, dataloader):
                # Loading Data

                exp_df, exp_image, exp_groundtruth = data_first
                exp_df = exp_df.to(device).float()
                exp_image = exp_image.to(device).float()
                exp_groundtruth = exp_groundtruth.to(device).float()

                df, image, groundtruth = data_second
                df = df.to(device).float()
                image = image.to(device).float()
                groundtruth = groundtruth.to(device).float()

                gen_output = gen_model(
                    image=image.to(device), nparray=df.to(device))

                optimizer_dis.zero_grad()
                exp_label = torch.full((df.shape[0], 1), 1, device=device)
                policy_label = torch.full((df.shape[0], 1), 0, device=device)
                prob_exp = dis_model(image=exp_image.to(device),
                                     nparray=exp_df.to(device),
                                     action=exp_groundtruth)

                disc_loss = loss_bce_obj(prob_exp, exp_label.float())

                prob_policy = dis_model(image=image.to(device),
                                        nparray=df.to(device),
                                        action=gen_output.detach())
                disc_loss += loss_bce_obj(prob_policy, policy_label.float())

                disc_loss.backward()
                optimizer_dis.step()
                optimizer_gen.zero_grad()

                generator_loss = -dis_model(image=image.to(device),
                                            nparray=df.to(device),
                                            action=gen_output)
                generator_loss.mean().backward()
                optimizer_gen.step()

                # Metrics
                with torch.no_grad():
                    mse_loss = loss_mse_obj(
                    gen_output.float().to(device).detach(),
                    groundtruth.to(device).detach())
                    mle_loss = (torch.abs(gen_output.float().to(device).detach()-groundtruth.to(device).detach())).mean()

                if args.track:
                    # if mse_loss is not None:
                    wandb.log({"MSE loss": mse_loss.cpu().detach().numpy()})
                    losses_mse.append(mse_loss.cpu().detach().numpy())


                    wandb.log({"MSE Variance": np.var(losses_mse)})
                    wandb.log({"MLE loss": mle_loss.cpu().detach().numpy()})

                    wandb.log({"Generator loss": generator_loss.mean().cpu().detach().numpy()})
                    losses_gen.append(generator_loss.mean().cpu().detach().numpy())

                    wandb.log({"Generator Variance": np.var(losses_gen)})

                    wandb.log({"Discriminator loss": disc_loss.cpu().detach().numpy()})
                    losses_disc.append(disc_loss.cpu().detach().numpy().item())

                    wandb.log({"Discriminator Variance": np.var(losses_disc)})

                    # if step is not None:
                    wandb.log({"steps": step})
                    wandb.log({"epochs": epoch})

                if args.print_flag:
                    print(f" ***** MSE Loss = {mse_loss.cpu().detach().numpy()}",
                          f"***** Gen Loss = {generator_loss.mean().cpu().detach().numpy()}",
                          f"***** Disc Loss = {disc_loss.cpu().detach().numpy().item()} *****")
                step += 1

                if step % int(len(dataloader) * args.val_starting_point) == 0:
                    gen_model.eval()
                    with torch.no_grad():
                        validation_starts(val_dataloader=val_dataloader,
                                          model=bc_model,
                                          args=args,
                                          epoch=epoch)
                    bc_model.train()
                    gen_model.train()

            if epoch % args.model_saverate == (args.model_saverate - 1):
                if args.save_model:
                    if not (os.path.exists(args.model_path)):
                        os.makedirs(args.model_path, exist_ok=False)
                    torch.save(gen_model.state_dict(),
                               os.path.join(args.model_path, f"algo_{args.algo}_{run_date_time}_update_{epoch}.pth"))
    else:
        sys.exit(30 * '*' + '  Exit: Unknown Algorithm ' + 30 * '*')
