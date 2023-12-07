import torch
import os
import torchvision
import wandb
import time
import math

import numpy as np
import torch.nn as nn
import lightning as pl
import torch.optim.lr_scheduler as tlr
import matplotlib.pyplot as plt
import pandas as pd

from model.network import get_activation, get_encoder
from utils.collector_utils import args_to_wandbnanme
from config import NUM_MOVE_OBJ, MOVE_OBJ_COLUMNS, MOVE_OBJ_COLUMNS_HYBRID
from utils.sim_env import SimPilotEnv
from utils.collector_utils import convert_image_to_lane_ids
from rule_based import RuleBasedDriver
from config import Config_TRJ
from mlagents_envs.exception import UnityCommunicatorStoppedException

# General Parameters
# Configurable parameters for rule based driver
TIME_PER_STEP = 0.02
EPSILON = 0.0001
LANE_CHANGE_TIME_LMT = 10 # Seconds
LANE_CHANGE_STEP_LMT = LANE_CHANGE_TIME_LMT / TIME_PER_STEP # Steps
FIRST_LANE_CHANGE_STEP_LMT = 300
NUM_FUTURE_TRJ = Config_TRJ.get("NUMBER_POINTS")
NUM_CONTROL_ELEMENTS = Config_TRJ.get("NUM_CONTROL_ELEMENTS")

CONTROLLER_LANE_CHANGE_LMT = 3 # Speed limit for controller lane change command
LANE_SWITCH = 2.0
NUM_MOVE_OBJS = 20
MAX_SPEED_TRAVEL_ASSIST = 44.5
EGO_COLLISION = [256.0, 512.0, 262400.0,263168.0]

ta_map = {0: "None",
          1: "Instantiated",
          2: "Ready to change Lane",
          3: "Started Movement",
          4: "None",
          5: "None",
          6: "None"}

class MultiheadSelfAttention(pl.LightningModule):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value = nn.Linear(input_dim, hidden_dim * num_heads)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim * num_heads, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size, sequence_length, input_dim = x.size()
        # Linear projection for Keys, Query and Values
        Q = self.query(x).reshape(batch_size, self.num_heads, sequence_length, self.hidden_dim)
        K = self.key(x).reshape(batch_size, self.num_heads, sequence_length, self.hidden_dim)
        V = self.value(x).reshape(batch_size, self.num_heads, sequence_length, self.hidden_dim)

        # Computing the scaled dot product of Queries and Keys
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.hidden_dim).float())
        # Getting attention map, where the sum of each row is 1
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Matrix multiplication of attention maps and Values
        attn_output = torch.matmul(attn_weights, V)
        # Reshaping output to match the input of self.output_layer
        attn_output = attn_output.reshape(batch_size, sequence_length,
                                          self.num_heads * self.hidden_dim)
        # Residual connection & Layer norm
        output = self.norm1(x + self.output_layer(attn_output))
        # I saw papers use Gelu as activation function, we can play around with this
        output = nn.functional.gelu(output)

        return output


class PositionalEncoding(pl.LightningModule):
    '''https://pytorch.org/tutorials/beginner/transformer_tutorial.html'''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(1,0)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(1,0)


class FeedForwardLayer(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        output = self.norm(x)
        return output


class TransformerEncoder(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0):
        super().__init__()
        self.attention = MultiheadSelfAttention(input_dim, hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardLayer(input_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        attention_output = self.attention(x)
        residual_output = x + attention_output
        normalized_output = self.norm(residual_output)
        feed_forward_output = self.feed_forward(normalized_output)
        output = normalized_output + feed_forward_output
        return output


class LitModel(pl.LightningModule):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, args, input_c, output_size):
        super().__init__()
        # Activating manual optimizers to have multiple optimizers at the same time
        self.automatic_optimization = False

        self.seq_len = args.num_framestack
        self.move_obj_columns = MOVE_OBJ_COLUMNS
        if args.proc == "INFERENCE" and args.infer_type == 'Hybrid':
            self.move_obj_columns = MOVE_OBJ_COLUMNS_HYBRID
        self.num_move_obj = NUM_MOVE_OBJ
        self.input_dim = args.dim_input_feature + self.num_move_obj * len(self.move_obj_columns)
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks

        self.image_h = args.img_height
        self.image_w = args.img_width
        self.args = args
        self.num_framestack = args.num_framestack
        self.model_filename = 'None'
        self.validation_step_outputs = []

        self.current_poses_mean = torch.zeros(args.batch_size, 2, 100).to(self.device)
        self.predicted_poses_mean = torch.zeros(args.batch_size, 2, 100).to(self.device)
        #  We have to figure out a way to remove LazyLinear
        self.encoder_image =  get_encoder(args.encoder, input_c, args)
        self.adjuster = nn.LazyLinear(args.num_framestack * args.dim_input_feature)

        self.encoder_nparray_fc = nn.Sequential(
            nn.Linear(self.num_framestack * (self.args.dim_input_feature + \
                (self.num_move_obj * len(self.move_obj_columns))), 32),
            nn.ReLU()
        )

        self.lane_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.args.num_poses * 4),
            # nn.Softmax()
        )

        self.encoder_bypass_fc = nn.Sequential(
            nn.Linear(self.num_framestack * self.args.dim_input_feature + 32,
                      self.num_framestack * self.args.dim_input_feature + 32),
            nn.ReLU()
        )

        self.pos_generation = nn.Sequential(
            nn.Linear(self.num_framestack * self.args.dim_input_feature + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            )

        self.velocity_generation = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, args.num_poses)
            )

        if self.args.car_network:
            self.matrix_distance_network = nn.Sequential(
                nn.Linear(self.num_framestack * (self.args.dim_input_feature + \
                    (self.num_move_obj * len(self.move_obj_columns))), 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, (NUM_MOVE_OBJ + 1) * (NUM_MOVE_OBJ + 1))
            )

        self.pos = PositionalEncoding(max_len=self.seq_len, d_model=self.input_dim)

        if self.args.base_model == 'mhsa':
            if self.args.swap:
                self.torso = nn.ModuleList([
                    MultiheadSelfAttention(input_dim=self.seq_len if x % 2  else self.input_dim,
                                        hidden_dim=self.hidden_dim,
                                        num_heads=self.num_heads)
                                        for x in range(self.num_blocks)])
            else:
                self.torso = nn.ModuleList([
                    MultiheadSelfAttention(input_dim=self.input_dim,
                                        hidden_dim=self.hidden_dim,
                                        num_heads=self.num_heads)
                                        for x in range(self.num_blocks)])

        elif self.args.base_model == 'transformer':
                if self.args.swap:
                    self.torso = nn.ModuleList([
                            TransformerEncoder(input_dim=self.seq_len if x % 2  else self.input_dim,
                                            hidden_dim=self.hidden_dim,
                                            num_heads=self.num_heads)
                            for x in range(self.num_blocks)]
                    )
                else:
                    self.torso = nn.ModuleList([
                            TransformerEncoder(input_dim=self.input_dim,
                                            hidden_dim=self.hidden_dim,
                                            num_heads=self.num_heads)
                            for x in range(self.num_blocks)]
                    )
        else:
            self.torso = nn.Sequential(
                nn.Linear(self.input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(), 
                nn.Linear(64, self.input_dim)
            )

    def forward(self, image, nparray):
        """_summary_

        Args:
            image (_type_): _description_
            nparray (_type_): _description_
            groundtruth (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        image = image.reshape(
            image.shape[0],
            self.num_framestack,
            self.image_h,
            self.image_w)

        encoded_image = self.encoder_image(image)
        encoded_image = self.adjuster(encoded_image)
        encoded_image = encoded_image.view(encoded_image.size(0), -1)

        if self.args.base_model in ('transformer', 'mhsa', 'mlp'):
            # Positional encoding for nparray
            if self.args.base_model != 'mlp':
                np_pos_encoded = self.pos(nparray)
                x = nparray
                for i in range(self.num_blocks):
                    # Feeding x to each of the multiheaded blocks
                    x = self.torso[i](x)
                    # Transposeing our matrix across features and then across time
                    if self.args.swap:
                        x = x.transpose(1, 2)

                x = torch.relu(x.reshape(x.shape[0], -1) + np_pos_encoded.reshape(x.shape[0], -1))
            else:
                x = self.torso(nparray)
                x = x.reshape(x.shape[0], -1)
            if self.args.car_network:
                pred_car_matrix = self.matrix_distance_network(x)
            else:
                pred_car_matrix = torch.zeros(1)
            npoutput = self.encoder_nparray_fc(x)
        else:
            npoutput = self.encoder_nparray_fc(nparray.reshape(nparray.shape[0], -1))

        # Reshaping the output of lane change commands to (batch_size, poses, num_classes)
        lane_change_command_logit = self.lane_fc(npoutput).reshape(npoutput.shape[0],
                                                                   self.args.num_poses,
                                                                   4)
        velocity = self.velocity_generation(npoutput)

        encoded = torch.cat([encoded_image, npoutput], dim=1)
        residual = encoded
        encoded = self.encoder_bypass_fc(encoded)

        if self.args.residual:
            encoded = torch.relu(encoded + residual)

        pose = self.pos_generation(encoded)

        return pose, velocity, lane_change_command_logit, pred_car_matrix

    def weighted_loss(self, pred, target):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        weighted = torch.e ** (1 / (100 - torch.arange(100, device=self.device))) - 1
        distance = torch.sum((pred - target) ** 2, dim=1)
        weighted_loss_compute = (distance * weighted).mean()
        return weighted_loss_compute

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        betas = (0.5, 0.999)
        if self.args.multi_opt:
            optimizer_bz = torch.optim.Adam(list(self.torso.parameters()) +
                                            list(self.encoder_image.parameters()) +
                                            list(self.adjuster.parameters()) +
                                            list(self.encoder_nparray_fc.parameters()) +
                                            list(self.encoder_bypass_fc.parameters()) +
                                            list(self.pos_generation.parameters()),
                                            betas=betas,
                                            lr=self.args.lr_bc,
                                            weight_decay=1e-5 if self.args.L2 else 0)

            optimizer_ta_speed = torch.optim.Adam(list(self.torso.parameters()) +
                                                list(self.encoder_nparray_fc.parameters()) +
                                                list(self.velocity_generation.parameters()),
                                                betas=betas,
                                                lr=self.args.lr_bc,
                                                weight_decay=1e-5 if self.args.L2 else 0)

            optimizer_ta_lane = torch.optim.Adam(list(self.torso.parameters()) +
                                                 list(self.encoder_nparray_fc.parameters()) +
                                                 list(self.lane_fc.parameters()),
                                                 betas=betas,
                                                 lr=self.args.lr_bc,
                                                 weight_decay=1e-5 if self.args.L2 else 0)

            if self.args.car_network:
                optimizer_car_network = torch.optim.Adam(list(self.torso.parameters()) +
                                                         list(self.matrix_distance_network.parameters()),
                                                         betas=betas,
                                                         lr=self.args.lr_bc,
                                                         weight_decay=1e-5 if self.args.L2 else 0)
            else:
                optimizer_car_network = torch.optim.Adam(list(self.torso.parameters()),
                                                         betas=betas,
                                                         lr=self.args.lr_bc,
                                                         weight_decay=1e-5 if self.args.L2 else 0)

            scheduler_bz = tlr.ReduceLROnPlateau(optimizer_bz, 'min', patience=self.args.scheduler, factor=self.args.lr_factor)
            scheduler_ta_speed = tlr.ReduceLROnPlateau(optimizer_ta_speed, 'min', patience=self.args.scheduler, factor=self.args.lr_factor)
            scheduler_ta_lane = tlr.ReduceLROnPlateau(optimizer_ta_lane, 'min', patience=self.args.scheduler, factor=self.args.lr_factor)
            scheduler_car_network = tlr.ReduceLROnPlateau(optimizer_car_network, 'min', patience=self.args.scheduler, factor=self.args.lr_factor)

            return [optimizer_bz, optimizer_ta_lane, optimizer_ta_speed, optimizer_car_network], [scheduler_bz, scheduler_ta_lane, scheduler_ta_speed, scheduler_car_network]
        optimizer_total = torch.optim.Adam(self.parameters(),
                                           betas=betas,
                                           lr=self.args.lr_bc,
                                           weight_decay=1e-5 if self.args.L2 else 0)

        scheduler = tlr.ReduceLROnPlateau(optimizer_total, 'min', patience=self.args.scheduler, factor=self.args.lr_factor)
        return [optimizer_total], [scheduler]

    def training_step(self, train_batch, batch_idx):
        """_summary_

        Args:
            train_batch (_type_): _description_
            batch_idx (_type_): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        # loss_ce_obj = nn.CrossEntropyLoss()
        loss_ce_obj = nn.CrossEntropyLoss()

        loss_mse_obj = nn.MSELoss()

        # Getting the optimizers
        if self.args.multi_opt:
            optimizer_bz, optimizer_ta_lane, optimizer_ta_speed, optimizer_car_network = self.optimizers()
            if self.args.scheduler != 0:
                scheduler_bz, scheduler_ta_lane, scheduler_ta_speed, scheduler_car_network = self.lr_schedulers()
        else:
            optimizer_total = self.optimizers()
            if self.args.scheduler != 0:
                scheduler_total = self.lr_schedulers()

        df_stacked, stacked_images, groundtruth_pose, \
                    future_v_global_tensor, groundtruth_pose_ta, car_matrix = train_batch

        predicted_pose, predicted_velocity, lane_change_command_logit, pred_car_matrix = \
            self(image=stacked_images, nparray=df_stacked)

        loss_lane_action = loss_ce_obj(lane_change_command_logit.transpose(2, 1),
                                       groundtruth_pose_ta.long())
        loss_v_action = loss_mse_obj(predicted_velocity,
                                     future_v_global_tensor)

        if self.args.car_network:
            batch_size = car_matrix.shape[0]
            gt_car_matrix = car_matrix.to(self.device)
            pred_car_matrix = pred_car_matrix.reshape(batch_size, NUM_MOVE_OBJ + 1, NUM_MOVE_OBJ + 1)
            mask = (car_matrix == -1).all(dim=2)
            gt_car_matrix = gt_car_matrix[~mask]
            pred_car_matrix = pred_car_matrix[~mask]
            loss_car_network = loss_mse_obj(pred_car_matrix, gt_car_matrix)
        else:
            loss_car_network = torch.zeros(1, device=self.device)

        batch_size = df_stacked.shape[0]
        if self.args.bezier:
            groundtruth_x = groundtruth_pose[:, :self.args.poly_points]
            groundtruth_y = groundtruth_pose[:, self.args.poly_points:2 * self.args.poly_points]
            control_points = predicted_pose

            zero_tensor = torch.zeros(batch_size, 2, device=self.device)
            control_points = torch.cat((zero_tensor, control_points), 1)
            control_points = self.quartic_bezier_curve(control_points, self.args.poly_points)
            combined_groundtruth = torch.stack((groundtruth_x, groundtruth_y), 1)
            predicted_pose = control_points.permute(0, 2, 1)
            loss_pose_mse = self.weighted_loss(predicted_pose,
                                               combined_groundtruth)
        else:
            loss_pose_mse = 0

        total_loss = loss_v_action + loss_pose_mse + loss_lane_action + loss_car_network
        if self.args.multi_opt:
            if self.args.car_network:
                optimizer_car_network.zero_grad()
            optimizer_bz.zero_grad()
            optimizer_ta_lane.zero_grad()
            optimizer_ta_speed.zero_grad()

            if self.args.car_network:
                self.manual_backward(loss_car_network, retain_graph=True)
            self.manual_backward(loss_pose_mse, retain_graph=True)
            self.manual_backward(loss_lane_action, retain_graph=True)
            self.manual_backward(loss_v_action)

            if self.args.car_network:
                optimizer_car_network.step()
            optimizer_bz.step()
            optimizer_ta_lane.step()
            optimizer_ta_speed.step()

            if self.args.scheduler != 0:
                if self.args.car_network:
                    scheduler_car_network.step(loss_car_network)
                scheduler_bz.step(loss_pose_mse)
                scheduler_ta_speed.step(loss_v_action)
                scheduler_ta_lane.step(loss_lane_action)
        else:
            optimizer_total.zero_grad()
            self.manual_backward(total_loss)
            optimizer_total.step()

            if self.args.scheduler != 0:
                scheduler_total.step(total_loss)

        if self.args.print_flag:
            print(f' *** DataLoader Step = {batch_idx} '
                  f' *** BZ (X,Y) MSE = {loss_pose_mse.item():.2f} '
                  f' *** (V) MSE = {loss_v_action.item():.2f} '
                  f' *** car network MSE = {loss_car_network.item():.2f} '
                  f' *** Lane CE = {loss_lane_action.item():.2f}')

        if self.args.track:
            self.log('Loss/Total loss', total_loss.item())
            self.log('Loss/Loss v MSE', loss_v_action.item())
            self.log('Loss/Loss lane CE', loss_lane_action.item())
            if self.args.car_network:
                self.log('Loss/Loss car network MSE', loss_car_network.item())
            if self.args.bezier:
                self.log('Loss/Loss pose MSE', loss_pose_mse.item())

            if batch_idx % 100 == 0 and self.args.bezier:
                self.predicted_poses_mean, self.current_poses_mean = \
                    self.plotpoly_trainer(predicted_pose, combined_groundtruth, self.args.bezier,
                                          self.predicted_poses_mean, self.current_poses_mean)
            if self.args.scheduler != 0:
                if self.args.multi_opt:
                    if self.args.car_network:
                        self.log('Car Network lr', optimizer_car_network.param_groups[0]['lr'])
                    self.log('Speed lr', optimizer_ta_speed.param_groups[0]['lr'])
                    self.log('Pose lr', optimizer_bz.param_groups[0]['lr'])
                    self.log('Lane lr', optimizer_ta_lane.param_groups[0]['lr'])
                else:
                    self.log('Optimizer lr', optimizer_total.param_groups[0]['lr'])

    def validation_step(self, val_batch, val_batch_idx, dataloader_idx=0):
        """_summary_

        Args:
            val_batch (_type_): _description_
            batch_idx (_type_): _description_
            dataloader_idx (int, optional): _description_. Defaults to 0.
        """
        pass

    def on_validation_epoch_end(self):
        """_summary_
        """
        pass

    def set_model_name(self, run_date_time):
        """_summary_

        Args:
            run_date_time (_type_): _description_
        """
        if self.args.single_head:
            single_name = "Single"
        else:
            single_name = "Multi"
        if self.args.bezier:
            bezier_name = "Bezier"
        else:
            bezier_name = "NonBezier"
        if self.args.travelassist_pred:
            travel_name = "TA"
        else:
            travel_name = "NonTA"
        if self.args.residual:
            residual_name = "Residual"
        else:
            residual_name = "NonResidual"
        if self.args.multi_opt:
            opt_name = "Multiopt"
        else:
            opt_name = "Singleopt"
        if self.args.car_network:
            car_net = "_CarNet"
        else:
            car_net = ""
        if self.args.swap:
            swap = "Swap"
        else:
            swap = "NoSwap"
        self.model_filename = travel_name + '_' + single_name +'_' + self.args.base_model + '_' + bezier_name + \
            car_net + '_' + residual_name + '_' + swap + f"_{self.args.algo}_encoder_{self.args.encoder}_" \
            f"act_{self.args.activation}_opt_{opt_name}_{run_date_time}_"

    def save_model(self, run_date_time, epoch):
        """_summary_

        Args:
            run_date_time (_type_): _description_
            epoch (_type_): _description_
        """
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path, exist_ok=False)
        if self.args.single_head:
            single_name = "Single"
        else:
            single_name = "Multi"
        if self.args.bezier:
            bezier_name = "Bezier"
        else:
            bezier_name = "NonBezier"
        if self.args.travelassist_pred:
            travel_name = "TA"
        else:
            travel_name = "NonTA"
        if self.args.residual:
            residual_name = "Residual"
        else:
            residual_name = "NonResidual"
        model_filename = travel_name + '_' + single_name +'_' + bezier_name + '_' + \
            residual_name + f"_{self.args.algo}_encoder_{self.args.encoder}_" \
            f"act_{self.args.activation}_{run_date_time}_epoch_{epoch}.pth"
        torch.save(self.state_dict(),
                    os.path.join(self.args.model_path, model_filename))
        print(" *************** MODEL SAVED *************** ")

    def quartic_bezier_curve(self, control_points, num_points):
        """_summary_

        Args:
            control_points (_type_): _description_
            num_points (_type_): _description_

        Returns:
            _type_: _description_
        """
        # reshape control_points to [batch_size, 5, 2]
        control_points = control_points.view(-1, 5, 2)
        # create a tensor of shape [batch_size, num_points, 1] containing values
        # from 0 to 1
        t = torch.linspace(0, 1, num_points, device=self.device).view(-1, num_points, 1)
        # calculate the coefficients for the quartic Bezier curve
        t_1 = 1 - t
        coeff_1 = t_1**4
        coeff_2 = 4 * t_1**3 * t
        coeff_3 = 6 * t_1**2 * t**2
        coeff_4 = 4 * t_1 * t**3
        coeff_5 = t**4
        tmp = coeff_1 * control_points[:, 0, :].unsqueeze(1)
        # calculate the points on the curve
        points = coeff_1 * control_points[:, 0, :].unsqueeze(1) + \
                 coeff_2 * control_points[:, 1, :].unsqueeze(1) + \
                 coeff_3 * control_points[:, 2, :].unsqueeze(1) + \
                 coeff_4 * control_points[:, 3, :].unsqueeze(1) + \
                 coeff_5 * control_points[:, 4, :].unsqueeze(1)

        return points

    def plotpoly_trainer(self, predicted_pose, combined_groundtruth, bezier,
                     predicted_poses_mean, current_poses_mean):
        """_summary_

        Args:
            predicted_pose (_type_): _description_
            combined_groundtruth (_type_): _description_
            bezier (_type_): _description_
            predicted_poses_mean (_type_): _description_
            current_poses_mean (_type_): _description_

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            if bezier:
                predicted_sample = torch.mean(predicted_pose.to(self.device) + \
                    predicted_poses_mean.to(self.device), 0).cpu().detach().numpy()
                groundtruth_sample = torch.mean(combined_groundtruth.to(self.device) + \
                    current_poses_mean.to(self.device), 0).cpu().detach().numpy()
                predicted_poses_mean = torch.tensor(predicted_sample).to(self.device)
                current_poses_mean = torch.tensor(groundtruth_sample).to(self.device)
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
                self.logger.experiment.log({'Bezier Figure': plt})
            else:
                self.logger.experiment.log({'Pose Figure' : plt})

        return predicted_poses_mean, current_poses_mean
