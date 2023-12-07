"""
#################################
# Python API: ML.NN section for the model implementation
#################################
"""

#########################################################
# import libraries
import os
import torch
import math
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from config import NUM_MOVE_OBJ, MOVE_OBJ_COLUMNS, MOVE_OBJ_COLUMNS_HYBRID

#########################################################
# General Parameters


#########################################################
# Function and class definition
def get_activation(activation):
    """_summary_

    Args:
        activation (_type_): _description_

    Returns:
        _type_: _description_
    """
    if activation == "ReLU":
        return nn.ReLU()
    if activation == "LeakyReLU":
        return nn.LeakyReLU()


def get_encoder(arch, input_c, args):
    """_summary_

    Args:
        arch (_type_): _description_

    Returns:
        _type_: _description_
    """
    if arch == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=False, progress=True)
        encoder.conv1 = nn.Conv2d(args.num_framestack, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if arch == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=False, progress=True)
        encoder.conv1 = nn.Conv2d(args.num_framestack, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if arch == "efficientb0":
        encoder = EfficientNet.from_pretrained('efficientnet-b0')
        encoder._conv_stem = nn.Conv2d(args.num_framestack, 32, kernel_size=3, stride=2, padding=1, bias=False)

    if arch == "efficientb2":
        encoder = EfficientNet.from_pretrained('efficientnet-b2')
        encoder._conv_stem = nn.Conv2d(args.num_framestack, 32, kernel_size=3, stride=2, padding=1, bias=False)

    if arch == "efficientb6":
        encoder = EfficientNet.from_pretrained('efficientnet-b6')
        encoder._conv_stem = nn.Conv2d(args.num_framestack, 56, kernel_size=3, stride=2, padding=1, bias=False)

    if arch == "custom":
        encoder = nn.Sequential(
            nn.Conv2d(input_c, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
    return encoder


class MultiheadSelfAttention(nn.Module):
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

        # Now we compute the scaled dot product of Querys and Keys
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.hidden_dim).float())
        # Now we get attention map, where the sum of each row equals to 1
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Matrix multiplication of attention maps and Values
        attn_output = torch.matmul(attn_weights, V)
        # Reshaping output to match the input of self.output_layer
        attn_output = attn_output.reshape(batch_size, sequence_length,
                                          self.num_heads * self.hidden_dim)
        # Residual connection & Layer norm
        output = self.norm1(x + self.output_layer(attn_output))
        # I saw papers use Gelu as activation function, we can play around with this
        output = F.gelu(output)

        return output


class PositionalEncoding(nn.Module):
    '''https://pytorch.org/tutorials/beginner/transformer_tutorial.html'''
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
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

class FeedForwardLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
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


class TransformerEncoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
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


class BehavioralCloning(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, args, input_c, output_size):
        super().__init__()
        self.args = args
        self.seq_len = args.num_framestack
        self.move_obj_columns = MOVE_OBJ_COLUMNS
        if args.proc == "INFERENCE" and args.infer_type == 'Hybrid':
            self.move_obj_columns = MOVE_OBJ_COLUMNS_HYBRID
        self.num_move_obj = NUM_MOVE_OBJ
        self.input_dim = args.dim_input_feature + self.num_move_obj * len(self.move_obj_columns)
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks
        # Create a number of multiheaded self-attention blocks
        if self.args.base_model == 'mhsa':
            self.torso = nn.ModuleList([
                MultiheadSelfAttention(input_dim=self.seq_len if x % 2  else self.input_dim,
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

        self.pos = PositionalEncoding(max_len=self.seq_len, d_model=self.input_dim)
        self.image_h = args.img_height
        self.image_w = args.img_width
        self.num_framestack = args.num_framestack
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

        # # Some models were trained with encoder_nparray_fc_ta (it's not used in forward pass)
        # self.encoder_nparray_fc_ta = nn.Sequential(
        #     nn.Linear(self.args.num_framestack * (self.args.dim_input_feature + 100), 32),
        #     nn.LeakyReLU()
        # )
        self.pos_generation = nn.Sequential(
            nn.Linear(self.num_framestack * self.args.dim_input_feature + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            )
        # self.pos_generation = nn.Sequential(
        #         nn.Linear(self.num_framestack * self.args.dim_input_feature + 32, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, output_size)
        #     )
        self.velocity_generation = nn.Sequential(
            # nn.Linear(self.num_framestack * (self.args.dim_input_feature + 100), 64),
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

    def save_model(self, run_date_time, epoch, step='None'):
        """_summary_

        Args:
            run_date_time (_type_): _description_
            epoch (_type_): _description_
            step (str, optional): _description_. Defaults to 'None'.
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
        if self.args.multi_opt:
            opt_name = "Multiopt"
        else:
            opt_name = "Singleopt"
        if self.args.car_network:
            car_net = "_CarNet"
        else:
            car_net = ""

        model_filename = travel_name + '_' + single_name +'_' + self.args.base_model + '_' + bezier_name + \
            car_net + '_' + residual_name + f"_{self.args.algo}_encoder_{self.args.encoder}_" \
            f"act_{self.args.activation}_opt_{opt_name}_{run_date_time}_epoch_{epoch}_step_{step}.pth"
        torch.save(self.state_dict(),
                    os.path.join(self.args.model_path, model_filename))
        print(" *************** MODEL SAVED *************** ")


class Generator(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, args, input_c, output_size):
        super().__init__()
        self.image_h = args.img_height
        self.image_w = args.img_width
        self.args = args
        self.num_framestack = self.args.num_framestack
        self.input_c = input_c

        self.encoder = get_encoder(args.encoder, self.input_c, self.args)
        network_activation = get_activation(args.activation)
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.num_framestack * self.args.dim_input_feature, 32),
            network_activation
        )

        self.encoder_fc2 = nn.Sequential(
            nn.LazyLinear(out_features=32),
            network_activation
        )

        self.actor = nn.Sequential(
            nn.Linear(32, 64),
            network_activation,
            nn.Linear(64, output_size)
        )

    def forward(self, image, nparray):
        """_summary_

        Args:
            image (_type_): _description_
            nparray (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = image.reshape(
            image.shape[0],
            self.num_framestack,
            self.image_h,
            self.image_w)
        # encoded = self.encoder(image)
        encoded = self.encoder(image)
        encoded = encoded.view(encoded.size(0), -1)
        npoutput = self.encoder_fc(nparray.reshape(nparray.shape[0], -1))
        encoded = torch.cat([encoded, npoutput], 1)
        encoded = self.encoder_fc2(encoded)
        action = self.actor(encoded)
        return action


class Discriminator(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, args, input_c):
        super().__init__()
        self.args = args
        self.image_h = args.img_height
        self.image_w = args.img_width
        self.num_framestack = self.args.num_framestack
        self.input_c = input_c

        self.encoder = get_encoder(args.encoder, self.input_c, self.args)
        network_activation = get_activation(args.activation)

        self.encoder_fc1 = nn.Sequential(
            nn.Linear(self.args.num_poses * self.args.num_featurespose, 32),
            network_activation
        )

        self.encoder_fc2 = nn.Sequential(
            nn.Linear(self.args.num_framestack * self.args.dim_input_feature, 32),
            network_activation
        )

        self.encoder_fc3 = nn.Sequential(
            nn.LazyLinear(out_features=64),
            network_activation,
            nn.Linear(64, 32),
            network_activation,
        )

        self.discriminator = nn.Sequential(
            nn.Linear(32, 64),
            network_activation,
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, nparray, action):
        """_summary_

        Args:
            image (_type_): _description_
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = image.reshape(
            image.shape[0],
            self.args.num_framestack,
            self.image_h,
            self.image_w)

        # encoded = self.encoder(image)
        encoded_image = self.encoder(image)
        encoded_image = encoded_image.view(encoded_image.size(0), -1)

        # Flattening output
        action_encoded = self.encoder_fc1(action.reshape(action.shape[0], -1))
        df_encoded = self.encoder_fc2(nparray.reshape(nparray.shape[0], -1))

        encoded = torch.cat([encoded_image, df_encoded, action_encoded], 1)
        encoded = self.encoder_fc3(encoded)
        disc_prob = self.discriminator(encoded)

        return disc_prob

    def turn_off(self):
        """_summary_
        """
        for i, param in enumerate(self.parameters()):
            if i == 0:
                param.requires_grad = False

    def turn_on(self):
        """_summary_
        """
        for i, param in enumerate(self.parameters()):
            if i == 0:
                param.requires_grad = True


