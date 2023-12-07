import torch
import os
import torchvision
import time

import numpy as np
import torch.nn as nn
import lightning as pl

from utils.data_utils import PickleDataset
from utils.collector_utils import HEADERS_TO_LOAD, HEADERS_TO_PREDICT, args_to_wandbnanme
from lightning_model import LitModel
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

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

def train(args):
    """_summary_

    Args:
        args (_type_): _description_
    """

    NON_BEZIER_DIM = args.num_poses * args.num_featurespose

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

    val_dataset = PickleDataset(file_path=args.validation_df_path,
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

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0)

    wandb_project_name = args_to_wandbnanme(args, run_date_time)

    if args.track:
        lightning_logger = WandbLogger(project=args.algo,
                                       name=wandb_project_name,
                                       entity=args.wandb_entity)

    model = LitModel(args=args,
                     input_c=args.num_framestack,
                     output_size=BEZIER_DIM if args.bezier else NON_BEZIER_DIM)
    model.set_model_name(run_date_time)

    batch = next(iter(dataloader))
    df_stacked, stacked_images, groundtruth_pose, \
        future_v_global_tensor, groundtruth_pose_ta, groundtruth_car_matrix = batch
    predicted_pose, predicted_velocity, lane_change_command_logit, prediceted_car_matrix = \
        model(image=stacked_images, nparray=df_stacked)

    checkpoint_callback = ModelCheckpoint(dirpath=args.model_path,
                                          filename=model.model_filename+'{epoch:03d}',
                                          save_top_k=-1, verbose=True,
                                          every_n_epochs=1)

    unused_parameters = 'true' if args.multi_opt or not(args.bezier) else 'false'
    strategy = f"ddp_find_unused_parameters_{unused_parameters}"

    trainer = pl.Trainer(strategy=strategy,
                         accelerator="gpu",
                         devices=args.num_gpus,
                         default_root_dir=args.model_path,
                         logger=lightning_logger if args.track else None,
                         enable_checkpointing=True,
                         callbacks=[checkpoint_callback],
                         log_every_n_steps=1,
                         max_epochs=args.num_epoch,
                         check_val_every_n_epoch=args.val_starting_epoch)

    if args.reset_training:
        trainer.fit(model, dataloader, val_dataloader, ckpt_path=args.saved_model_path)
    else:
        trainer.fit(model, dataloader, val_dataloader)
