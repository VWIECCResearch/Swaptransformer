"""
#################################
# Configuration File
#################################
"""

#########################################################
# import libraries

#########################################################
# Configuration

Config_General = {
    'FLAG_DEBUG_PRINT': False,
    'FLAG_SAVE_MODEL': True,
    "FLOAT_DECIMAL": 3,
    'FLAG_PLOT_REWARD': True,
    'FLAG_SAVE_PLOT': True,
    'FLAG_SAVE_BC_DATA': False,
    'FLAG_SAVE_BC_IMAGE': False}

Config_Expert = {'SPEED_STEP': 5, 'STEER_STEP': 10, 'STEER_RANGE': 40}

Config_DQN = {
    'NUM_EPISODE': 10000000,
    'BUFFER_LENGTH': 100000,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 1e-3,
    'NUM_STEP': 1000,
    'EPSILON_EXPLORATION': 1.0,
    'EPSILON_MIN': 0.001,
    'RAND_SEED': 1449,
    'REQ_STEPS': 100,
    'DISCOUNT_FACTOR_GAMMA': 0.90}

Config_DQN['EPSILON_DECAY'] = 1 / Config_DQN.get("NUM_EPISODE")
Config_ROAD = {'ROAD_THRESH': 0.2, 'LANE_THRESH': 1.8}
# Configurations for visual observation
Config_BC = {'BATCH_SIZE': 28, "NUM_EPOCHS": 100, "SAVE_INTERVAL": 3}

Config_TRJ = {
    "NUMBER_POINTS": 23,
    'FLAG_LOAD_MODEL': True,
    "NUM_CONTROL_ELEMENTS": 10,
    "NUM_EGO_ELEMENTS": 8,
    "TRJ_TIME_INTERVAL": 0.3,
    "MAX_ACC": 2.5,
    "MIN_ACC": -4.5,
    "MAX_KAPPA": 0.2,
    "MIN_KAPPA": -0.2,
    "TRJ_LENGTH_TIME": 6.9}

# Config_GAN = {"image_size": 64,
#               "nc": 3, "nz": 100, "n_gpu": 1,
#               "lr": 0.0002, "beta1": 0.5,
#               "num_epochs": 200,
#               "num_workers": 2, "batch_size": 128
#               }

Config_GAIL = {"lr_gen": 0.001, "lr_dis": 0.001}

Config_WANDB = {"wandb_name": None,
                "wandb_entity": None,
                "algo": None,
                "initials": None,
                "args": None}

MOVE_OBJ_COLUMNS = {
    'pos_x': 1,
    'pos_y': 2,
    'velocity': 3,
    'Continuous Lane Id': 6,
    'Bounding box length': 7}

# if args.proc == "INFERENCE" and args.infer_type == 'Hybrid':
MOVE_OBJ_COLUMNS_HYBRID = {"id": 0, "x": 1, "y": 2, "vx": 3, "vy": 4, "theta": 5,
                           "lane": 6, "length": 7, "width": 8, "type": 9,
                           "relative_t": 10}
NUM_MOVE_OBJ = 20

PATH_FIG = "Figures"
PATH_SAVE_MODEL = "SavedModels"
PATH_SAVE_BC_DATA = "CollectData"
PATH_LOAD_GAN_DATA = "CollectData/GANData"
Config_Path = {"PATH_SAVE_MODEL": PATH_SAVE_MODEL,
               "PATH_FIG": PATH_FIG,
               "PATH_SAVE_BC_DATA": PATH_SAVE_BC_DATA,
               "PATH_LOAD_GAN_DATA": PATH_LOAD_GAN_DATA
               }


def update_config(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    Config_Expert['SPEED_STEP'] = args.speed_step
    Config_Expert['STEER_STEP'] = args.steer_step

    Config_WANDB["args"] = args
    Config_WANDB["wandb_name"] = args.wandb_name
    Config_WANDB["wandb_entity"] = args.wandb_entity
    Config_WANDB["algo"] = args.algo
    Config_WANDB["initials"] = args.initials
