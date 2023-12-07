'''
#################################
# Python API: ArgParser for Input Arguments
#################################
'''

#########################################################
# import libraries
import argparse
from distutils.util import strtobool

#########################################################
# General Parameters

#########################################################
# Function definition


def parse_args():
    '''_summary_

    Returns:
        _type_: _description_
    '''

    parser = argparse.ArgumentParser(
        description='AITP Imitation Learning')
    # *********************** General ***********************
    parser.add_argument('--proc', default='MERGE',
                        help='procedure to use: | EXPERT | PREPROCESS | MERGE '
                        '| TRAIN | LIGHTNING | INFERENCE')
    parser.add_argument('--visu', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='PLANNER VISUALIZATION: False | True')
    parser.add_argument('--print-rate', type=int, default=30,
                        help='Refresh rate for printing (default: 30)')
    parser.add_argument('--vis-rate', type=int, default=60,
                        help='Refresh rate for visualization (default: 60)')
    parser.add_argument('--initials', type=str, default='PA',
                        help='Initials of the person who is running API')
    parser.add_argument('--print-flag', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if you want to print for debugging')
    parser.add_argument('--milestone', default='M26',
                        help='Milestone corresponding to this data collection')
    parser.add_argument('--task', default='FMD',
                        help='Task for that PI')
    parser.add_argument('--sumo', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if user wants to drive with Sumo')

    # *********************** Simulation ***********************
    parser.add_argument('--exec-path', type=str,
                        default=None,
                        help='Executable path')
    parser.add_argument('--scene', default='OSCOVAL',
                        help='Specific scene: EHRA | MEGA | BERLIN | OSCOVAL (default)'
                             '| Highway3LanesOneway')
    parser.add_argument('--no-graphic', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='NO GRAPHIC: False | True')
    parser.add_argument('--editor', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='editor: False | True')
    parser.add_argument('--num-eps', type=int, default=200,
                        help='Number of episodes for simulation (default: 200)')
    parser.add_argument('--sim-steptime', type=int, default=20,
                        help='time interval between steps (default: 20 ms)')
    parser.add_argument('--img-height', type=int, default=100,
                        help='edits BEVResolutionHeight on projectsetting.json')
    parser.add_argument('--img-width', type=int, default=60,
                        help='edits BEVResolutionWidth on projectsetting.json')
    parser.add_argument('--bev-size', type=int, default=60,
                        help='edits BEVSize on projectsetting.json')
    parser.add_argument('--controller', type=str, default='ExternalEgoCarController',
                        help='edits Controller on projectsetting.json')
    parser.add_argument('--maxsteps', type=int, default=3000,
                        help='edits the maxSteps  projectsetting.json')
    parser.add_argument('--bevoffsetx', type=float, default=10.0,
                        help='edits BEVOffsetX on projectsetting.json')
    parser.add_argument('--bevoffsety', type=float, default=0.0,
                        help='edits BEVOffsetY on projectsetting.json')
    parser.add_argument('--vulkan', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='vulkan for GPU purpose rendering: False | True')

    # *********************** Approach ***********************
    parser.add_argument('--pose-steptime', type=int, default=500,
                        help='time interval between each pose (default: 500 ms)')
    parser.add_argument('--num-poses', type=int, default=5,
                        help='Number of future poses (default: 5)')
    parser.add_argument('--num-framestack', type=int, default=6,
                        help='Number of frames getting stacked (default: 5)')
    parser.add_argument('--stack-time', type=float, default=5000.0,
                        help='Duration of the stack (in mili-second) ' +
                        'that we want to pass to the model as a frame-stacking')
    parser.add_argument('--num-featurespose', type=int, default=3,
                        help='Number of features in pose (default: 3)')
    parser.add_argument('--travelassist-pred', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='For travelassist pred models')
    parser.add_argument('--travelassist-command', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='To send travel assist prediction ')

    # *********************** Data collection ***********************
    parser.add_argument('--human', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if user wants to drive the car manually')
    parser.add_argument('--record-data', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if user wants to save data for training')
    parser.add_argument('--curve-turn', default=False, action='store_true',
                        help='Having turns in a pre-defined path')
    parser.add_argument('--target-speed', type=float, default=20,
                        help="eGo target speed")
    parser.add_argument('--target-radius', default=2000,
                        help="eGo target radius for turns")
    parser.add_argument('--speed-step', type=int, default=5,
                        help="The step size to increase or decrease speed per arrow key")
    parser.add_argument('--steer-step', type=int, default=10,
                        help='The step size to increase or decrease steering'+
                             'wheel angle per arrow key')
    parser.add_argument('--LaneIDSensor', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If user wants save LaneIDSensor')
    parser.add_argument('--RoadIDSensor', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If user wants save RoadIDSensor')
    parser.add_argument('--DrivableSensor', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If user wants save DrivableSensor')
    parser.add_argument('--semi-auto', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If user wants to change lanes as sumo'+
                             'drives the car with arrow keys')
    parser.add_argument('--rule-based', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If user wants to have a rule-based driver')
    parser.add_argument('--randomize-rule-based', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help="If user wants to randomize rule based driver params")
    parser.add_argument('--randomization-env', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Do you want randomization of ego agent when game resets')
    parser.add_argument('--spawnpoints-path', type=str,
                        default='/home/aghazap/Desktop/SimulationExecutables/SimPilotExe_V14_0_2_B193_Highway3LanesOneway/cpc_aitp_imitation/scenedata/newspawnpoints.csv',
                        help='Path to newspawnpoints csv file')
    parser.add_argument('--randomization-laneid', type=int, default=0,
                        help="Which lanes you want ego agent to spawn at [ 3 | 4 | 5 | 0 ]")
    parser.add_argument('--rand-num-vehicles',nargs=2, type=int, default=None,
                        metavar=('0', '20'),
                        help="Takes two values as input min and max and we" +
                        " (This is for advanced_settings.json) randomly")
    parser.add_argument('--new-rand-eps', type=int, default=5,
                        help="This controls the number of random episode runs")

    # *********************** Pre-process ***********************
    parser.add_argument('--rawdata-path', type=str,
                        default=None,
                        help='Raw Data path for Data Collection and pre-processing')
    parser.add_argument('--processeddata-path',
                        default=None,
                        help='Path to save processed data')
    parser.add_argument('--compresseddata-path',
                        default=None,
                        help='Path to save compressed processed data')
    parser.add_argument('--dest', type=str,
                        default=None,
                        help='Path to merge the data to in Apollo')
    parser.add_argument('--large-df-path', type=str,
                        default=None,
                        help='Path to the large DataFrame file in Apollo')
    parser.add_argument('--img-folder', type=str,
                        default=None,
                        help='Path to the large Image folder in Apollo')
    parser.add_argument('--archive-path', type=str,
                        default=None,
                        help='Path to the Archive  folder in Apollo')
    parser.add_argument('--compress', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, it will compress the processed folder as a tar.gz file')
    parser.add_argument('--multiprocess', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, it will pre-process with multiple CPUs')
    parser.add_argument('--compress-name', type=str,
                        default='compressed_test.tar.gz',
                        help='Name of the compressed file')
    parser.add_argument('--apollo', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If user wants to transfer the compressed data to Apollo')
    parser.add_argument('--time-based', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Pre-process polyfit based on time or not')
    parser.add_argument('--apollo-user', type=str,
                        default=None,
                        help="Username for connecting to Apollo")
    parser.add_argument('--apollo-pass', type=str,
                        default=None,
                        help="Password for apollo connection")
    parser.add_argument('--add-lane-changes', type=int,
                        default=0,
                        help='Number of artificial lane changes added to '+
                        'future and history of the actual lane change')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='How many workers should be used for dataloader?')

    # *********************** Sweep Params ***********************
    parser.add_argument('--encoder', type=str,
                        default='custom',
                        help="Encoder's network")
    parser.add_argument('--lr-gen', type=float,
                        default=0.001,
                        help="Generator's learning rate")
    parser.add_argument('--lr-dis', type=float,
                        default=0.001,
                        help="Discriminator's learning rate")
    parser.add_argument('--lr-bc', type=float,
                        default=0.0005,
                        help="BC model's learning rate")
    parser.add_argument('--clip', type=int,
                        default=1,
                        help="Clipping value for gradients")
    parser.add_argument('--activation', type=str,
                        default='ReLU',
                        help="Activation function of the model")
    parser.add_argument('--L2', type=lambda x: bool(strtobool(x)),
                        default=False,
                        help="L2 normalization during training")

    # *********************** Training and Inference ***********************
    parser.add_argument('--algo', type=str, default='BC',
                        help='Algorithm for BC | GAN | GAIL')
    parser.add_argument('--base-model', type=str, default='transformer',
                        help='Transformer | MHSA | MLP')
    parser.add_argument('--training-df-path', type=str,
                        default=None,
                        help='Path for training data (DataFrame) (default: )')
    parser.add_argument('--validation-df-path', type=str,
                        default=None,
                        help='Path for validation data (DataFrame) (default: )')
    parser.add_argument('--training-image-path', type=str,
                        default=None,
                        help='Path for training data (Image) (default: )')
    parser.add_argument('--validation-image-path', type=str,
                        default=None,
                        help='Path for validation data (Image) (default: )')
    parser.add_argument('--dim-input-feature', type=int, default=6,
                        help='Number of the features we feed into to the model (default: 6)')
    parser.add_argument('--model-saverate', type=int, default=10,
                        help='Refresh rate to save models in step (default: 10,000 steps)')
    parser.add_argument('--num-epoch', type=int, default=200,
                        help='Number of epochs for training (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--train-split', type=float, default=0.98,
                        help='Splitting dataset into train dataloader')
    parser.add_argument('--val-starting-point', type=float, default=0.5,
                        help='This gives asks at what percent of the training'
                        'data should you begin the validation process')
    parser.add_argument('--val-starting-epoch', type=int, default=5,
                        help='Set the value for check_val_every_n_epoch for pytorh lightning')
    parser.add_argument('--model-path', default=None,
                        help='Model path to save or load')
    parser.add_argument('--save-model', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, this experiment will save the model')
    parser.add_argument('--replay-data', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, inference replays expert data')
    parser.add_argument('--model-name', type=str, default='model_w',
                        help='Name to save model weights')
    parser.add_argument('--disc-freq', type=int, default=1,
                        help='How frequent you should update the discrimator model')
    parser.add_argument('--infer-type', type=str, default='Online',
                        help='Either Online(Simulation) | Offline for inference | Hybrid')
    parser.add_argument('--bezier', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Add Bezier to model')
    parser.add_argument('--reset-training', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Reset the training with the weights')
    parser.add_argument('--saved-model-path', type=str, default='opt/ml/models/model_w.pth',
                        help='Path to saved model weights for resetting the training')
    parser.add_argument('--poly-points', type=int, default=100,
                        help='Number of points to fit for Bezier (linspace)  (default: 100)')
    parser.add_argument('--residual', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Add residual connection to network or not')
    parser.add_argument('--single-head', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Single head network predict velocity and pose together')
    parser.add_argument('--multi-opt', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Multiple optimizers (one per each head)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='How many workers should be used for dataloader?')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='How many gpus should be used for training?')
    parser.add_argument('--scheduler', type=int, default=0,
                        help='Whether to use scheduler or not')
    parser.add_argument('--lr-factor', type=float, default=0.9,
                        help='Scheduler factor')
    parser.add_argument('--num-blocks', type=int, default=5,
                        help='Number of MHSA | Transformer blocks')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of heads in our MHSA | Transformer block')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Our hidden dimension for each MHSA | Transformer block')
    parser.add_argument('--evaluate', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Evaluate model to record data')
    parser.add_argument('--car-network', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Used to predict Car Network ')
    parser.add_argument('--mask', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Use masks to predict Car Network or not')
    parser.add_argument('--adaptive-cruise-control', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='To set the adaptive cruise control')
    parser.add_argument('--swap', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Enable Swapping feature')
    # *********************** Monitoring and WANDB ***********************
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-name', type=str, default='simpilot_imitation',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default='icc-aitp',
                        help='team name for wandB project')
    parser.add_argument('--sweep', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be sweeped with Weights and Biases')

    args = parser.parse_args()
    # args.sumo = not args.human

    return args
