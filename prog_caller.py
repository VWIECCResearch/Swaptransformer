"""
Created on Jan. 26, 2023
@author:    Ali Shams
@Project:   Python API AI Agent
            Paper: ### TODO
            Arxiv: ### TODO
            ID: ### TODO
@Volkswagen Group of America VWGoA (IECC)
Info:
Anaconda:
Python:
MLAgent:
MLAgents:
MLAgents-envs:
Torch:
"""

##################################################################
# Main File
##################################################################

# ############# import libraries
# General Modules
import sys
import lightning_trainer
# Customized Modules
from inference import inference
from expert import expert_drive
from config import update_config
from trainer import train
from utils.argparser import parse_args
from utils.data_utils import pre_process_data
from utils.data_utils import global_merge, local_merge
from getpass import getpass

#########################################################
# Main Definition


def main():
    """_summary_

    Returns:
        _type_: _description_
    """
    print('--' * 40)
    print('..' * 10 + 'Python API for 0_0 Controller' + ".." * 10)
    print('--' * 40)
    return parse_args()


if __name__ == "__main__":
    args_cl = main()
    update_config(args_cl)

    if args_cl.apollo:
        args_cl.apollo_pass = getpass(
            "Please enter your password for Apollo\n")

    if args_cl.proc == "EXPERT":
        print(10 * '*' + '  INIT: Start Expert ' + 10 * '*')
        expert_drive(args_cl)
        print(10 * '*' + '  Done: Done with Expert ' + 10 * '*')
    elif args_cl.proc == "PREPROCESS":
        print(10 * '*' + '  INIT: Start Pre-Processings ' + 10 * '*')
        pre_process_data(args_cl)
        print(10 * '*' + '  Done: Done with Pre-Processing ' + 10 * '*')
    elif args_cl.proc == "TRAIN":
        print(10 * '*' + '  INIT: Start Training ' + 10 * '*')
        train(args_cl)
        print(10 * '*' + '  Done: Done with Training ' + 10 * '*')
    elif args_cl.proc == "LIGHTNING":
        print(10 * '*' + '  INIT: Start Training ' + 10 * '*')
        lightning_trainer.train(args_cl)
        print(10 * '*' + '  Done: Done with Training ' + 10 * '*')
    elif args_cl.proc == "INFERENCE":
        print(10 * '*' + '  INIT: Start Inference ' + 10 * '*')
        inference(args_cl)
        print(10 * '*' + '  Done: Done with Inference ' + 10 * '*')
    elif args_cl.proc == "MERGE":
        print(10 * '*' + '  INIT: Start Merge ' + 10 * '*')
        global_merge(args_cl)
        print(10 * '*' + '  Done: Done with Merge ' + 10 * '*')
    elif args_cl.proc == "LOCAL_MERGE":
        print(10 * '*' + '  INIT: Start Merge ' + 10 * '*')
        local_merge(args_cl)
        print(10 * '*' + '  Done: Done with Local Merge ' + 10 * '*')    
    else:
        sys.exit(30 * '*' + '  Exit: Unknown procedure ' + 30 * '*')
