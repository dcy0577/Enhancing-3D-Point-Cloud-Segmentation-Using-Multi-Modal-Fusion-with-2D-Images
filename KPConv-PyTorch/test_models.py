#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
# from datasets.ScanNet_beta import *
from torch.utils.data import DataLoader
#from datasets.ScanNet_new import *
from datasets.ScanNet_sphere_color import *
# from datasets.ScanNet_sphere_color import *
# from datasets.ScanNet_baseline_color import *

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import  KPFCNN
# from models.architectures_new import KPFCNN_featureAggre
from models.architectures_sphere import KPFCNN_featureAggre
# from models.architectures_sphere_late_fusion import KPFCNN_featureAggre
# from models.architectures_sphere_middle_fusion import KPFCNN_featureAggre




# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ScanNet']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ScanNet']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = 'results/Log_2021-08-24_15-21-55'

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    # idx sort by number ascending
    # chkp_idx = None
    chkp_idx = 0

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 0.0001
    # config.augment_symmetries = [False, False, False]
    # config.augment_rotation = None
    #### choose which fusion
    config.early_fusion = True
    # config.late_fusion = True
    # config.middle_fusion = True
    config.validation_size = 200
    # config.validation_size = 50
    config.input_threads = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ScanNet':
        # test_dataset = ScanNetDataset_baseline(config, set='validation', split= 'val',use_potentials=True)
        test_dataset = ScanNetDataset(config,
                                      set='validation',
                                      split='val',
                                      use_potentials=True,
                                      load_data=True,
                                      num_rgbd_frames=6,
                                      resize=(160, 120),
                                      image_normalizer=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      # image_normalizer=None,
                                      k=3,
                                      flip=0.0,
                                      color_jitter=None,
                                      use_point_color=False,
                                      )
        test_sampler = ScanNetSampler(test_dataset)
        collate_fn = ScanNetCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    # elif config.dataset_task in ['cloud_segmentation']:
    #     net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    if config.dataset_task in ['cloud_segmentation']:
        net = KPFCNN_featureAggre(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)

    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)