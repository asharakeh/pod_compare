import numpy as np
import os
import random
import torch

from shutil import copyfile

# Detectron imports
import detectron2.utils.comm as comm

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.engine import default_argument_parser, default_setup
from detectron2.utils.logger import setup_logger

# Project imports
import core

from core.datasets.setup_datasets import setup_all_datasets
from probabilistic_modeling.probabilistic_retinanet import ProbabilisticRetinaNet


def setup_arg_parser():
    """
    Sets up argument parser for python scripts.

    Returns:
        arg_parser (ArgumentParser): Argument parser updated with probabilistic detectron args.

    """
    arg_parser = default_argument_parser()

    arg_parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="path to dataset directory")

    arg_parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed to be used for all scientific computing libraries")

    # Inference arguments, will not be used during training.
    arg_parser.add_argument(
        "--inference-config",
        type=str,
        default="",
        help="Inference parameter: Path to the inference config, which is different from training config. Check readme for more information.")

    arg_parser.add_argument(
        "--test-dataset",
        type=str,
        default="",
        help="Inference parameter: Dataset used for testing. Can be one of the following: 'coco_2017_custom_val', 'voc_val', 'bdd_val', or 'kitti_val'")

    # Evaluation arguments, will not be used during training.
    arg_parser.add_argument(
        "--iou-min",
        type=float,
        default=0.1,
        help="Evaluation parameter: IOU threshold bellow which a detection is considered a false positive.")

    arg_parser.add_argument(
        "--iou-correct",
        type=float,
        default=0.7,
        help="Evaluation parameter: IOU threshold above which a detection is considered a true positive.")

    arg_parser.add_argument(
        "--min-allowed-score",
        type=float,
        default=0.0,
        help="Evaluation parameter:Minimum classification score for which a detection is considered in the evaluation.")

    return arg_parser


def add_probabilistic_config(cfg):
    """
        Add configuration elements specific to probabilistic detectron.

    Args:
        cfg (CfgNode): detectron2 configuration node.

    """
    _C = cfg

    # Probabilistic Modeling Setup
    _C.MODEL.PROBABILISTIC_MODELING = CN()
    _C.MODEL.PROBABILISTIC_MODELING.MC_DROPOUT = CN()
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS = CN()
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS = CN()

    # Annealing step for losses that require some form of annealing
    _C.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP = 0

    # Monte-Carlo Dropout Settings
    _C.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE = 0.0

    # Loss configs
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME = 'none'
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES = 3

    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME = 'none'
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE = 'diagonal'
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES = 1000

    # Probabilistic Inference Setup
    _C.PROBABILISTIC_INFERENCE = CN()
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT = CN()
    _C.PROBABILISTIC_INFERENCE.BAYES_OD = CN()
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES_DROPOUT = CN()
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES = CN()

    # General Inference Configs
    _C.PROBABILISTIC_INFERENCE.INFERENCE_MODE = 'standard_nms'
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE = False
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS = 1
    _C.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD = 0.7

    # Bayes OD Configs
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.BOX_MERGE_MODE = 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.CLS_MERGE_MODE = 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.DIRCH_PRIOR = 'uniform'

    # MC Dropout Ensembles Configs
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES_DROPOUT.BOX_MERGE_MODE = 'pre_nms'

    # Ensembles Configs
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE = 'pre_nms'
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.RANDOM_SEED_NUMS = [
        0, 1000, 2000, 3000, 4000]


def setup_config(args, random_seed=None, is_testing=False):
    """
    Sets up config node with probabilistic detectron elements. Also sets up a fixed random seed for all scientific
    computing libraries, and sets up all supported datasets as instances of coco.

    Args:
        args (Namespace): args from argument parser
        random_seed (int): set a fixed random seed throughout torch, numpy, and python
        is_testing (bool): set to true if inference. If true function will return an error if checkpoint directory not
        already existing.
    Returns:
        (CfgNode) detectron2 config object
    """
    # Get default detectron config file
    cfg = get_cfg()
    add_probabilistic_config(cfg)

    # Update default config file with custom config file
    configs_dir = core.configs_dir()
    args.config_file = os.path.join(configs_dir, args.config_file)
    cfg.merge_from_file(args.config_file)

    # Add dropout rate for faster RCNN box head
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_RATE = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE

    # Update config with inference configurations. Only applicable for when in
    # probabilistic inference mode.
    if args.inference_config != "":
        args.inference_config = os.path.join(
            configs_dir, args.inference_config)
        cfg.merge_from_file(args.inference_config)

    # Create output directory
    model_name = os.path.split(os.path.split(args.config_file)[0])[-1]
    dataset_name = os.path.split(os.path.split(
        os.path.split(args.config_file)[0])[0])[-1]

    cfg['OUTPUT_DIR'] = os.path.join(core.data_dir(),
                                     dataset_name,
                                     model_name,
                                     os.path.split(args.config_file)[-1][:-5],
                                     'random_seed_' + str(random_seed))
    if is_testing:
        if not os.path.isdir(cfg['OUTPUT_DIR']):
            raise NotADirectoryError(
                "Checkpoint directory {} does not exist.".format(
                    cfg['OUTPUT_DIR']))

    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)

    # copy config file to output directory
    copyfile(args.config_file, os.path.join(
        cfg['OUTPUT_DIR'], os.path.split(args.config_file)[-1]))

    # Freeze config file
    cfg['SEED'] = random_seed
    cfg.freeze()

    # Initiate default setup
    default_setup(cfg, args)

    # Setup logger for probabilistic detectron module
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="Probabilistic Detectron")

    # Set a fixed random seed for all numerical libraries
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Setup datasets
    dataset_dir = os.path.expanduser(args.dataset_dir)
    setup_all_datasets(dataset_dir)
    return cfg
