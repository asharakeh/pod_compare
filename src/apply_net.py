"""
Inference Script
"""
import json
import os
import torch
import tqdm
from shutil import copyfile

# Detectron imports
from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, MetadataCatalog

# Project imports
import core.datasets.metadata as metadata

from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_probabilistic_metrics, compute_calibration_errors
from probabilistic_inference.probabilistic_inference import build_predictor
from probabilistic_inference.inference_utils import instances_to_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Setup config
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)

    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Set up number of cpu threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = os.path.join(
        cfg['OUTPUT_DIR'],
        'inference',
        args.test_dataset,
        os.path.split(args.inference_config)[-1][:-5])
    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(args.inference_config, os.path.join(
        inference_output_dir, os.path.split(args.inference_config)[-1]))

    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    if (train_thing_dataset_id_to_contiguous_id == test_thing_dataset_id_to_contiguous_id) or (
            cfg.DATASETS.TRAIN[0] == 'coco_not_in_voc_2017_train'):
        cat_mapping_dict = dict(
            (v, k) for k, v in test_thing_dataset_id_to_contiguous_id.items())
    else:
        # If not equal, two situations: 1) BDD to KITTI and 2) COCO to PASCAL
        cat_mapping_dict = dict(
            (v, k) for k, v in test_thing_dataset_id_to_contiguous_id.items())
        if 'voc' in args.test_dataset and 'coco' in cfg.DATASETS.TRAIN[0]:
            dataset_mapping_dict = dict(
                (v, k) for k, v in metadata.COCO_TO_VOC_CONTIGUOUS_ID.items())
        elif 'kitti' in args.test_dataset and 'bdd' in cfg.DATASETS.TRAIN[0]:
            dataset_mapping_dict = dict(
                (v, k) for k, v in metadata.BDD_TO_KITTI_CONTIGUOUS_ID.items())
        else:
            ValueError(
                'Cannot generate category mapping dictionary. Please check if training and inference datasets are compatible.')
        cat_mapping_dict = dict(
            (dataset_mapping_dict[k], v) for k, v in cat_mapping_dict.items())

    # Build predictor
    predictor = build_predictor(cfg)
    test_data_loader = build_detection_test_loader(
        cfg, dataset_name=args.test_dataset)

    final_output_list = []
    if not args.eval_only:
        with torch.no_grad():
            with tqdm.tqdm(total=len(test_data_loader)) as pbar:
                for idx, input_im in enumerate(test_data_loader):
                    outputs = predictor(input_im)

                    final_output_list.extend(
                        instances_to_json(
                            outputs,
                            input_im[0]['image_id'],
                            cat_mapping_dict))
                    pbar.update(1)

        with open(os.path.join(inference_output_dir, 'coco_instances_results.json'), 'w') as fp:
            json.dump(final_output_list, fp, indent=4,
                      separators=(',', ': '))

    compute_average_precision.main(args, cfg)
    compute_probabilistic_metrics.main(args, cfg)
    compute_calibration_errors.main(args, cfg)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # Support single gpu inference only.
    args.num_gpus = 1
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
