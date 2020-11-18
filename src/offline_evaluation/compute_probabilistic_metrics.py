import numpy as np
import os
import torch
import pickle

from prettytable import PrettyTable

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import evaluation_utils
from core.evaluation_tools import scoring_rules
from core.evaluation_tools.evaluation_utils import get_thing_dataset_id_to_contiguous_id_dict
from core.setup import setup_config, setup_arg_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
        args,
        cfg=None,
        iou_min=None,
        iou_correct=None,
        min_allowed_score=None):
    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Setup torch device and num_threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Build path to gt instances and inference output
    inference_output_dir = os.path.join(
        cfg['OUTPUT_DIR'],
        'inference',
        args.test_dataset,
        os.path.split(args.inference_config)[-1][:-5])

    # Get thresholds to perform evaluation on
    if iou_min is None:
        iou_min = args.iou_min
    if iou_correct is None:
        iou_correct = args.iou_correct

    if min_allowed_score is None:
        # Check if F-1 Score has been previously computed ON THE ORIGINAL
        # DATASET such as COCO even when evaluating on VOC.
        try:
            train_set_inference_output_dir = os.path.join(
                cfg['OUTPUT_DIR'],
                'inference',
                cfg.DATASETS.TEST[0],
                os.path.split(args.inference_config)[-1][:-5])
            with open(os.path.join(train_set_inference_output_dir, "mAP_res.txt"), "r") as f:
                min_allowed_score = f.read().strip('][\n').split(', ')[-1]
                min_allowed_score = round(float(min_allowed_score), 4)
        except FileNotFoundError:
            # If not, process all detections. Not recommended as the results might be influenced by very low scoring
            # detections that would normally be removed in robotics/vision
            # applications.
            min_allowed_score = 0.0

    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    cat_mapping_dict = get_thing_dataset_id_to_contiguous_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)

    # Get matched results by either generating them or loading from file.
    with torch.no_grad():
        matched_results = evaluation_utils.get_matched_results(
            cfg, inference_output_dir,
            iou_min=iou_min,
            iou_correct=iou_correct,
            min_allowed_score=min_allowed_score)

        # Build preliminary dicts required for computing classification scores.
        for matched_results_key in matched_results.keys():
            if 'gt_cat_idxs' in matched_results[matched_results_key].keys():
                # First we convert the written things indices to contiguous
                # indices.
                gt_converted_cat_idxs = matched_results[matched_results_key]['gt_cat_idxs'].squeeze(
                    1)
                gt_converted_cat_idxs = torch.as_tensor([cat_mapping_dict[class_idx.cpu(
                ).tolist()] for class_idx in gt_converted_cat_idxs]).to(device)
                matched_results[matched_results_key]['gt_converted_cat_idxs'] = gt_converted_cat_idxs.to(
                    device)
                if 'predicted_cls_probs' in matched_results[matched_results_key].keys(
                ):
                    predicted_cls_probs = matched_results[matched_results_key]['predicted_cls_probs']
                    # This is required for evaluation of retinanet based
                    # detections.
                    matched_results[matched_results_key]['predicted_score_of_gt_category'] = torch.gather(
                        predicted_cls_probs, 1, gt_converted_cat_idxs.unsqueeze(1)).squeeze(1)
                matched_results[matched_results_key]['gt_cat_idxs'] = gt_converted_cat_idxs
            else:
                # For false positives, the correct category is background. For retinanet, since no explicit
                # background category is available, this value is computed as 1.0 - score of the predicted
                # category.
                predicted_class_probs, predicted_class_idx = matched_results[matched_results_key]['predicted_cls_probs'].max(
                    1)
                matched_results[matched_results_key]['predicted_score_of_gt_category'] = 1.0 - \
                    predicted_class_probs
                matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_class_idx

        # Load the different detection partitions
        true_positives = matched_results['true_positives']
        false_negatives = matched_results['false_negatives']
        false_positives = matched_results['false_positives']

        # Get the number of elements in each partition
        num_true_positives = true_positives['predicted_box_means'].shape[0]
        num_false_negatives = false_negatives['gt_box_means'].shape[0]
        num_false_positives = false_positives['predicted_box_means'].shape[0]

        per_class_output_list = []
        for class_idx in [1, 3]:
            true_positives_valid_idxs = true_positives['gt_converted_cat_idxs'] == class_idx
            false_positives_valid_idxs = false_positives['predicted_cat_idxs'] == class_idx

            # Compute classification metrics for every partition
            true_positives_cls_analysis = scoring_rules.retinanet_compute_cls_scores(
                true_positives, true_positives_valid_idxs)
            false_positives_cls_analysis = scoring_rules.retinanet_compute_cls_scores(
                false_positives, false_positives_valid_idxs)

            # Compute regression metrics for every partition
            true_positives_reg_analysis = scoring_rules.compute_reg_scores(
                true_positives, true_positives_valid_idxs)
            false_positives_reg_analysis = scoring_rules.compute_reg_scores_fn(
                false_positives, false_positives_valid_idxs)

            per_class_output_list.append(
                {'true_positives_cls_analysis': true_positives_cls_analysis,
                 'true_positives_reg_analysis': true_positives_reg_analysis,
                 'false_positives_cls_analysis': false_positives_cls_analysis,
                 'false_positives_reg_analysis': false_positives_reg_analysis})

        final_accumulated_output_dict = dict()
        final_average_output_dict = dict()

        for key in per_class_output_list[0].keys():
            average_output_dict = dict()
            for inner_key in per_class_output_list[0][key].keys():
                collected_values = [per_class_output[key][inner_key]
                                    for per_class_output in per_class_output_list if per_class_output[key][inner_key] is not None]
                collected_values = np.array(collected_values)

                if key in average_output_dict.keys():
                    # Use nan mean since some classes do not have duplicates for
                    # instance or has one duplicate for instance. torch.std returns nan in that case
                    # so we handle those here. This should not have any effect on the final results, as
                    # it only affects inter-class variance which we do not
                    # report anyways.
                    average_output_dict[key].update(
                        {inner_key: np.nanmean(collected_values)})
                    final_accumulated_output_dict[key].update(
                        {inner_key: collected_values})
                else:
                    average_output_dict.update(
                        {key: {inner_key: np.nanmean(collected_values)}})
                    final_accumulated_output_dict.update(
                        {key: {inner_key: collected_values}})

            final_average_output_dict.update(average_output_dict)
        # Summarize and print all
        table = PrettyTable()
        table.field_names = (['Output Type',
                              'Number of Instances',
                              'Cls Ignorance Score',
                              'Reg Ignorance Score'])
        table.add_row(
            [
                "True Positives:",
                num_true_positives,
                '{:.4f}'.format(
                    final_average_output_dict['true_positives_cls_analysis']['ignorance_score_mean']),
                '{:.4f}'.format(
                    final_average_output_dict['true_positives_reg_analysis']['ignorance_score_mean'])])

        table.add_row(
            [
                "False Positives:",
                num_false_positives,
                '{:.4f}'.format(
                    final_average_output_dict['false_positives_cls_analysis']['ignorance_score_mean']),
                '{:.4f}'.format(
                    final_average_output_dict['false_positives_reg_analysis']['total_entropy_mean'])])

        table.add_row(["False Negatives:",
                       num_false_negatives,
                       '-',
                       '-'])
        print(table)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
