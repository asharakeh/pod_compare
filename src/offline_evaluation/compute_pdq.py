import json
import numpy as np
import os

from collections import defaultdict
from prettytable import PrettyTable

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import pdq_data_holders, pdq
from core.evaluation_tools.evaluation_utils import get_thing_dataset_id_to_contiguous_id_dict
from core.setup import setup_config, setup_arg_parser


def main(args, min_allowed_score=None):
    # Setup config
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    # Build path to inference output
    inference_output_dir = os.path.join(
        cfg['OUTPUT_DIR'],
        'inference',
        args.test_dataset,
        os.path.split(args.inference_config)[-1][:-5])

    # Check if F-1 Score has been previously computed.
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

    prediction_file_name = os.path.join(
        inference_output_dir,
        'coco_instances_results.json')

    meta_catalog = MetadataCatalog.get(args.test_dataset)
    gt_info = json.load(open(meta_catalog.json_file, 'r'))
    gt_image_info = gt_info['images']
    gt_instances = gt_info['annotations']

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

    gt_boxes, gt_cat_idxs, gt_image_sizes = defaultdict(
        list), defaultdict(list), defaultdict(list)
    for gt_instance in gt_instances:
        gt_boxes[gt_instance['image_id']].append(gt_instance['bbox'])
        gt_cat_idxs[gt_instance['image_id']].append(gt_instance['category_id'])
        gt_image_sizes[gt_instance['image_id']] = gt_image_info

    predicted_instances = json.load(open(prediction_file_name, 'r'))

    predicted_boxes, predicted_scores, predicted_covar_mats = defaultdict(
        list), defaultdict(list), defaultdict(list)

    for predicted_instance in predicted_instances:
        predicted_boxes[predicted_instance['image_id']].append(
            predicted_instance['bbox'])
        predicted_scores[predicted_instance['image_id']].append(
            predicted_instance['cls_prob'])
        predicted_covar_mats[predicted_instance['image_id']].append(
            predicted_instance['bbox_covar'])

    print("Constructing Per-Frame GT/Prediction Lists ...")

    gt_image_info_list = [gt_image_info[x:x + 1000]
                          for x in range(0, len(gt_image_info), 1000)]

    score = []
    TP = []
    FP = []
    FN = []
    avg_spatial_quality = []
    avg_label_quality = []
    avg_overall_quality = []

    for gt_image_info_i in gt_image_info_list:
        match_list = []
        for frame in gt_image_info_i:
            frame_id = frame['id']

            frame_boxes_gt = np.array(gt_boxes[frame_id])
            frame_cat_gt = np.array(gt_cat_idxs[frame_id])

            # Create GT list
            gt_instance_list = []
            for cat_gt, box_2d_gt in zip(frame_cat_gt, frame_boxes_gt):
                if cat_gt not in [1, 3]:
                    continue
                seg_mask = np.zeros(
                    [frame['height'], frame['width']], dtype=np.bool)
                box_inds = box_2d_gt.astype(np.int32).tolist()
                box_inds = [
                    box_inds[0],
                    box_inds[1],
                    box_inds[0] +
                    box_inds[2],
                    box_inds[1] +
                    box_inds[3]]
                seg_mask[box_inds[1]:box_inds[3],
                         box_inds[0]:box_inds[2]] = True
                gt_instance = pdq_data_holders.GroundTruthInstance(
                    seg_mask, cat_mapping_dict[cat_gt], 0, 0, bounding_box=box_inds)
                gt_instance_list.append(gt_instance)

            # Create Detection list
            frame_boxes_predicted = np.array(predicted_boxes[frame_id])
            frame_score_predicted = np.array(predicted_scores[frame_id])
            frame_cov_predicted = np.array(predicted_covar_mats[frame_id])

            det_instance_list = []
            for box_2d_pred, score_pred, cov_pred in zip(
                    frame_boxes_predicted, frame_score_predicted, frame_cov_predicted):
                transformation_mat = np.array([[1.0, 0, 0, 0],
                                               [0, 1.0, 0, 0],
                                               [1.0, 0, 1.0, 0],
                                               [0, 1.0, 0.0, 1.0]])
                cov_pred = np.matmul(
                    np.matmul(
                        transformation_mat,
                        cov_pred),
                    transformation_mat.T)

                box_2d_pred = [
                    box_2d_pred[0],
                    box_2d_pred[1],
                    box_2d_pred[0] +
                    box_2d_pred[2],
                    box_2d_pred[1] +
                    box_2d_pred[3]]
                if np.max(score_pred) >= min_allowed_score:
                    box_processed = np.array(box_2d_pred).astype(np.int32)
                    cov_processed = [cov_pred[0:2, 0:2] + np.diag(np.array([0.001, 0.001])), cov_pred[2:4, 2:4] + np.diag(np.array([0.001, 0.001]))]
                    det_instance = pdq_data_holders.PBoxDetInst(
                        score_pred, box_processed, cov_processed)
                    det_instance_list.append(det_instance)
            match_list.append((gt_instance_list, det_instance_list))
        print("Done")

        print("PDQ starting:")
        evaluator = pdq.PDQ()
        score.append(evaluator.score(match_list) * 100)
        TP_i, FP_i, FN_i = evaluator.get_assignment_counts()
        TP.append(TP_i)
        FP.append(FP_i)
        FN.append(FN_i)
        avg_spatial_quality.append(evaluator.get_avg_spatial_score())
        avg_label_quality.append(evaluator.get_avg_label_score())
        avg_overall_quality.append(
            evaluator.get_avg_overall_quality_score())
        print("PDQ Ended")

    score = sum(score) / len(score)
    TP = sum(TP)
    FP = sum(FP)
    FN = sum(FN)
    avg_spatial_quality = sum(avg_spatial_quality) / len(avg_spatial_quality)
    avg_label_quality = sum(avg_label_quality) / len(avg_label_quality)
    avg_overall_quality = sum(avg_overall_quality) / len(avg_overall_quality)

    table = PrettyTable(['score',
                         'True Positives',
                         'False Positives',
                         'False Negatives',
                         'Average Spatial Quality',
                         'Average Label Quality',
                         'Average Overall Quality'])

    table.add_row([score, TP, FP, FN, avg_spatial_quality,
                   avg_label_quality, avg_overall_quality])

    print(table)

    text_file_name = os.path.join(
        inference_output_dir,
        'pdq_res.txt')

    with open(text_file_name, "w") as text_file:
        print(table, file=text_file)


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
