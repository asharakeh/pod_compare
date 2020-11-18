import numpy as np
import os
import tqdm
import torch
import ujson as json

from collections import defaultdict

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, pairwise_iou

# Project imports
from core.datasets import metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_predictions_preprocess(
        predicted_instances,
        min_allowed_score=0.0,
        is_odd=False):
    predicted_boxes, predicted_cls_probs, predicted_covar_mats = defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor), defaultdict(
            torch.Tensor)

    for predicted_instance in predicted_instances:
        # Remove predictions with undefined category_id. This is used when the training and inference datasets come from
        # different data such as COCO-->VOC or BDD-->Kitti. Only happens if not ODD dataset, else all detections will
        # be removed.
        if not is_odd:
            skip_test = (
                predicted_instance['category_id'] == -
                1) or (
                np.array(
                    predicted_instance['cls_prob']).max(0) < min_allowed_score)
        else:
            skip_test = np.array(
                predicted_instance['cls_prob']).max(0) < min_allowed_score

        if skip_test:
            continue

        box_inds = predicted_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])

        predicted_boxes[predicted_instance['image_id']] = torch.cat((predicted_boxes[predicted_instance['image_id']].to(
            device), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))

        predicted_cls_probs[predicted_instance['image_id']] = torch.cat((predicted_cls_probs[predicted_instance['image_id']].to(
            device), torch.as_tensor([predicted_instance['cls_prob']], dtype=torch.float32).to(device)))

        box_covar = np.array(predicted_instance['bbox_covar'])
        transformation_mat = np.array([[1.0, 0, 0, 0],
                                       [0, 1.0, 0, 0],
                                       [1.0, 0, 1.0, 0],
                                       [0, 1.0, 0.0, 1.0]])
        cov_pred = np.matmul(
            np.matmul(
                transformation_mat,
                box_covar),
            transformation_mat.T).tolist()

        predicted_covar_mats[predicted_instance['image_id']] = torch.cat(
            (predicted_covar_mats[predicted_instance['image_id']].to(device), torch.as_tensor([cov_pred], dtype=torch.float32).to(device)))

    return dict({'predicted_boxes': predicted_boxes,
                 'predicted_cls_probs': predicted_cls_probs,
                 'predicted_covar_mats': predicted_covar_mats})


def eval_gt_preprocess(gt_instances):
    gt_boxes, gt_cat_idxs = defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor)
    for gt_instance in gt_instances:
        box_inds = gt_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])
        gt_boxes[gt_instance['image_id']] = torch.cat((gt_boxes[gt_instance['image_id']].cuda(
        ), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))
        gt_cat_idxs[gt_instance['image_id']] = torch.cat((gt_cat_idxs[gt_instance['image_id']].cuda(
        ), torch.as_tensor([[gt_instance['category_id']]], dtype=torch.float32).to(device)))

    return dict({'gt_boxes': gt_boxes,
                 'gt_cat_idxs': gt_cat_idxs})


def get_matched_results(
        cfg,
        inference_output_dir,
        iou_min=0.1,
        iou_correct=0.7,
        min_allowed_score=0.0):
    try:
        matched_results = torch.load(
            os.path.join(
                inference_output_dir,
                "matched_results_{}_{}_{}.pth".format(
                    iou_min,
                    iou_correct,
                    min_allowed_score)), map_location=device)

        return matched_results
    except FileNotFoundError:
        preprocessed_predicted_instances, preprocessed_gt_instances = get_per_frame_preprocessed_instances(
            cfg, inference_output_dir, min_allowed_score)
        predicted_box_means = preprocessed_predicted_instances['predicted_boxes']
        predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs']
        predicted_box_covariances = preprocessed_predicted_instances['predicted_covar_mats']
        gt_box_means = preprocessed_gt_instances['gt_boxes']
        gt_cat_idxs = preprocessed_gt_instances['gt_cat_idxs']

        matched_results = match_predictions_to_groundtruth(
            predicted_box_means,
            predicted_cls_probs,
            predicted_box_covariances,
            gt_box_means,
            gt_cat_idxs,
            iou_min,
            iou_correct)

        torch.save(
            matched_results,
            os.path.join(
                inference_output_dir,
                "matched_results_{}_{}_{}.pth".format(
                    iou_min,
                    iou_correct,
                    min_allowed_score)))

        return matched_results


def get_per_frame_preprocessed_instances(
        cfg, inference_output_dir, min_allowed_score=0.0):
    prediction_file_name = os.path.join(
        inference_output_dir,
        'coco_instances_results.json')

    meta_catalog = MetadataCatalog.get(cfg.ACTUAL_TEST_DATASET)
    # Process GT
    print("Began pre-processing ground truth annotations...")
    try:
        preprocessed_gt_instances = torch.load(
            os.path.join(
                os.path.split(meta_catalog.json_file)[0],
                "preprocessed_gt_instances.pth"), map_location=device)
    except FileNotFoundError:
        gt_info = json.load(
            open(
                meta_catalog.json_file,
                'r'))
        gt_instances = gt_info['annotations']
        preprocessed_gt_instances = eval_gt_preprocess(
            gt_instances)
        torch.save(
            preprocessed_gt_instances,
            os.path.join(
                os.path.split(meta_catalog.json_file)[0],
                "preprocessed_gt_instances.pth"))
    print("Done!")
    print("Began pre-processing predicted instances...")
    try:
        preprocessed_predicted_instances = torch.load(
            os.path.join(
                inference_output_dir,
                "preprocessed_predicted_instances_{}.pth".format(min_allowed_score)),
            map_location=device)
    # Process predictions
    except FileNotFoundError:
        predicted_instances = json.load(open(prediction_file_name, 'r'))
        preprocessed_predicted_instances = eval_predictions_preprocess(
            predicted_instances, min_allowed_score)
        torch.save(
            preprocessed_predicted_instances,
            os.path.join(
                inference_output_dir,
                "preprocessed_predicted_instances_{}.pth".format(min_allowed_score)))
    print("Done!")

    return preprocessed_predicted_instances, preprocessed_gt_instances


def match_predictions_to_groundtruth(predicted_box_means,
                                     predicted_cls_probs,
                                     predicted_box_covariances,
                                     gt_box_means,
                                     gt_cat_idxs,
                                     iou_min=0.1,
                                     iou_correct=0.7):

    true_positives = dict(
        {
            'predicted_box_means': torch.Tensor().to(device),
            'predicted_box_covariances': torch.Tensor().to(device),
            'predicted_cls_probs': torch.Tensor().to(device),
            'gt_box_means': torch.Tensor().to(device),
            'gt_cat_idxs': torch.Tensor().to(device),
            'iou_with_ground_truth': torch.Tensor().to(device)})

    duplicates = dict({'predicted_box_means': torch.Tensor().to(device),
                       'predicted_box_covariances': torch.Tensor().to(device),
                       'predicted_cls_probs': torch.Tensor().to(device),
                       'gt_box_means': torch.Tensor().to(device),
                       'gt_cat_idxs': torch.Tensor().to(device),
                       'iou_with_ground_truth': torch.Tensor().to(device)})

    false_positives = dict({'predicted_box_means': torch.Tensor().to(device),
                            'predicted_box_covariances': torch.Tensor().to(device),
                            'predicted_cls_probs': torch.Tensor().to(device)})

    false_negatives = dict({'gt_box_means': torch.Tensor().to(device),
                            'gt_cat_idxs': torch.Tensor().to(device)})

    with tqdm.tqdm(total=len(predicted_box_means)) as pbar:
        for key in predicted_box_means.keys():
            pbar.update(1)

            # Check if gt available, if not all detections go to false
            # positives
            if key not in gt_box_means.keys():
                false_positives['predicted_box_means'] = torch.cat(
                    (false_positives['predicted_box_means'], predicted_box_means[key]))
                false_positives['predicted_cls_probs'] = torch.cat(
                    (false_positives['predicted_cls_probs'], predicted_cls_probs[key]))
                false_positives['predicted_box_covariances'] = torch.cat(
                    (false_positives['predicted_box_covariances'], predicted_box_covariances[key]))
                continue

            # Compute iou between gt boxes and all predicted boxes in frame
            frame_gt_boxes = Boxes(gt_box_means[key])
            frame_predicted_boxes = Boxes(predicted_box_means[key])

            match_iou = pairwise_iou(frame_gt_boxes, frame_predicted_boxes)

            # Get false negative ground truth, which are fully missed.
            # These can be found by looking for ground truth boxes that have an
            # iou < iou_min with any detection
            false_negative_idxs = (match_iou <= iou_min).all(1)
            false_negatives['gt_box_means'] = torch.cat(
                (false_negatives['gt_box_means'],
                 gt_box_means[key][false_negative_idxs]))
            false_negatives['gt_cat_idxs'] = torch.cat(
                (false_negatives['gt_cat_idxs'],
                 gt_cat_idxs[key][false_negative_idxs]))

            # False positives are detections that have an iou < match iou with
            # any ground truth object.
            false_positive_idxs = (match_iou <= iou_min).all(0)
            false_positives['predicted_box_means'] = torch.cat(
                (false_positives['predicted_box_means'],
                 predicted_box_means[key][false_positive_idxs]))
            false_positives['predicted_cls_probs'] = torch.cat(
                (false_positives['predicted_cls_probs'],
                 predicted_cls_probs[key][false_positive_idxs]))
            false_positives['predicted_box_covariances'] = torch.cat(
                (false_positives['predicted_box_covariances'],
                 predicted_box_covariances[key][false_positive_idxs]))

            # True positives are any detections with match iou > iou correct. We need to separate these detections to
            # True positive and duplicate set. The true positive detection is the detection assigned the highest score
            # by the neural network.
            true_positive_idxs = torch.nonzero(match_iou >= iou_correct)

            # Setup tensors to allow assignment of detections only once.
            gt_idxs_processed = torch.tensor(
                []).type(torch.LongTensor).to(device)

            for i in torch.arange(frame_gt_boxes.tensor.shape[0]):
                # Check if true positive has been previously assigned to a ground truth box and remove it if this is
                # the case. Very rare occurrence but need to handle it
                # nevertheless.
                gt_idxs = true_positive_idxs[true_positive_idxs[:, 0] == i][:, 1]
                non_valid_idxs = torch.nonzero(
                    gt_idxs_processed[..., None] == gt_idxs)

                if non_valid_idxs.shape[0] > 0:
                    gt_idxs[non_valid_idxs[:, 1]] = -1
                    gt_idxs = gt_idxs[gt_idxs != -1]

                if gt_idxs.shape[0] > 0:
                    current_matches_predicted_cls_probs = predicted_cls_probs[key][gt_idxs]
                    max_score, _ = torch.max(
                        current_matches_predicted_cls_probs, 1)
                    _, max_idxs = max_score.topk(max_score.shape[0])

                    if max_idxs.shape[0] > 1:
                        max_idx = max_idxs[0]
                        duplicate_idxs = max_idxs[1:]
                    else:
                        max_idx = max_idxs
                        duplicate_idxs = torch.empty(0).to(device)

                    current_matches_predicted_box_means = predicted_box_means[key][gt_idxs]
                    current_matches_predicted_box_covariances = predicted_box_covariances[
                        key][gt_idxs]

                    # Highest scoring detection goes to true positives
                    true_positives['predicted_box_means'] = torch.cat(
                        (true_positives['predicted_box_means'],
                         current_matches_predicted_box_means[max_idx:max_idx + 1, :]))
                    true_positives['predicted_cls_probs'] = torch.cat(
                        (true_positives['predicted_cls_probs'],
                         current_matches_predicted_cls_probs[max_idx:max_idx + 1, :]))
                    true_positives['predicted_box_covariances'] = torch.cat(
                        (true_positives['predicted_box_covariances'],
                         current_matches_predicted_box_covariances[max_idx:max_idx + 1, :]))

                    true_positives['gt_box_means'] = torch.cat(
                        (true_positives['gt_box_means'], gt_box_means[key][i:i + 1, :]))
                    true_positives['gt_cat_idxs'] = torch.cat(
                        (true_positives['gt_cat_idxs'], gt_cat_idxs[key][i:i + 1, :]))
                    true_positives['iou_with_ground_truth'] = torch.cat(
                        (true_positives['iou_with_ground_truth'], match_iou[i, gt_idxs][max_idx:max_idx + 1]))

                    # Lower scoring redundant detections go to duplicates
                    if duplicate_idxs.shape[0] > 1:
                        duplicates['predicted_box_means'] = torch.cat(
                            (duplicates['predicted_box_means'], current_matches_predicted_box_means[duplicate_idxs, :]))
                        duplicates['predicted_cls_probs'] = torch.cat(
                            (duplicates['predicted_cls_probs'], current_matches_predicted_cls_probs[duplicate_idxs, :]))
                        duplicates['predicted_box_covariances'] = torch.cat(
                            (duplicates['predicted_box_covariances'],
                             current_matches_predicted_box_covariances[duplicate_idxs, :]))

                        duplicates['gt_box_means'] = torch.cat(
                            (duplicates['gt_box_means'], gt_box_means[key][np.repeat(i, duplicate_idxs.shape[0]), :]))
                        duplicates['gt_cat_idxs'] = torch.cat(
                            (duplicates['gt_cat_idxs'], gt_cat_idxs[key][np.repeat(i, duplicate_idxs.shape[0]), :]))
                        duplicates['iou_with_ground_truth'] = torch.cat(
                            (duplicates['iou_with_ground_truth'],
                             match_iou[i, gt_idxs][duplicate_idxs]))

                    elif duplicate_idxs.shape[0] == 1:
                        # Special case when only one duplicate exists, required to
                        # index properly for torch.cat
                        duplicates['predicted_box_means'] = torch.cat(
                            (duplicates['predicted_box_means'],
                             current_matches_predicted_box_means[duplicate_idxs:duplicate_idxs + 1, :]))
                        duplicates['predicted_cls_probs'] = torch.cat(
                            (duplicates['predicted_cls_probs'],
                             current_matches_predicted_cls_probs[duplicate_idxs:duplicate_idxs + 1, :]))
                        duplicates['predicted_box_covariances'] = torch.cat(
                            (duplicates['predicted_box_covariances'],
                             current_matches_predicted_box_covariances[duplicate_idxs:duplicate_idxs + 1, :]))

                        duplicates['gt_box_means'] = torch.cat(
                            (duplicates['gt_box_means'], gt_box_means[key][i:i + 1, :]))
                        duplicates['gt_cat_idxs'] = torch.cat(
                            (duplicates['gt_cat_idxs'], gt_cat_idxs[key][i:i + 1, :]))
                        duplicates['iou_with_ground_truth'] = torch.cat(
                            (duplicates['iou_with_ground_truth'],
                             match_iou[i, gt_idxs][duplicate_idxs:duplicate_idxs + 1]))

    matched_results = dict()
    matched_results.update({"true_positives": true_positives,
                            "duplicates": duplicates,
                            "false_positives": false_positives,
                            "false_negatives": false_negatives})
    return matched_results


def get_thing_dataset_id_to_contiguous_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id):

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    if (train_thing_dataset_id_to_contiguous_id == test_thing_dataset_id_to_contiguous_id) or (
            cfg.DATASETS.TRAIN[0] == 'coco_not_in_voc_2017_train'):
        cat_mapping_dict = test_thing_dataset_id_to_contiguous_id
    else:
        # If not equal, two situations: 1) BDD to KITTI or 2) COCO to PASCAL
        cat_mapping_dict = test_thing_dataset_id_to_contiguous_id
        if 'voc' in args.test_dataset and 'coco' in cfg.DATASETS.TRAIN[0]:
            dataset_mapping_dict = dict(
                (v, k) for k, v in metadata.COCO_TO_VOC_CONTIGUOUS_ID.items())
        elif 'kitti' in args.test_dataset and 'bdd' in cfg.DATASETS.TRAIN[0]:
            dataset_mapping_dict = dict(
                (v, k) for k, v in metadata.BDD_TO_KITTI_CONTIGUOUS_ID.items())
        else:
            ValueError(
                'Cannot generate category mapping dictionary. Please check if provided training and inference dataset names are compatible.')

        cat_mapping_dict = dict(
            (k, dataset_mapping_dict[v]) for k, v in cat_mapping_dict.items())

    return cat_mapping_dict
