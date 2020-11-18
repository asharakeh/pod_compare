import cv2
import numpy as np
import os
import ujson as json

from scipy.stats import entropy
from matplotlib import cm, pyplot as plt

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from core.evaluation_tools import evaluation_utils
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer


# noinspection PyTypeChecker
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

    # Build path to gt instances and inference output
    inference_output_dir = os.path.join(
        cfg['OUTPUT_DIR'],
        'inference',
        args.test_dataset,
        os.path.split(args.inference_config)[-1][:-5])

    # Get thresholds to perform evaluation on
    if min_allowed_score is None:
        # Check if F-1 Score has been previously computed.
        try:
            with open(os.path.join(inference_output_dir, "mAP_res.txt"), "r") as f:
                min_allowed_score = f.read().strip('][\n').split(', ')[-1]
                min_allowed_score = round(float(min_allowed_score), 4)
        except FileNotFoundError:
            # If not, process all detections. Not recommended as the results might be influenced by very low scoring
            # detections that would normally be removed in robotics/vision
            # applications.
            min_allowed_score = 0.0
    min_allowed_score = 0.5

    # get preprocessed instances
    preprocessed_predicted_instances, preprocessed_gt_instances = evaluation_utils.get_per_frame_preprocessed_instances(
        cfg, inference_output_dir, min_allowed_score)

    # get metacatalog and image infos
    meta_catalog = MetadataCatalog.get(args.test_dataset)
    images_info = json.load(open(meta_catalog.json_file, 'r'))['images']

    # Loop over all images and visualize errors
    for image_info in images_info:
        image_id = image_info['id']
        print(image_info['file_name'])
        image = cv2.imread(
            os.path.join(
                meta_catalog.image_root,
                image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        v = ProbabilisticVisualizer(
            image,
            meta_catalog,
            scale=1.5)
        class_list = v.metadata.as_dict()['thing_classes']

        predicted_box_means = preprocessed_predicted_instances['predicted_boxes'][image_id].cpu(
        ).numpy()
        gt_box_means = preprocessed_gt_instances['gt_boxes'][image_id].cpu(
        ).numpy()
        predicted_box_covariances = preprocessed_predicted_instances[
            'predicted_covar_mats'][image_id].cpu(
        ).numpy()

        predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs'][image_id]

        if predicted_cls_probs.shape[0] > 0:
            if cfg.MODEL.META_ARCHITECTURE == "ProbabilisticGeneralizedRCNN":
                _, predicted_classes = predicted_cls_probs[:, :-1].max(
                    1)
                predicted_entropies = entropy(
                    predicted_cls_probs.cpu().numpy(), base=2)

            else:
                predicted_scores, predicted_classes = predicted_cls_probs.max(
                    1)
                predicted_entropies = entropy(
                    np.stack(
                        (predicted_scores.cpu().numpy(),
                         1 - predicted_scores.cpu().numpy())),
                    base=2)
            predicted_classes = predicted_classes.cpu(
            ).numpy()
            predicted_classes = [class_list[p_class]
                                 for p_class in predicted_classes]
            assigned_colors = cm.autumn(predicted_entropies)
            predicted_scores = predicted_scores.cpu().numpy()
        else:
            predicted_scores=np.array([])
            predicted_classes = np.array([])
            assigned_colors = []

        gt_cat_idxs = preprocessed_gt_instances['gt_cat_idxs'][image_id].cpu(
        ).numpy()
        thing_dataset_id_to_contiguous_id = meta_catalog.thing_dataset_id_to_contiguous_id
        if gt_cat_idxs.shape[0] > 0:
            gt_labels = [class_list[thing_dataset_id_to_contiguous_id[gt_class]]
                         for gt_class in gt_cat_idxs[:, 0]]
        else:
            gt_labels = []

        # noinspection PyTypeChecker
        _ = v.overlay_covariance_instances(
            boxes=gt_box_means,
            assigned_colors=[
                'lightgreen' for _ in gt_box_means],
            labels=gt_labels,
            alpha=1.0)
        plotted_detections = v.overlay_covariance_instances(
            boxes=predicted_box_means,
            covariance_matrices=predicted_box_covariances,
            assigned_colors=assigned_colors,
            alpha=1.0,
            labels=predicted_scores)

        cv2.imshow(
            'Detected Instances.',
            cv2.cvtColor(
                plotted_detections.get_image(),
                cv2.COLOR_RGB2BGR))
        cv2.waitKey()


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
