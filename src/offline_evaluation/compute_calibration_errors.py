import calibration as cal
import os
import torch

from prettytable import PrettyTable

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import evaluation_utils
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
                matched_results[matched_results_key]['gt_cat_idxs'] = gt_converted_cat_idxs
            if 'predicted_cls_probs' in matched_results[matched_results_key].keys(
            ):
                predicted_class_probs, predicted_cat_idxs = matched_results[
                    matched_results_key]['predicted_cls_probs'][:, :-1].max(1)

                matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_cat_idxs
                matched_results[matched_results_key]['output_logits'] = predicted_class_probs

        # Load the different detection partitions
        true_positives = matched_results['true_positives']
        duplicates = matched_results['duplicates']
        false_positives = matched_results['false_positives']

        # Get the number of elements in each partition
        cls_min_uncertainty_error_list = []

        reg_maximum_calibration_error_list = []
        reg_expected_calibration_error_list = []
        reg_min_uncertainty_error_list = []

        all_predicted_scores = torch.cat(
            (true_positives['predicted_cls_probs'].flatten(),
             duplicates['predicted_cls_probs'].flatten(),
             false_positives['predicted_cls_probs'].flatten()),
            0)
        all_gt_scores = torch.cat(
            (torch.nn.functional.one_hot(
                true_positives['gt_cat_idxs'],
                true_positives['predicted_cls_probs'].shape[1]).flatten().to(device),
                torch.nn.functional.one_hot(
                duplicates['gt_cat_idxs'],
                duplicates['predicted_cls_probs'].shape[1]).flatten().to(device),
                torch.zeros_like(
                false_positives['predicted_cls_probs'].type(
                    torch.LongTensor).flatten()).to(device)),
            0)

        # Compute classification calibration error using calibration
        # library
        cls_marginal_calibration_error = cal.get_calibration_error(
            all_predicted_scores.cpu().numpy(), all_gt_scores.cpu().numpy())

        for class_idx in cat_mapping_dict.values():
            true_positives_valid_idxs = true_positives['gt_converted_cat_idxs'] == class_idx
            duplicates_valid_idxs = duplicates['gt_converted_cat_idxs'] == class_idx
            false_positives_valid_idxs = false_positives['predicted_cat_idxs'] == class_idx

            # For the rest of the code, gt_scores need to be ones or zeros. All
            # processing is done on a per-class basis
            all_gt_scores = torch.cat(
                (torch.ones_like(
                    true_positives['gt_converted_cat_idxs'][true_positives_valid_idxs]).to(device),
                    torch.zeros_like(
                    duplicates['gt_converted_cat_idxs'][duplicates_valid_idxs]).to(device),
                    torch.zeros_like(
                    false_positives['predicted_cat_idxs'][false_positives_valid_idxs]).to(device)),
                0).type(
                torch.DoubleTensor)

            # Compute classification minimum uncertainty error
            distribution_params = torch.cat(
                (true_positives['output_logits'][true_positives_valid_idxs],
                 duplicates['output_logits'][duplicates_valid_idxs],
                 false_positives['output_logits'][false_positives_valid_idxs]),
                0)
            all_predicted_cat_entropy = -torch.log(distribution_params)

            random_idxs = torch.randperm(all_predicted_cat_entropy.shape[0])

            all_predicted_cat_entropy = all_predicted_cat_entropy[random_idxs]
            all_gt_scores_cls = all_gt_scores[random_idxs]
            sorted_entropies, sorted_idxs = all_predicted_cat_entropy.sort()
            sorted_gt_idxs_tp = all_gt_scores_cls[sorted_idxs]
            sorted_gt_idxs_fp = 1.0 - sorted_gt_idxs_tp

            tp_cum_sum = torch.cumsum(sorted_gt_idxs_tp, 0)
            fp_cum_sum = torch.cumsum(sorted_gt_idxs_fp, 0)
            cls_u_errors = 0.5 * (sorted_gt_idxs_tp.sum(0) - tp_cum_sum) / \
                sorted_gt_idxs_tp.sum(0) + 0.5 * fp_cum_sum / sorted_gt_idxs_fp.sum(0)
            cls_min_u_error = cls_u_errors.min()
            cls_min_uncertainty_error_list.append(cls_min_u_error)

            # Compute regression calibration errors. False negatives cant be evaluated since
            # those do not have ground truth.
            all_predicted_means = torch.cat(
                (true_positives['predicted_box_means'][true_positives_valid_idxs],
                 duplicates['predicted_box_means'][duplicates_valid_idxs]),
                0)

            all_predicted_covariances = torch.cat(
                (true_positives['predicted_box_covariances'][true_positives_valid_idxs],
                 duplicates['predicted_box_covariances'][duplicates_valid_idxs]),
                0)

            all_predicted_gt = torch.cat(
                (true_positives['gt_box_means'][true_positives_valid_idxs],
                 duplicates['gt_box_means'][duplicates_valid_idxs]),
                0)

            all_predicted_covariances = torch.diagonal(
                all_predicted_covariances, dim1=1, dim2=2)

            # The assumption of uncorrelated components is not accurate, especially when estimating full
            # covariance matrices. However, using scipy to compute multivariate cdfs is very very
            # time consuming for such large amounts of data.
            reg_maximum_calibration_error = []
            reg_expected_calibration_error = []

            # Regression calibration is computed for every box dimension
            # separately, and averaged after.
            for box_dim in range(all_predicted_gt.shape[1]):
                all_predicted_means_current_dim = all_predicted_means[:, box_dim]
                all_predicted_gt_current_dim = all_predicted_gt[:, box_dim]
                all_predicted_covariances_current_dim = all_predicted_covariances[:, box_dim]
                normal_dists = torch.distributions.Normal(
                    all_predicted_means_current_dim,
                    scale=torch.sqrt(all_predicted_covariances_current_dim))
                all_predicted_scores = normal_dists.cdf(
                    all_predicted_gt_current_dim)

                reg_calibration_error = []
                histogram_bin_step_size = 1 / 15.0
                for i in torch.arange(
                        0.0,
                        1.0 - histogram_bin_step_size,
                        histogram_bin_step_size):
                    # Get number of elements in bin
                    elements_in_bin = (
                        all_predicted_scores < (i + histogram_bin_step_size))
                    num_elems_in_bin_i = elements_in_bin.type(
                        torch.FloatTensor).to(device).sum()

                    # Compute calibration error from "Accurate uncertainties for deep
                    # learning using calibrated regression" paper.
                    reg_calibration_error.append(
                        (num_elems_in_bin_i / all_predicted_scores.shape[0] - (i + histogram_bin_step_size)) ** 2)

                calibration_error = torch.stack(
                    reg_calibration_error).to(device)
                reg_maximum_calibration_error.append(calibration_error.max())
                reg_expected_calibration_error.append(calibration_error.mean())

            reg_maximum_calibration_error_list.append(
                reg_maximum_calibration_error)
            reg_expected_calibration_error_list.append(
                reg_expected_calibration_error)

            # Compute regression minimum uncertainty error
            all_predicted_covars = torch.cat(
                (true_positives['predicted_box_covariances'][true_positives_valid_idxs],
                 duplicates['predicted_box_covariances'][duplicates_valid_idxs],
                 false_positives['predicted_box_covariances'][false_positives_valid_idxs]),
                0)

            all_predicted_distributions = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(
                all_predicted_covars.shape[0:2]).to(device), all_predicted_covars + 1e-4 * torch.eye(all_predicted_covars.shape[2]).to(device))

            all_predicted_reg_entropy = all_predicted_distributions.entropy()
            random_idxs = torch.randperm(all_predicted_reg_entropy.shape[0])

            all_predicted_reg_entropy = all_predicted_reg_entropy[random_idxs]
            all_gt_scores_reg = all_gt_scores[random_idxs]

            sorted_entropies, sorted_idxs = all_predicted_reg_entropy.sort()
            sorted_gt_idxs_tp = all_gt_scores_reg[sorted_idxs]
            sorted_gt_idxs_fp = 1.0 - sorted_gt_idxs_tp

            tp_cum_sum = torch.cumsum(sorted_gt_idxs_tp, 0)
            fp_cum_sum = torch.cumsum(sorted_gt_idxs_fp, 0)
            reg_u_errors = 0.5 * ((sorted_gt_idxs_tp.sum(0) - tp_cum_sum) /
                                  sorted_gt_idxs_tp.sum(0)) + 0.5 * (fp_cum_sum / sorted_gt_idxs_fp.sum(0))
            reg_min_u_error = reg_u_errors.min()
            reg_min_uncertainty_error_list.append(reg_min_u_error)

        # Summarize and print all
        table = PrettyTable()
        table.field_names = (['Cls Marginal Calibration Error',
                              'Reg Expected Calibration Error',
                              'Reg Maximum Calibration Error',
                              'Cls Minimum Uncertainty Error',
                              'Reg Minimum Uncertainty Error'])

        reg_expected_calibration_error = torch.stack([torch.stack(
            reg, 0) for reg in reg_expected_calibration_error_list], 0)
        reg_expected_calibration_error = reg_expected_calibration_error[
            ~torch.isnan(reg_expected_calibration_error)].mean()

        reg_maximum_calibration_error = torch.stack([torch.stack(
            reg, 0) for reg in reg_maximum_calibration_error_list], 0)
        reg_maximum_calibration_error = reg_maximum_calibration_error[
            ~torch.isnan(reg_maximum_calibration_error)].mean()

        cls_min_u_error = torch.stack(cls_min_uncertainty_error_list, 0)
        cls_min_u_error = cls_min_u_error[
            ~torch.isnan(cls_min_u_error)].mean()

        reg_min_u_error = torch.stack(reg_min_uncertainty_error_list, 0)
        reg_min_u_error = reg_min_u_error[
            ~torch.isnan(reg_min_u_error)].mean()

        table.add_row(['{:.4f}'.format(cls_marginal_calibration_error),
                       '{:.4f}'.format(reg_expected_calibration_error.cpu().numpy().tolist()),
                       '{:.4f}'.format(reg_maximum_calibration_error.cpu().numpy().tolist()),
                       '{:.4f}'.format(cls_min_u_error.cpu().numpy().tolist()),
                       '{:.4f}'.format(reg_min_u_error.cpu().numpy().tolist())])
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
