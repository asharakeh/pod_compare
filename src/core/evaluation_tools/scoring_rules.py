import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retinanet_compute_cls_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for multilabel classification results provided by retinanet.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation
    Returns:
        output_dict (dict): dictionary containing ignorance and brier score.
    """

    output_dict = {}
    num_forecasts = input_matches['predicted_cls_probs'][valid_idxs].shape[0]

    # Construct binary probability vectors. Essential for RetinaNet as it uses
    # multilabel and not multiclass formulation.

    predicted_class_probs = input_matches['predicted_score_of_gt_category'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_class_probs.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None})
        return output_dict
    predicted_multilabel_probs = torch.stack(
        [predicted_class_probs, 1.0 - predicted_class_probs], dim=1)

    correct_multilabel_probs = torch.stack(
        [torch.ones(num_forecasts),
         torch.zeros(num_forecasts)], dim=1).to(device)

    predicted_log_likelihood_of_correct_category = (
        -correct_multilabel_probs * torch.log(predicted_multilabel_probs)).sum(1)

    cls_ignorance_score_mean = predicted_log_likelihood_of_correct_category.mean()
    output_dict.update({'ignorance_score_mean': cls_ignorance_score_mean.to(device).tolist()})

    return output_dict


def compute_reg_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for regression results.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation

    Returns:
        output_dict (dict): dictionary containing ignorance and energy scores.
    """
    output_dict = {}

    predicted_box_means = input_matches['predicted_box_means'][valid_idxs]
    predicted_box_covars = input_matches['predicted_box_covariances'][valid_idxs]
    gt_box_means = input_matches['gt_box_means'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_box_means.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None,
                            'mean_squared_error': None})
        return output_dict

    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        predicted_box_means, predicted_box_covars + 1e-2 * torch.eye(predicted_box_covars.shape[2]).to(device))
    negative_log_prob = - \
        predicted_multivariate_normal_dists.log_prob(gt_box_means)
    negative_log_prob_mean = negative_log_prob.mean()
    output_dict.update({'ignorance_score_mean': negative_log_prob_mean.to(
        device).tolist()})

    mean_squared_error = ((predicted_box_means - gt_box_means)**2).mean()

    output_dict.update({'mean_squared_error': mean_squared_error.to(device).tolist(
    )})

    return output_dict


def compute_reg_scores_fn(false_negatives, valid_idxs):
    """
    Computes proper scoring rule for regression false positive.

    Args:
        false_negatives (dict): dictionary containing false_negatives
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation

    Returns:
        output_dict (dict): dictionary containing false positives ignorance and energy scores.
    """
    output_dict = {}

    predicted_box_means = false_negatives['predicted_box_means'][valid_idxs]
    predicted_box_covars = false_negatives['predicted_box_covariances'][valid_idxs]

    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        predicted_box_means, predicted_box_covars + 1e-2 * torch.eye(predicted_box_covars.shape[2]).to(device))

    # If no valid idxs, do not perform computation
    if predicted_box_means.shape[0] == 0:
        output_dict.update({'total_entropy_mean': None})
        return output_dict

    total_entropy = predicted_multivariate_normal_dists.entropy()
    total_entropy_mean = total_entropy.mean()

    output_dict.update({'total_entropy_mean': total_entropy_mean.to(
        device).tolist()})

    return output_dict
