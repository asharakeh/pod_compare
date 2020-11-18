"""
This code is all adapted from the original ACRV Robotic Vision Challenge code.
Adaptations have been made to enable some of the extra functionality needed in this repository.
Link to original code: https://github.com/jskinn/rvchallenge-evaluation/blob/master/pdq.py
Link to challenge websites:
    - CVPR 2019: https://competitions.codalab.org/competitions/20940
    - Continuous: https://competitions.codalab.org/competitions/21727
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gmean
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


_SMALL_VAL = 1e-14


class PDQ(object):
    """
    Class for calculating PDQ for a set of images.
    Extension of the code used in the 1st robotic vision challenge (RVC1) code.
    Link to RVC1 PDQ code: https://github.com/jskinn/rvchallenge-evaluation/blob/master/pdq.py
    """
    def __init__(self, filter_gts=False, segment_mode=False, greedy_mode=False):
        """
        Initialisation function for PDQ evaluator.
        :param filter_gts: boolean describing if output should be filtered by ground-truth size (used for rvc1 only)
        (default False)
        :param segment_mode: boolean describing if gt_objects will be evaluated using only their segmentation masks
        i.e. not discounting pixels within GT bounding box that are part of the background. (default False)
        :param greedy_mode: Boolean flag for if PDQ should utilise greedy assignment strategy rather than optimal.
        Can lead to sped-up evaluation time but differs from official utilisation of PDQ. (default False)
        """
        super(PDQ, self).__init__()
        self.greedy_mode = greedy_mode
        self.segment_mode = segment_mode
        self.filter_gts = filter_gts
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fg_quality = 0.0
        self._tot_bg_quality = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._tot_fp_cost = 0.0
        self._det_evals = []
        self._gt_evals = []

    def add_img_eval(self, gt_instances, det_instances):
        """
        Adds a single image's detections and ground-truth to the overall evaluation analysis.
        :param gt_instances: list of GroundTruthInstance objects present in the given image.
        :param det_instances: list of DetectionInstance objects provided for the given image
        :return: None
        """
        results = _calc_qual_img(gt_instances, det_instances)
        self._tot_overall_quality += results['overall']
        self._tot_spatial_quality += results['spatial']
        self._tot_label_quality += results['label']
        self._tot_fg_quality += results['fg']
        self._tot_bg_quality += results['bg']
        self._tot_TP += results['TP']
        self._tot_FP += results['FP']
        self._tot_FN += results['FN']
        self._tot_fp_cost += results['fp_cost']
        self._det_evals.append(results['img_det_evals'])
        self._gt_evals.append(results['img_gt_evals'])

    def get_pdq_score(self):
        """
        Get the current PDQ score for all frames analysed at the current time.
        :return: The average PDQ across all images as a float.
        """
        denominator = self._tot_TP + self._tot_FN + self._tot_fp_cost
        return self._tot_overall_quality/denominator

    def reset(self):
        """
        Reset all internally stored evaluation measures to zero.
        :return: None
        """
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fg_quality = 0.0
        self._tot_bg_quality = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._tot_fp_cost = 0.0
        self._det_evals = []
        self._gt_evals = []

    def score(self, pdq_param_lists):
        """
        Calculates the average probabilistic detection quality for a set of detections on
        a set of ground truth objects over a series of images.
        The average is calculated as the average pairwise quality over the number of object-detection pairs observed.
        Note that this removes any evaluation information that had been stored for previous images.
        Assumes you want to score just the full list you are given.
        :param pdq_param_lists: A list of tuples where each tuple holds a list of GroundTruthInstances and a list of
        DetectionInstances. Each image observed is an entry in the main list.
        :return: The average PDQ across all images as a float.
        """
        self.reset()

        pool = Pool(processes=cpu_count())

        num_imgs = len(pdq_param_lists)
        for img_results in tqdm(pool.imap(self._get_image_evals, pdq_param_lists),
                                total=num_imgs, desc='PDQ Images'):
            self._tot_overall_quality += img_results['overall']
            self._tot_spatial_quality += img_results['spatial']
            self._tot_label_quality += img_results['label']
            self._tot_fg_quality += img_results['fg']
            self._tot_bg_quality += img_results['bg']
            self._tot_TP += img_results['TP']
            self._tot_FP += img_results['FP']
            self._tot_FN += img_results['FN']
            self._tot_fp_cost += img_results['fp_cost']
            self._det_evals.append(img_results['img_det_evals'])
            self._gt_evals.append(img_results['img_gt_evals'])

        pool.close()
        pool.join()

        return self.get_pdq_score()

    def get_avg_spatial_score(self):
        """
        Get the average spatial quality score for all assigned detections in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_spatial_quality / float(self._tot_TP)
        return 0.0

    def get_avg_label_score(self):
        """
        Get the average label quality score for all assigned detections in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average label quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_label_quality / float(self._tot_TP)
        return 0.0

    def get_avg_fp_label_score(self):
        if self._tot_FP > 0:
            return (self._tot_FP - self._tot_fp_cost) / self._tot_FP
        return 0.0

    def get_avg_overall_quality_score(self):
        """
        Get the average overall pairwise quality score for all assigned detections
        in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average overall pairwise quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_overall_quality / float(self._tot_TP)
        return 0.0

    def get_avg_fg_quality_score(self):
        """
        Get the average foreground spatial quality score for all assigned detections
        in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average overall pairwise foreground spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_fg_quality / float(self._tot_TP)
        return 0.0

    def get_avg_bg_quality_score(self):
        """
        Get the average background spatial quality score for all assigned detections
        in all frames analysed at the current time.
        Note that this is averaged over the number of assigned detections (TPs) and not the full set of TPs, FPs,
        and FNs like the final PDQ score.
        :return: average overall pairwise background spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_bg_quality / float(self._tot_TP)
        return 0.0

    def get_assignment_counts(self):
        """
        Get the total number of TPs, FPs, and FNs across all frames analysed at the current time.
        :return: tuple containing (TP, FP, FN)
        """
        return self._tot_TP, self._tot_FP, self._tot_FN

    def _get_image_evals(self, parameters):
        """
        Evaluate the results for a given image
        :param parameters: tuple containing list of GroundTruthInstances and DetectionInstances
        :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
        detections, total label quality on positively assigned detections, total foreground spatial quality on positively
        assigned detections, total background spatial quality on positively assigned detections, number of true positives,
        number of false positives, number false negatives, detection evaluation summary, and ground-truth evaluation summary
        for the given image.
        Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
        'fg':<tot_tp_foreground_quality>, 'bg':<tot_tp_background_quality>, 'TP': <num_true_positives>,
        'FP': <num_false_positives>, 'FN': <num_false_positives>, 'img_det_evals':<detection_evaluation_summary>,
        'img_gt_evals':<ground-truth_evaluation_summary>}
        """
        gt_instances, det_instances = parameters
        results = _calc_qual_img(gt_instances, det_instances, self.filter_gts, self.segment_mode, self.greedy_mode)
        return results


def _vectorize_img_gts(gt_instances, img_shape, segment_mode):
    """
    Vectorizes the required elements for all GroundTruthInstances as necessary for a given image.
    These elements are the segmentation mask, background mask, number of foreground pixels, and label for each.
    :param gt_instances: list of all GroundTruthInstances for a given image
    :param img_shape: shape of the image that the GroundTruthInstances lie within
    :param segment_mode: boolean describing if we are in segment mode or not. If so, then the background region is
    outside the ground-truth segmentation mask and if not, it is the region outside the ground-truth bounding box.
    :return: (gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec).
    gt_seg_mat: h x w x g boolean numpy array depicting the ground truth pixels for each of the g GroundTruthInstances
    within an h x w image.
    bg_seg_mat: h x w x g boolean numpy array depicting the background pixels for each of the g GroundTruthInstances
    (pixels outside the segmentation mask or bounding box depending on mode) within an h x w image.
    num_fg_pixels_vec: g x 1 int numpy array containing the number of foreground (object) pixels for each of
    the g GroundTruthInstances.
    gt_label_vec: g, numpy array containing the class label as an integer for each of the g GroundTruthInstances
    """
    gt_seg_mat = np.stack([gt_instance.segmentation_mask for gt_instance in gt_instances], axis=2)   # h x w x g
    num_fg_pixels_vec = np.array([[gt_instance.num_pixels] for gt_instance in gt_instances], dtype=np.int)  # g x 1
    gt_label_vec = np.array([gt_instance.class_label for gt_instance in gt_instances], dtype=np.int)        # g,

    if segment_mode:
        bg_seg_mat = np.logical_not(gt_seg_mat)
    else:
        bg_seg_mat = np.ones(img_shape + (len(gt_instances),), dtype=np.bool)  # h x w x g
        for gt_idx, gt_instance in enumerate(gt_instances):
            gt_box = gt_instance.bounding_box
            bg_seg_mat[gt_box[1]:gt_box[3]+1, gt_box[0]:gt_box[2]+1, gt_idx] = False

    return gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec


def _vectorize_img_dets(det_instances, img_shape):
    """
    Vectorize the required elements for all DetectionInstances as necessary for a given image.
    These elements are the thresholded detection heatmap, and the detection label list for each.
    :param det_instances: list of all DetectionInstances for a given image.
    :param img_shape: shape of the image that the DetectionInstances lie within.
    :return: (det_seg_heatmap_mat, det_label_prob_mat)
    det_seg_heatmap_mat: h x w x d float32 numpy array depciting the probability that each pixel is part of the
    detection within an h x w image. Note that this is thresholded so pixels with particularly low probabilities instead
    have a probability in the heatmap of zero.
    det_label_prob_mat: d x c numpy array of label probability scores across all c classes for each of the d detections
    """
    det_label_prob_mat = np.stack([det_instance.class_list for det_instance in det_instances], axis=0)  # d x c
    det_seg_heatmap_mat = np.stack([det_instance.calc_heatmap(img_shape) for det_instance in det_instances], axis=2)
    return det_seg_heatmap_mat, det_label_prob_mat


def _calc_bg_loss(bg_seg_mat, det_seg_heatmap_mat):
    """
    Calculate the background pixel loss for all detections on all ground truth objects for a given image.
    :param bg_seg_mat: h x w x g vectorized background masks for each ground truth object in the image.
    :param det_seg_heatmap_mat: h x w x d vectorized segmented heatmaps for each detection in the image.
    :return: (bg_loss_sum, num_bg_pixels_mat)
    bg_loss_sum: g x d total background loss between each of the g ground truth objects and d detections.
    num_bg_pixels_mat: g x d number of background pixels examined for each combination of g ground truth objects and d
    detections.
    """
    bg_log_loss_mat = _safe_log(1-det_seg_heatmap_mat) * (det_seg_heatmap_mat > 0)
    bg_loss_sum = np.tensordot(bg_seg_mat, bg_log_loss_mat, axes=([0, 1], [0, 1]))  # g x d
    return bg_loss_sum


def _calc_fg_loss(gt_seg_mat, det_seg_heatmap_mat):
    """
    Calculate the foreground pixel loss for all detections on all ground truth objects for a given image.
    :param gt_seg_mat: h x w x g vectorized segmentation masks for each ground truth object in the image.
    :param det_seg_heatmap_mat: h x w x d vectorized segmented heatmaps for each detection in the image.
    :return: fg_loss_sum: g x d total foreground loss between each of the g ground truth objects and d detections.
    """
    log_heatmap_mat = _safe_log(det_seg_heatmap_mat)
    fg_loss_sum = np.tensordot(gt_seg_mat, log_heatmap_mat, axes=([0, 1], [0, 1]))  # g x d
    return fg_loss_sum


def _safe_log(mat):
    """
    Function for performing safe log (avoiding infinite loss) for all elements of a given matrix by adding _SMALL_VAL
    to all elements.
    :param mat: matrix of values
    :return: safe log of matrix elements
    """
    return np.log(mat + _SMALL_VAL)


def _calc_spatial_qual(fg_loss_sum, bg_loss_sum, num_fg_pixels_vec):
    """
    Calculate the spatial quality for all detections on all ground truth objects for a given image.
    :param fg_loss_sum: g x d total foreground loss between each of the g ground truth objects and d detections.
    :param bg_loss_sum: g x d total background loss between each of the g ground truth objects and d detections.
    :param num_fg_pixels_vec: g x 1 number of pixels for each of the g ground truth objects.
    :return: spatial_quality: g x d spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    total_loss = fg_loss_sum + bg_loss_sum

    loss_per_gt_pixel = total_loss/num_fg_pixels_vec

    spatial_quality = np.exp(loss_per_gt_pixel)

    # Deal with tiny floating point errors or tiny errors caused by _SMALL_VAL that prevent perfect 0 or 1 scores
    spatial_quality[np.isclose(spatial_quality, 0)] = 0
    spatial_quality[np.isclose(spatial_quality, 1)] = 1

    return spatial_quality


def _calc_label_qual(gt_label_vec, det_label_prob_mat):
    """
    Calculate the label quality for all detections on all ground truth objects for a given image.
    :param gt_label_vec:  g, numpy array containing the class label as an integer for each object.
    :param det_label_prob_mat: d x c numpy array of label probability scores across all c classes
    for each of the d detections.
    :return: label_qual_mat: g x d label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    label_qual_mat = det_label_prob_mat[:, gt_label_vec].T.astype(np.float32)     # g x d
    return label_qual_mat


def _calc_overall_qual(label_qual, spatial_qual):
    """
    Calculate the overall quality for all detections on all ground truth objects for a given image
    :param label_qual: g x d label quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :param spatial_qual: g x d spatial quality score between zero and one for each possible combination of
    g ground truth objects and d detections.
    :return: overall_qual_mat: g x d overall label quality between zero and one for each possible combination of
    g ground truth objects and d detections.
    """
    combined_mat = np.dstack((label_qual, spatial_qual))

    # Calculate the geometric mean between label quality and spatial quality.
    # Note we ignore divide by zero warnings here for log(0) calculations internally.
    with np.errstate(divide='ignore'):
        overall_qual_mat = gmean(combined_mat, axis=2)

    return overall_qual_mat


def _gen_cost_tables(gt_instances, det_instances, segment_mode):
    """
    Generate the cost tables containing the cost values (1 - quality) for each combination of ground truth objects and
    detections within a given image.
    :param gt_instances: list of all GroundTruthInstances for a given image.
    :param det_instances: list of all DetectionInstances for a given image.
    :return: dictionary of g x d cost tables for each combination of ground truth objects and detections.
    Note that all costs are simply 1 - quality scores (required for Hungarian algorithm implementation)
    Format: {'overall': overall pPDQ cost table, 'spatial': spatial quality cost table,
    'label': label quality cost table, 'fg': foreground quality cost table, 'bg': background quality cost table}
    """
    # Initialise cost tables
    n_pairs = max(len(gt_instances), len(det_instances))
    overall_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    spatial_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    label_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    bg_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    fg_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    img_shape = gt_instances[0].segmentation_mask.shape

    # Generate all the matrices needed for calculations
    gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec = _vectorize_img_gts(gt_instances, img_shape, segment_mode)
    img_shape = gt_instances[0].segmentation_mask.shape
    det_seg_heatmap_mat, det_label_prob_mat = _vectorize_img_dets(det_instances, img_shape)

    # Calculate spatial and label qualities
    label_qual_mat = _calc_label_qual(gt_label_vec, det_label_prob_mat)
    fg_loss = _calc_fg_loss(gt_seg_mat, det_seg_heatmap_mat)
    bg_loss = _calc_bg_loss(bg_seg_mat, det_seg_heatmap_mat)
    spatial_qual = _calc_spatial_qual(fg_loss, bg_loss, num_fg_pixels_vec)

    # Calculate foreground quality
    fg_loss_per_gt_pixel = fg_loss/num_fg_pixels_vec
    fg_qual = np.exp(fg_loss_per_gt_pixel)
    fg_qual[np.isclose(fg_qual, 0)] = 0
    fg_qual[np.isclose(fg_qual, 1)] = 1

    # Calculate background quality
    bg_loss_per_gt_pixel = bg_loss/num_fg_pixels_vec
    bg_qual = np.exp(bg_loss_per_gt_pixel)
    bg_qual[np.isclose(bg_qual, 0)] = 0
    bg_qual[np.isclose(bg_qual, 1)] = 1

    # Generate the overall cost table (1 - overall quality)
    overall_cost_table[:len(gt_instances), :len(det_instances)] -= _calc_overall_qual(label_qual_mat,
                                                                                      spatial_qual)

    # Generate the spatial and label cost tables
    spatial_cost_table[:len(gt_instances), :len(det_instances)] -= spatial_qual
    label_cost_table[:len(gt_instances), :len(det_instances)] -= label_qual_mat

    # Generate foreground and background cost tables
    fg_cost_table[:len(gt_instances), :len(det_instances)] -= fg_qual
    bg_cost_table[:len(gt_instances), :len(det_instances)] -= bg_qual

    return {'overall': overall_cost_table, 'spatial': spatial_cost_table, 'label': label_cost_table,
            'fg': fg_cost_table, 'bg': bg_cost_table}


def _calc_qual_img(gt_instances, det_instances, filter_gt, segment_mode, greedy_mode):
    """
    Calculates the sum of qualities for the best matches between ground truth objects and detections for an image.
    Each ground truth object can only be matched to a single detection and vice versa as an object-detection pair.
    Note that if a ground truth object or detection does not have a match, the quality is counted as zero.
    This represents a theoretical object-detection pair with the object or detection and a counterpart which
    does not describe it at all.
    Any provided detection with a zero-quality match will be counted as a false positive (FP).
    Any ground-truth object with a zero-quality match will be counted as a false negative (FN).
    All other matches are counted as "true positives" (TP)
    If there are no ground truth objects or detections for the image, the system returns zero and this image
    will not contribute to average_PDQ.
    :param gt_instances: list of GroundTruthInstance objects describing the ground truth objects in the current image.
    :param det_instances: list of DetectionInstance objects describing the detections for the current image.
    :param filter_gt: boolean depicting if _is_gt_included should filter gt objects based on their size
    :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
    detections, total label quality on positively assigned detections, total forerground quality on positively assigned
    detections, total background quality on positively assigned detections, number of true positives,
    number of false positives, number false negatives, detection evaluation summary,
    and ground-truth evaluation summary for for the given image.
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'fg':<tot_tp_foreground_quality>, 'bg':<tot_tp_background_quality>, 'TP': <num_true_positives>,
    'FP': <num_false_positives>, 'FN': <num_false_positives>, 'img_det_evals':<detection_evaluation_summary>,
    'img_gt_evals':<ground-truth_evaluation_summary>}
    """
    # Record the full evaluation details for every match
    img_det_evals = []
    img_gt_evals = []
    tot_fp_cost = 0

    # if there are no detections or gt instances respectively the quality is zero
    if len(gt_instances) == 0 or len(det_instances) == 0:
        if len(det_instances) > 0:
            img_det_evals = [{"det_id": idx, "gt_id": None, "ignore": False, "matched": False,
                              "pPDQ": 0.0, "spatial": 0.0, "label": 0.0, "correct_class": None,
                              'bg': 0.0, 'fg': 0.0}
                             for idx in range(len(det_instances))]
            tot_fp_cost = np.sum([np.max(det_instance.class_list) for det_instance in det_instances])

        # Filter out GT instances which are to be ignored because they are too small
        FN = 0
        if len(gt_instances) > 0:
            for gt_idx, gt_instance in enumerate(gt_instances):
                gt_eval_dict = {"det_id": None, "gt_id": gt_idx, "ignore": False, "matched": False,
                                "pPDQ": 0.0, "spatial": 0.0, "label": 0.0, "correct_class": gt_instance.class_label,
                                'fg': 0.0, 'bg': 0.0}
                if _is_gt_included(gt_instance, filter_gt):
                    FN += 1
                else:
                    gt_eval_dict["ignore"] = True
                img_gt_evals.append(gt_eval_dict)

        return {'overall': 0.0, 'spatial': 0.0, 'label': 0.0, 'fg': 0.0, 'bg': 0.0, 'TP': 0, 'FP': len(det_instances),
                'FN': FN, "img_det_evals": img_det_evals, "img_gt_evals": img_gt_evals, 'fp_cost': tot_fp_cost}

    # For each possible pairing, calculate the quality of that pairing and convert it to a cost
    # to enable use of the Hungarian algorithm.
    cost_tables = _gen_cost_tables(gt_instances, det_instances, segment_mode)

    # Use the Hungarian algorithm with the cost table to find the best match between ground truth
    # object and detection (lowest overall cost representing highest overall pairwise quality)
    if greedy_mode:
        row_idxs, col_idxs = _assign_greedy(cost_tables['overall'])
    else:
        row_idxs, col_idxs = linear_sum_assignment(cost_tables['overall'])

    # Transform the loss tables back into quality tables with values between 0 and 1
    overall_quality_table = 1 - cost_tables['overall']
    spatial_quality_table = 1 - cost_tables['spatial']
    label_quality_table = 1 - cost_tables['label']
    fg_quality_table = 1 - cost_tables['fg']
    bg_quality_table = 1 - cost_tables['bg']

    # Go through all optimal assignments and summarize all pairwise statistics
    # Calculate the number of TPs, FPs, and FNs for the image during the process
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    false_positive_idxs = []

    for match_idx, match in enumerate(zip(row_idxs, col_idxs)):
        row_id, col_id = match
        det_eval_dict = {"det_id": int(col_id), "gt_id": int(row_id), "matched": True, "ignore": False,
                         "pPDQ": float(overall_quality_table[row_id, col_id]),
                         "spatial": float(spatial_quality_table[row_id, col_id]),
                         "label": float(label_quality_table[row_id, col_id]),
                         'fg': float(fg_quality_table[row_id, col_id]),
                         'bg': float(bg_quality_table[row_id, col_id]),
                         "correct_class": None}
        gt_eval_dict = det_eval_dict.copy()
        if overall_quality_table[row_id, col_id] > 0:
            det_eval_dict["correct_class"] = gt_instances[row_id].class_label
            gt_eval_dict["correct_class"] = gt_instances[row_id].class_label
            if row_id < len(gt_instances) and _is_gt_included(gt_instances[row_id], filter_gt):
                true_positives += 1
            else:
                # ignore detections on samples which are too small to be considered a valid object
                det_eval_dict["ignore"] = True
                gt_eval_dict["ignore"] = True
                # Set the overall quality table value to zero so it does not get included in final total
                overall_quality_table[row_id, col_id] = 0.0
            img_det_evals.append(det_eval_dict)
            img_gt_evals.append(gt_eval_dict)
        else:
            if row_id < len(gt_instances):
                gt_eval_dict["correct_class"] = gt_instances[row_id].class_label
                gt_eval_dict["det_id"] = None
                gt_eval_dict["matched"] = False
                if _is_gt_included(gt_instances[row_id], filter_gt):
                    false_negatives += 1
                else:
                    gt_eval_dict["ignore"] = True
                img_gt_evals.append(gt_eval_dict)

            if col_id < len(det_instances):
                det_eval_dict["gt_id"] = None
                det_eval_dict["matched"] = False
                false_positives += 1
                false_positive_idxs.append(col_id)
                img_det_evals.append(det_eval_dict)

    # Calculate the sum of quality at the best matching pairs to calculate total qualities for the image
    tot_overall_img_quality = np.sum(overall_quality_table[row_idxs, col_idxs])

    # Force spatial and label qualities to zero for total calculations as there is no actual association between
    # detections and therefore no TP when this is the case.
    spatial_quality_table[overall_quality_table == 0] = 0.0
    label_quality_table[overall_quality_table == 0] = 0.0
    fg_quality_table[overall_quality_table == 0] = 0.0
    bg_quality_table[overall_quality_table == 0] = 0.0

    # Calculate the sum of spatial and label qualities only for TP samples
    tot_tp_spatial_quality = np.sum(spatial_quality_table[row_idxs, col_idxs])
    tot_tp_label_quality = np.sum(label_quality_table[row_idxs, col_idxs])
    tot_tp_fg_quality = np.sum(fg_quality_table[row_idxs, col_idxs])
    tot_tp_bg_quality = np.sum(bg_quality_table[row_idxs, col_idxs])

    # Calculate the penalty for assigning a high label probability to false positives
    tot_fp_cost = np.sum([np.max(det_instances[i].class_list) for i in false_positive_idxs])

    # Sort the evaluation details to match the order of the detections and ground truths
    img_det_eval_idxs = [det_eval_dict["det_id"] for det_eval_dict in img_det_evals]
    img_gt_eval_idxs = [gt_eval_dict["gt_id"] for gt_eval_dict in img_gt_evals]
    img_det_evals = [img_det_evals[idx] for idx in np.argsort(img_det_eval_idxs)]
    img_gt_evals = [img_gt_evals[idx] for idx in np.argsort(img_gt_eval_idxs)]

    return {'overall': tot_overall_img_quality, 'spatial': tot_tp_spatial_quality, 'label': tot_tp_label_quality,
            'fg': tot_tp_fg_quality, 'bg': tot_tp_bg_quality,
            'TP': true_positives, 'FP': false_positives, 'FN': false_negatives,
            'img_gt_evals': img_gt_evals, 'img_det_evals': img_det_evals, 'fp_cost':tot_fp_cost}


def _is_gt_included(gt_instance, filter_gt):
    """
    Determines if a ground-truth instance is large enough to be considered valid for detection
    :param gt_instance: GroundTruthInstance object being evaluated
    :param filter_gt: parameter depicting if gts should be filtered at all
    :return: Boolean describing if the object is valid for detection
    """
    if not filter_gt:
        return True
    return (gt_instance.bounding_box[2] - gt_instance.bounding_box[0] > 10) and \
           (gt_instance.bounding_box[3] - gt_instance.bounding_box[1] > 10) and \
           np.count_nonzero(gt_instance.segmentation_mask) > 100


def _assign_greedy(cost_mat):
    """
    Assign detections to ground truths in a greedy fashion (highest pPDQ scores assigned)
    :param cost_mat: Costs matrix (ground-truths x detections) square matrix with zeros padding
    :return: row_idxs, col_idxs for assignments
    """
    if cost_mat.shape[0] != cost_mat.shape[1]:
        print("ERROR! Cost matrix must be square")
        return [], []
    match_order = np.argsort(cost_mat.flatten())
    rows = []   # gts
    cols = []   # dets
    n_assign = cost_mat.shape[0]
    for match_idx in match_order:
        row_idx = match_idx // n_assign
        col_idx = match_idx % n_assign

        if row_idx not in rows and col_idx not in cols:
            rows.append(row_idx)
            cols.append(col_idx)
            # Break out of the loop if all assignments have been made
            if len(rows) == n_assign:
                break

    if len(rows) != n_assign or len(cols) != n_assign:
        print("ERROR! This should not happen")
        return [], []

    return rows, cols
