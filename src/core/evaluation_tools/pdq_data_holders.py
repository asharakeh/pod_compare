from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce  # forward compatibility for Python 3
import operator
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

_HEATMAP_THRESH = 0.0027
_2D_MAH_DIST_THRESH = 3.439
_SMALL_VAL = 1e-14


class GroundTruthInstance(object):
    def __init__(
            self,
            segmentation_mask,
            true_class_label,
            image_id,
            instance_id,
            bounding_box=None,
            num_pixels=None):
        self.segmentation_mask = segmentation_mask
        self.class_label = true_class_label
        self.image_id = image_id
        self.instance_id = instance_id

        if bounding_box is not None and len(bounding_box) > 0:
            self.bounding_box = bounding_box
        else:
            self.bounding_box = generate_bounding_box_from_mask(
                segmentation_mask)
        if num_pixels is not None and num_pixels > 0:
            self.num_pixels = num_pixels
        else:
            self.num_pixels = np.count_nonzero(segmentation_mask)
        # Calculate the number of pixels in the ground-truth bounding box
        # (length x height)
        self.num_bbox_pixels = (
            self.bounding_box[2] + 1 - self.bounding_box[0]) * (
            self.bounding_box[3] + 1 - self.bounding_box[1])


class DetectionInstance(object):
    def __init__(self, class_list, heatmap=None):
        self._heatmap = heatmap
        self.class_list = class_list

    def calc_heatmap(self, img_size):
        return self._heatmap

    def get_max_class(self):
        return np.argmax(self.class_list)

    def get_max_score(self):
        return np.amax(self.class_list)


class BBoxDetInst(DetectionInstance):
    def __init__(self, class_list, box, pos_prob=1.0):
        """
        Initialisation function for a BBox detection instance
        :param class_list: list of class probabilities for the detection
        :param box: list of box corners that contain the object detected (inclusively).
        Formatted [x1, y1, x2, y2]
        :param pos_prob: float depicting the positional confidence
        """
        super(BBoxDetInst, self).__init__(class_list)
        self.box = box
        self.pos_prob = pos_prob

    def calc_heatmap(self, img_size):
        heatmap = np.zeros(img_size, dtype=np.float32)
        x1, y1, x2, y2 = self.box
        x1_c, y1_c = np.ceil(self.box[0:2]).astype(np.int)
        x2_f, y2_f = np.floor(self.box[2:]).astype(np.int)
        # Even if the coordinates are integers, there should always be a range
        x1_f = x1_c - 1
        y1_f = y1_c - 1
        x2_c = x2_f + 1
        y2_c = y2_f + 1

        heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]),
                max(x1_f, 0):min(x2_c + 1, img_size[1])] = self.pos_prob
        if y1_f >= 0:
            heatmap[y1_f, max(x1_f, 0):min(x2_c + 1, img_size[1])] *= y1_c - y1
        if y2_c < img_size[0]:
            heatmap[y2_c, max(x1_f, 0):min(x2_c + 1, img_size[1])] *= y2 - y2_f
        if x1_f >= 0:
            heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]), x1_f] *= x1_c - x1
        if x2_c < img_size[1]:
            heatmap[max(y1_f, 0):min(y2_c + 1, img_size[0]), x2_c] *= x2 - x2_f
        return heatmap


class PBoxDetInst(DetectionInstance):
    def __init__(self, class_list, box, covs):
        """
        Initialisation function for a PBox detection instance
        :param class_list: list of class probabilities for the detection
        :param box: list of BBox corners, used to define the Gaussian corner mean locations for the box.
        Formatted [x1, y1, x2, y2]
        :param covs: list of two 2D covariance matrices used to define the covariances of the Gaussian corners.
        Formatted [cov1, cov2] where cov1 and cov2 are formatted [[var_x, corr], [corr, var_y]]
        """
        super(PBoxDetInst, self).__init__(class_list)
        self.box = box
        self.covs = covs

    def calc_heatmap(self, img_size):

        # get all covs in format (y,x) to match matrix ordering
        covs2 = [np.flipud(np.fliplr(cov)) for cov in self.covs]

        prob1 = gen_single_heatmap(
            img_size, [self.box[1], self.box[0]], covs2[0])
        prob2 = gen_single_heatmap(img_size,
                                   [img_size[0] - (self.box[3] + 1),
                                    img_size[1] - (self.box[2] + 1)],
                                   np.array(covs2[1]).T)
        # flip left-right and up-down to provide probability in from
        # bottom-right corner
        prob2 = np.fliplr(np.flipud(prob2))

        # generate final heatmap
        heatmap = prob1 * prob2

        # Hack to enforce that there are no pixels with probs greater than 1
        # due to floating point errors
        heatmap[heatmap > 1] = 1

        heatmap[heatmap < _HEATMAP_THRESH] = 0

        return heatmap


def find_roi(img_size, mean, cov):
    """
    Function for finding the region of interest for a probability heatmap generated by a Gaussian corner.
    This region of interest is the area with most change therein, with probabilities above 0.0027 and below 0.9973
    :param img_size: tuple: formatted (n_rows, n_cols) depicting the size of the image
    :param mean: list: formatted [mu_y, mu_x] describes the location of the mean of the Gaussian corner.
    :param cov: 2D array: formatted [[var_y, corr], [corr, var_x]] describes the covariance of the Gaussian corner.
    :return: roi_box formatted [x1, y1, x2, y2] depicting the corners of the region of interest (inclusive)
    """

    # Calculate approximate ROI
    stdy = cov[0, 0] ** 0.5
    stdx = cov[1, 1] ** 0.5

    minx = int(max(mean[1] - stdx * 5, 0))
    miny = int(max(mean[0] - stdy * 5, 0))
    maxx = int(min(mean[1] + stdx * 5, img_size[1] - 1))
    maxy = int(min(mean[0] + stdy * 5, img_size[0] - 1))

    # If the covariance is singular, we can't do any better in our estimate.
    if np.abs(np.linalg.det(cov)) < 1e-8:
        return minx, miny, max(0, maxx), max(0, maxy)

    # produce list of positions [y,x] to compare to the given mean location
    approx_roi_shape = (max(maxy + 1 - miny, 1), max(maxx + 1 - minx, 1))
    positions = np.indices(approx_roi_shape).T.reshape(-1, 2)
    positions[:, 0] += miny
    positions[:, 1] += minx
    # Calculate the mahalanobis distances to those locations (number of standard deviations)
    # Can only do this for non-singular matrices
    mdists = cdist(
        positions,
        np.array(
            [mean]),
        metric='mahalanobis',
        VI=np.linalg.inv(cov))
    mdists = mdists.reshape(approx_roi_shape[1], approx_roi_shape[0]).T

    # Shift around the mean to change which corner of the pixel we're using
    # for the mahalanobis distance
    dist_meany = max(min(int(mean[0] - miny), img_size[0] - 1), 0)
    dist_meanx = max(min(int(mean[1] - minx), img_size[1] - 1), 0)
    if 0 < dist_meany < img_size[0] - 1:
        mdists[:dist_meany, :] = mdists[1:dist_meany + 1, :]
    if 0 < dist_meanx < img_size[1] - 1:
        mdists[:, :dist_meanx] = mdists[:, 1:dist_meanx + 1]

    # Mask out samples that are outside the desired distance (extremely low
    # probability points)
    mask = mdists <= _2D_MAH_DIST_THRESH
    # Force the pixel containing the mean to be true, we always care about that
    mask[dist_meany, dist_meanx] = True
    roi_box = generate_bounding_box_from_mask(mask)

    roi_box_out = [roi_box[0] + minx, roi_box[1] +
                   miny, roi_box[2] + minx, roi_box[3] + miny]

    roi_box_out = [max(0, roi_dim) for roi_dim in roi_box_out]

    return roi_box_out


def gen_single_heatmap(img_size, mean, cov):
    """
    Function for generating the heatmap for a given Gaussian corner.
    :param img_size: tuple: formatted (n_rows, n_cols) depicting the size of the image
    :param mean: list: formatted [mu_y, mu_x] describes the location of the mean of the Gaussian corner.
    :param cov: 2D array: formatted [[var_y, corr], [corr, var_x]] describes the covariance of the Gaussian corner.
    :return: heatmap image of size <img_size> with spatial probabilities between 0 and 1.
    """
    heatmap = np.zeros(img_size, dtype=np.float32)
    g = multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    roi_box = find_roi(img_size, mean, cov)
    # Note that we subtract small value to avoid fencepost issues with
    # extremely low covariances.
    positions = np.dstack(
        np.mgrid[roi_box[1] + 1:roi_box[3] + 2, roi_box[0] + 1:roi_box[2] + 2]) - _SMALL_VAL

    prob = g.cdf(positions)

    if len(prob.shape) == 1:
        prob.shape = (roi_box[3] + 1 - roi_box[1], roi_box[2] + 1 - roi_box[0])

    heatmap[roi_box[1]:roi_box[3] + 1, roi_box[0]:roi_box[2] + 1] = prob
    heatmap[roi_box[3]:, roi_box[0]:roi_box[2] +
            1] = np.array(heatmap[roi_box[3], roi_box[0]:roi_box[2] + 1], ndmin=2)
    heatmap[roi_box[1]:roi_box[3] +
            1, roi_box[2]:] = np.array(heatmap[roi_box[1]:roi_box[3] +
                                               1, roi_box[2]], ndmin=2).T
    heatmap[roi_box[3] + 1:, roi_box[2] + 1:] = 1.0

    # If your region of interest includes outside the main image, remove probability of existing outside the image
    # Remove probability of being outside in the x direction
    if roi_box[0] == 0:
        # points left of the image
        pos_outside_x = np.dstack(
            np.mgrid[roi_box[1] + 1:roi_box[3] + 2, 0:1]) - _SMALL_VAL
        prob_outside_x = np.zeros((img_size[0], 1), dtype=np.float32)
        prob_outside_x[roi_box[1]:roi_box[3] + 1, 0] = g.cdf(pos_outside_x)
        prob_outside_x[roi_box[3] + 1:, 0] = prob_outside_x[roi_box[3], 0]
        # Final probability is your overall cdf minus the probability in-line with that point along
        # the border for both dimensions plus the cdf at (-1, -1) which has
        # points counted twice otherwise
        heatmap -= prob_outside_x

    # Remove probability of being outside in the x direction
    if roi_box[1] == 0:
        # points above the image
        pos_outside_y = np.dstack(
            np.mgrid[0:1, roi_box[0] + 1:roi_box[2] + 2]) - _SMALL_VAL
        prob_outside_y = np.zeros((1, img_size[1]), dtype=np.float32)
        prob_outside_y[0, roi_box[0]:roi_box[2] + 1] = g.cdf(pos_outside_y)
        prob_outside_y[0, roi_box[2] + 1:] = prob_outside_y[0, roi_box[2]]
        heatmap -= prob_outside_y

    # If we've subtracted twice, we need to re-add the probability of the far
    # corner
    if roi_box[0] == 0 and roi_box[1] == 0:
        heatmap += g.cdf([[[0 - _SMALL_VAL, 0 - _SMALL_VAL]]])

    heatmap[heatmap < _HEATMAP_THRESH] = 0

    return heatmap


def generate_bounding_box_from_mask(mask):
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        raise ValueError(
            "No positive pixels found, cannot compute bounding box")
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return [xmin, ymin, xmax, ymax]


def bb_iou(boxA, boxB):
    # Code from online source
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    # determine the (x, y)-coordinates of the intersection rectangle
    # Assumes boxes input in format [[xmin,ymin],[xmax,ymax]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, (xB - xA) + 1) * max(0, (yB - yA) + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = ((boxA[2] - boxA[0]) + 1) * ((boxA[3] - boxA[1]) + 1)
    boxBArea = ((boxB[2] - boxB[0]) + 1) * ((boxB[3] - boxB[1]) + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def put_image_in_centre_of_other_image(img1, img2, add=False):
    """
    Code to add a mask to the centre of a given image (both 2D at the moment)
    :param img1: image to be putting other image in the centre of
    :param img2: image to put in the centre of another image
    :param add: flag as to whether to add the image to the centre of the other one or replace all pixels
    :return: Final image with img2 now in the centre of img1
    """

    borderx_size = (img1.shape[1] - img2.shape[1]) / 2
    bordery_size = (img1.shape[0] - img2.shape[0]) / 2

    x1 = int(borderx_size)
    x2 = int(borderx_size + img2.shape[1])
    y1 = int(bordery_size)
    y2 = int(bordery_size + img2.shape[0])

    new_img = img1.copy()
    if add:
        new_img[y1:y2, x1:x2] += img2
    else:
        new_img[y1:y2, x1:x2] = img2

    return new_img


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
