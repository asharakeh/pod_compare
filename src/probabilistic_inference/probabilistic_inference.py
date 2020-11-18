import cv2
import numpy as np
import os
import torch

from abc import ABC, abstractmethod

# Detectron Imports
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer

# Project Imports
from probabilistic_inference import inference_utils
from probabilistic_modeling.modeling_utils import covariance_output_to_cholesky


def build_predictor(cfg):
    """
    Builds probabilistic predictor according to architecture in config file.
    Args:
        cfg (CfgNode): detectron2 configuration node.

    Returns:
        Instance of the correct predictor.
    """
    if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
        return RetinaNetProbabilisticPredictor(cfg)
    else:
        raise ValueError(
            'Invalid meta-architecture {}.'.format(cfg.MODEL.META_ARCHITECTURE))


class ProbabilisticPredictor(ABC):
    """
    Abstract class for probabilistic predictor.
    """

    def __init__(self, cfg):
        # Create common attributes.
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model_list = []

        # Parse config
        self.inference_mode = self.cfg.PROBABILISTIC_INFERENCE.INFERENCE_MODE
        self.mc_dropout_enabled = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE
        self.num_mc_dropout_runs = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS

        # Set model to train for MC-Dropout runs
        if self.mc_dropout_enabled:
            self.model.train()
        else:
            self.model.eval()

        # Create ensemble if applicable.
        if self.inference_mode == 'ensembles':
            ensemble_random_seeds = self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.RANDOM_SEED_NUMS

            for i, random_seed in enumerate(ensemble_random_seeds):
                model = build_model(self.cfg)
                model.eval()

                checkpoint_dir = os.path.join(
                    os.path.split(
                        self.cfg.OUTPUT_DIR)[0],
                    'random_seed_' +
                    str(random_seed))
                # Load last checkpoint.
                DetectionCheckpointer(
                    model,
                    save_dir=checkpoint_dir).resume_or_load(
                    cfg.MODEL.WEIGHTS,
                    resume=True)
                self.model_list.append(model)
        else:
            # Or Load single model last checkpoint.
            DetectionCheckpointer(
                self.model,
                save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=True)

    def __call__(self, input_im):
        # Generate detector output.
        if self.inference_mode == 'standard_nms':
            results = self.post_processing_standard_nms(input_im)
        elif self.inference_mode == 'mc_dropout_ensembles':
            results = self.post_processing_mc_dropout_ensembles(
                input_im)
        elif self.inference_mode == 'anchor_statistics':
            results = self.post_processing_anchor_statistics(
                input_im)
        elif self.inference_mode == 'ensembles':
            results = self.post_processing_ensembles(input_im, self.model_list)
        elif self.inference_mode == 'bayes_od':
            results = self.post_processing_bayes_od(input_im)
        else:
            raise ValueError(
                'Invalid inference mode {}.'.format(
                    self.inference_mode))

        # Perform post processing on detector output.
        height = input_im[0].get("height", results.image_size[0])
        width = input_im[0].get("width", results.image_size[1])
        results = inference_utils.probabilistic_detector_postprocess(results,
                                                                     height,
                                                                     width)
        return results

    def visualize_inference(self, inputs, results):
        """
        A function used to visualize final network predictions.
        It shows the original image and up to 20
        predicted object bounding boxes on the original image.

        Valuable for debugging inference methods.

        Args:
            inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        max_boxes = 20

        required_width = inputs[0]['width']
        required_height = inputs[0]['height']

        img = inputs[0]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.model.input_format == "RGB":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (required_width, required_height))

        predicted_boxes = results.pred_boxes.tensor.cpu().numpy()
        predicted_covar_mats = results.pred_boxes_covariance.cpu().numpy()

        v_pred = ProbabilisticVisualizer(img, None)
        v_pred = v_pred.overlay_covariance_instances(
            boxes=predicted_boxes[0:max_boxes], covariance_matrices=predicted_covar_mats[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_name = f"{max_boxes} Highest Scoring Results"
        cv2.imshow(vis_name, prop_img)
        cv2.waitKey()

    @abstractmethod
    def post_processing_standard_nms(self, input_im):
        pass

    @abstractmethod
    def post_processing_anchor_statistics(self, input_im):
        pass

    @abstractmethod
    def post_processing_mc_dropout_ensembles(self, input_im):
        pass

    @abstractmethod
    def post_processing_ensembles(self, input_im, model_list):
        pass

    @abstractmethod
    def post_processing_bayes_od(self, input_im):
        pass


class RetinaNetProbabilisticPredictor(ProbabilisticPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Create transform
        self.sample_box2box_transform = inference_utils.SampleBox2BoxTransform(
            self.cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

    def retinanet_probabilistic_inference(
            self,
            input_im,
            outputs=None,
            ensemble_inference=False,
            outputs_list=None):
        """
        General RetinaNet probabilistic anchor-wise inference. Preliminary inference step for many post-processing
        based inference methods such as standard_nms, anchor_statistics, and bayes_od.
        Args:
            input_im (list): an input im list generated from dataset handler.
            outputs (list): outputs from model.forward. Will be computed internally if not provided.
            ensemble_inference (bool): True if ensembles are used for inference. If set to true, outputs_list must be externally provided.
            outputs_list (list): List of model() outputs, usually generated from ensembles of models.
        Returns:
            all_predicted_boxes,
            all_predicted_boxes_covariance (Tensor): Nx4x4 vectors used
            all_predicted_prob (Tensor): Nx1 scores which represent max of all_pred_prob_vectors. For usage in NMS and mAP computation.
            all_classes_idxs (Tensor): Nx1 Class ids to be used for NMS.
            all_predicted_prob_vectors (Tensor): NxK tensor where K is the number of classes.
        """
        is_epistemic = ((self.mc_dropout_enabled and self.num_mc_dropout_runs > 1)
                        or ensemble_inference) and outputs is None
        if is_epistemic:
            if self.mc_dropout_enabled and self.num_mc_dropout_runs > 1:
                outputs_list = self.model(
                    input_im,
                    return_anchorwise_output=True,
                    num_mc_dropout_runs=self.num_mc_dropout_runs)
                n_fms = len(self.model.in_features)
                outputs_list = [{key: value[i * n_fms:(i + 1) * n_fms] if value is not None else value for key,
                                 value in outputs_list.items()} for i in range(self.num_mc_dropout_runs)]

            outputs = {'anchors': outputs_list[0]['anchors']}

            # Compute box classification and classification variance means
            box_cls = [output['box_cls'] for output in outputs_list]

            box_cls_mean = box_cls[0]
            for i in range(len(box_cls) - 1):
                box_cls_mean = [box_cls_mean[j] + box_cls[i][j]
                                for j in range(len(box_cls_mean))]
            box_cls_mean = [
                box_cls_f_map /
                len(box_cls) for box_cls_f_map in box_cls_mean]
            outputs.update({'box_cls': box_cls_mean})

            if outputs_list[0]['box_cls_var'] is not None:
                box_cls_var = [output['box_cls_var']
                               for output in outputs_list]
                box_cls_var_mean = box_cls_var[0]
                for i in range(len(box_cls_var) - 1):
                    box_cls_var_mean = [
                        box_cls_var_mean[j] +
                        box_cls_var[i][j] for j in range(
                            len(box_cls_var_mean))]
                box_cls_var_mean = [
                    box_cls_var_f_map /
                    len(box_cls_var) for box_cls_var_f_map in box_cls_var_mean]
            else:
                box_cls_var_mean = None
            outputs.update({'box_cls_var': box_cls_var_mean})

            # Compute box regression epistemic variance and mean, and aleatoric
            # variance mean
            box_delta_list = [output['box_delta']
                              for output in outputs_list]
            box_delta_mean = box_delta_list[0]
            for i in range(len(box_delta_list) - 1):
                box_delta_mean = [
                    box_delta_mean[j] +
                    box_delta_list[i][j] for j in range(
                        len(box_delta_mean))]
            box_delta_mean = [
                box_delta_f_map /
                len(box_delta_list) for box_delta_f_map in box_delta_mean]
            outputs.update({'box_delta': box_delta_mean})

            if outputs_list[0]['box_reg_var'] is not None:
                box_reg_var = [output['box_reg_var']
                               for output in outputs_list]
                box_reg_var_mean = box_reg_var[0]
                for i in range(len(box_reg_var) - 1):
                    box_reg_var_mean = [
                        box_reg_var_mean[j] +
                        box_reg_var[i][j] for j in range(
                            len(box_reg_var_mean))]
                box_reg_var_mean = [
                    box_delta_f_map /
                    len(box_reg_var) for box_delta_f_map in box_reg_var_mean]
            else:
                box_reg_var_mean = None
            outputs.update({'box_reg_var': box_reg_var_mean})

        elif outputs is None:
            outputs = self.model(input_im, return_anchorwise_output=True)

        all_anchors = []
        all_predicted_deltas = []
        all_predicted_boxes_cholesky = []
        all_predicted_prob = []
        all_classes_idxs = []
        all_predicted_prob_vectors = []
        all_predicted_boxes_epistemic_covar = []

        for i, anchors in enumerate(outputs['anchors']):
            box_cls = outputs['box_cls'][i][0]
            box_delta = outputs['box_delta'][i][0]

            # If classification aleatoric uncertainty available, perform
            # monte-carlo sampling to generate logits.
            if outputs['box_cls_var'] is not None:
                box_cls_var = outputs['box_cls_var'][i][0]
                box_cls_dists = torch.distributions.normal.Normal(
                    box_cls, scale=torch.sqrt(torch.exp(box_cls_var)))
                box_cls = box_cls_dists.rsample(
                    (self.model.cls_var_num_samples,))
                box_cls = torch.mean(box_cls.sigmoid_(), 0)
            else:
                box_cls = box_cls.sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.model.test_topk_candidates, box_delta.size(0))
            predicted_prob, classes_idxs = torch.max(box_cls, 1)
            predicted_prob, topk_idxs = predicted_prob.topk(num_topk)
            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.model.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            anchor_idxs = topk_idxs
            classes_idxs = classes_idxs[topk_idxs]

            box_delta = box_delta[anchor_idxs]
            anchors = anchors[anchor_idxs]

            cholesky_decomp = None

            if outputs['box_reg_var'] is not None:
                box_reg_var = outputs['box_reg_var'][i][0][anchor_idxs]
                # Construct cholesky decomposition using diagonal vars
                cholesky_decomp = covariance_output_to_cholesky(box_reg_var)

            # In case dropout is enabled, we need to compute aleatoric
            # covariance matrix and add it here:
            box_reg_epistemic_covar = None
            if is_epistemic:
                # Compute epistemic box covariance matrix
                box_delta_list_i = [
                    self.model.box2box_transform.apply_deltas(
                        box_delta_i[i][0][anchor_idxs],
                        anchors.tensor) for box_delta_i in box_delta_list]

                _, box_reg_epistemic_covar = inference_utils.compute_mean_covariance_torch(
                    box_delta_list_i)

            all_predicted_deltas.append(box_delta)
            all_predicted_boxes_cholesky.append(cholesky_decomp)
            all_anchors.append(anchors.tensor)
            all_predicted_prob.append(predicted_prob)
            all_predicted_prob_vectors.append(box_cls[anchor_idxs])
            all_classes_idxs.append(classes_idxs)
            all_predicted_boxes_epistemic_covar.append(box_reg_epistemic_covar)

        box_delta = cat(all_predicted_deltas)
        anchors = cat(all_anchors)

        if isinstance(all_predicted_boxes_cholesky[0], torch.Tensor):
            # Generate multivariate samples to be used for monte-carlo simulation. We can afford much more samples
            # here since the matrix dimensions are much smaller and therefore
            # have much less memory footprint. Keep 100 or less to maintain
            # reasonable runtime speed.
            cholesky_decomp = cat(all_predicted_boxes_cholesky)

            multivariate_normal_samples = torch.distributions.MultivariateNormal(
                box_delta, scale_tril=cholesky_decomp)

            # Define monte-carlo samples
            distributions_samples = multivariate_normal_samples.rsample(
                (1000,))
            distributions_samples = torch.transpose(
                torch.transpose(distributions_samples, 0, 1), 1, 2)
            samples_anchors = torch.repeat_interleave(
                anchors.unsqueeze(2), 1000, dim=2)

            # Transform samples from deltas to boxes
            t_dist_samples = self.sample_box2box_transform.apply_samples_deltas(
                distributions_samples, samples_anchors)

            # Compute samples mean and covariance matrices.
            all_predicted_boxes, all_predicted_boxes_covariance = inference_utils.compute_mean_covariance_torch(
                t_dist_samples)
            if isinstance(
                    all_predicted_boxes_epistemic_covar[0],
                    torch.Tensor):
                epistemic_covar_mats = cat(
                    all_predicted_boxes_epistemic_covar)
                all_predicted_boxes_covariance += epistemic_covar_mats
        else:
            # This handles the case where no aleatoric uncertainty is available
            if is_epistemic:
                all_predicted_boxes_covariance = cat(
                    all_predicted_boxes_epistemic_covar)
            else:
                all_predicted_boxes_covariance = []

            # predict boxes
            all_predicted_boxes = self.model.box2box_transform.apply_deltas(
                box_delta, anchors)

        return all_predicted_boxes, all_predicted_boxes_covariance, cat(
            all_predicted_prob), cat(all_classes_idxs), cat(all_predicted_prob_vectors)

    def post_processing_standard_nms(self, input_im):
        """
        This function produces results using standard non-maximum suppression. The function takes into
        account any probabilistic modeling method when computing the results. It can combine aleatoric uncertainty
        from heteroscedastic regression and epistemic uncertainty from monte-carlo dropout for both classification and
        regression results.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        outputs = self.retinanet_probabilistic_inference(input_im)

        return inference_utils.general_standard_nms_postprocessing(
            input_im, outputs, self.model.test_nms_thresh, self.model.max_detections_per_image)

    def post_processing_anchor_statistics(self, input_im):
        """
        This function produces box covariance matrices using anchor statistics. Uses the fact that multiple anchors are
        regressed to the same spatial location for clustering and extraction of box covariance matrix.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        outputs = self.retinanet_probabilistic_inference(input_im)

        return inference_utils.general_anchor_statistics_postprocessing(
            input_im,
            outputs,
            self.model.test_nms_thresh,
            self.model.max_detections_per_image,
            self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)

    def post_processing_mc_dropout_ensembles(self, input_im):
        """
        This function produces results using multiple runs of MC dropout, through fusion before or after
        the non-maximum suppression step.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES_DROPOUT.BOX_MERGE_MODE == 'pre_nms':
            return self.post_processing_standard_nms(input_im)
        else:
            outputs_dict = self.model(
                input_im,
                return_anchorwise_output=False,
                num_mc_dropout_runs=self.num_mc_dropout_runs)
            n_fms = len(self.model.in_features)
            outputs_list = [{key: value[i * n_fms:(i + 1) * n_fms] if value is not None else value for key,
                             value in outputs_dict.items()} for i in range(self.num_mc_dropout_runs)]

            # Merge results:
            results = [
                inference_utils.general_standard_nms_postprocessing(
                    input_im,
                    self.retinanet_probabilistic_inference(
                        input_im,
                        outputs=outputs),
                    self.model.test_nms_thresh,
                    self.model.max_detections_per_image) for outputs in outputs_list]

            # Append per-ensemble outputs after NMS has been performed.
            ensemble_pred_box_list = [
                result.pred_boxes.tensor for result in results]
            ensemble_pred_prob_vectors_list = [
                result.pred_cls_probs for result in results]
            ensembles_class_idxs_list = [
                result.pred_classes for result in results]
            ensembles_pred_box_covariance_list = [
                result.pred_boxes_covariance for result in results]

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.model.test_nms_thresh,
                self.model.max_detections_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)

    def post_processing_ensembles(self, input_im, model_dict):
        """
        This function produces results using multiple runs of independently trained models, through fusion before or after
        the non-maximum suppression step.

        Args:
            input_im (list): an input im list generated from dataset handler.
            model_dict (dict): dictionary containing list of models comprising the ensemble.
        Returns:
            result (instances): object instances

        """
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE == 'pre_nms':
            outputs_list = []

            for model in model_dict:
                outputs = model(input_im, return_anchorwise_output=True)
                outputs_list.append(outputs)

            outputs = self.retinanet_probabilistic_inference(
                input_im, ensemble_inference=True, outputs_list=outputs_list)
            return inference_utils.general_standard_nms_postprocessing(
                input_im, outputs, self.model.test_nms_thresh, self.model.max_detections_per_image)
        else:
            outputs_list = []
            for model in model_dict:
                self.model = model
                outputs_list.append(
                    self.post_processing_standard_nms(input_im))

            # Merge results:
            ensemble_pred_box_list = []
            ensemble_pred_prob_vectors_list = []
            ensembles_class_idxs_list = []
            ensembles_pred_box_covariance_list = []
            for results in outputs_list:
                # Append per-ensemble outputs after NMS has been performed.
                ensemble_pred_box_list.append(results.pred_boxes.tensor)
                ensemble_pred_prob_vectors_list.append(results.pred_cls_probs)
                ensembles_class_idxs_list.append(results.pred_classes)
                ensembles_pred_box_covariance_list.append(
                    results.pred_boxes_covariance)

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.model.test_nms_thresh,
                self.model.max_detections_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)

    def post_processing_bayes_od(self, input_im):
        """
        This function produces results using forms of bayesian inference instead of NMS for both category and box results.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        box_merge_mode = self.cfg.PROBABILISTIC_INFERENCE.BAYES_OD.BOX_MERGE_MODE
        cls_merge_mode = self.cfg.PROBABILISTIC_INFERENCE.BAYES_OD.CLS_MERGE_MODE

        outputs = self.retinanet_probabilistic_inference(input_im)

        predicted_boxes, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors = outputs

        keep = batched_nms(
            predicted_boxes,
            predicted_prob,
            classes_idxs,
            self.model.test_nms_thresh)

        keep = keep[: self.model.max_detections_per_image]

        match_quality_matrix = pairwise_iou(
            Boxes(predicted_boxes), Boxes(predicted_boxes))

        box_clusters_inds = match_quality_matrix[keep, :]
        box_clusters_inds = box_clusters_inds > self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD

        # Compute mean and covariance for every cluster.

        predicted_prob_vectors_list = []
        predicted_boxes_list = []
        predicted_boxes_covariance_list = []

        predicted_prob_vectors_centers = predicted_prob_vectors[keep]
        for box_cluster, predicted_prob_vectors_center in zip(
                box_clusters_inds, predicted_prob_vectors_centers):
            cluster_categorical_params = predicted_prob_vectors[box_cluster]
            center_binary_score, center_cat_idx = torch.max(
                predicted_prob_vectors_center, 0)
            cluster_binary_scores, cat_idx = cluster_categorical_params.max(
                1)
            class_similarity_idx = cat_idx == center_cat_idx
            if cls_merge_mode == 'bayesian_inference':
                predicted_prob_vectors_list.append(
                    cluster_categorical_params.mean(0).unsqueeze(0))
            else:
                predicted_prob_vectors_list.append(
                    predicted_prob_vectors_center.unsqueeze(0))

            # Switch to numpy as torch.inverse is too slow.
            cluster_means = predicted_boxes[box_cluster,
                                            :][class_similarity_idx].cpu().numpy()
            cluster_covs = predicted_boxes_covariance[box_cluster, :][class_similarity_idx].cpu(
            ).numpy()

            predicted_box, predicted_box_covariance = inference_utils.bounding_box_bayesian_inference(
                cluster_means, cluster_covs, box_merge_mode)
            predicted_boxes_list.append(
                torch.from_numpy(np.squeeze(predicted_box)))
            predicted_boxes_covariance_list.append(
                torch.from_numpy(predicted_box_covariance))

        # Switch back to cuda for the remainder of the inference process.
        result = Instances(
            (input_im[0]['image'].shape[1],
             input_im[0]['image'].shape[2]))

        if len(predicted_boxes_list) > 0:
            if cls_merge_mode == 'bayesian_inference':
                predicted_prob_vectors = torch.cat(
                    predicted_prob_vectors_list, 0)
                predicted_prob, classes_idxs = torch.max(
                    predicted_prob_vectors, 1)
            elif cls_merge_mode == 'max_score':
                predicted_prob_vectors = predicted_prob_vectors[keep]
                predicted_prob = predicted_prob[keep]
                classes_idxs = classes_idxs[keep]
            result.pred_boxes = Boxes(
                torch.stack(
                    predicted_boxes_list,
                    0).to(self.model.device))
            result.scores = predicted_prob
            result.pred_classes = classes_idxs
            result.pred_cls_probs = predicted_prob_vectors
            result.pred_boxes_covariance = torch.stack(
                predicted_boxes_covariance_list, 0).to(self.model.device)
        else:
            result.pred_boxes = Boxes(predicted_boxes)
            result.scores = torch.zeros(
                predicted_boxes.shape[0]).to(
                self.model.device)
            result.pred_classes = classes_idxs
            result.pred_cls_probs = predicted_prob_vectors
            result.pred_boxes_covariance = torch.empty(
                (predicted_boxes.shape + (4,))).to(self.model.device)
        return result
