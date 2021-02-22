# A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving

This repository is the official implementation of [A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving](https://arxiv.org/abs/2011.10671).
## Requirements

#### Software Support:
Name | Supported Versions
--- | --- |
Ubuntu |18.04, 20.04
Python | 3.7 ,3.8
CUDA | 10.1 ,10.2, 11.0
Cudnn | 7.6.5 , 8.0.1
PyTorch | 1.5 , 1.5.1, 1.6

To install requirements virtualenv and virtualenvwrapper should be available on the target machine.

**Virtual Environment Creation:**
```
# Clone repo
git clone https://github.com/asharakeh/pod_compare.git
cd pod_compare

# Create python virtual env
mkvirtualenv pod_compare

# Add library path to virtual env
add2virtualenv src

# Install requirements
cat requirements.txt | xargs -n 1 -L 1 pip install
```
## Datasets
Download the Berkeley Deep Drive (BDD) Object Detection Dataset [here](https://bdd-data.berkeley.edu/). The BDD
dataset should have the following structure:
<br>
 
     └── BDD_DATASET_ROOT
         ├── info
         |   └── 100k
         |       ├── train
         |       └── val
         ├── labels
         └── images
                ├── 10K
                └── 100K
                    ├── test
                    ├── train
                    └── val
                   
Download the KITTI Object Detection Dataset [here](http://www.cvlibs.net/datasets/kitti/eval_object.php). The KITTI
dataset should have the following structure:
<br> 

    └── KITTI_DATASET_ROOT
        ├── object
            ├── training    <-- 7481 train data
            |   ├── image_2
            |   ├── calib
            |   └── label_2
            └── testing     <-- 7580 test data
                   ├── image_2
                   └── calib

Download the Lyft Object Detection Dataset [here](https://self-driving.lyft.com/level5/data/). The Lyft
dataset needs to be converted to KITTI format first using the [official Lyft dataset API](https://github.com/lyft/nuscenes-devkit).
The Lyft dataset should have the following structure:
<br> 

    └── LYFT_DATASET_ROOT
        └── training
            ├── image_2
            └── label_2

For all three datasets, labels need to be converted to COCO format. To do so, run the following:
```
python src/core/datasets convert_bdd_to_coco.py --dataset-dir /path/to/bdd/dataset/root
```
```
python src/core/datasets convert_kitti_to_coco.py --dataset-dir /path/to/kitti/dataset/root
```
```
python src/core/datasets convert_lyft_to_coco.py --dataset-dir /path/to/lyft/dataset/root
```
If the script to convert BDD labels to COCO format does not work, please use [these pre-converted labels](https://drive.google.com/file/d/1hOd3zX1Qt0_uV64uJBLidavjbtrv1tXI/view?usp=sharing). 
## Training
To train the model(s) in the paper, run this command:

``` train
python src/train_net.py
--num-gpus 2
--dataset-dir /path/to/bdd/dataset/root
--config-file BDD-Detection/retinanet/name_of_config.yaml
--random-seed xx
--resume
```

## Evaluation
For running inference and evaluation of a model, run the following code:
```eval
python src/apply_net.py --dataset-dir /path/to/test/dataset/root --test-dataset test_dataset_name --config-file BDD-Detection/retinanet/name_of_config.yaml --inference-config Inference/name_of_inference_config.yaml
```

`--test-dataset` can be one of `bdd_val`, `kitti_val`, or `lyft_val`. `--dataset-dir` corresponds to the root directory of the dataset used.

Evaluation code will run inference on the test dataset and then will generate mAP, Negative Log Likelihood, Calibration Error, and Minimum Uncertainty Error results. If only evaluation of metrics is required,
add `--eval-only` to the above code snippet.

We provide a list of config combinations that generate the architectures used in our paper:

Method Name | Config File | Inference Config File | Pretrained Model
--- | --- | --- | ---
Baseline RetinaNet | retinanet_R_50_FPN_1x.yaml| standard_nms.yaml | [retinanet_R_50_FPN_1x.pth](https://drive.google.com/file/d/1kOuXSjRoBLTDXYtCpjzI6Q-TBizFRaw5/view?usp=sharing)
Loss Attenuation |retinanet_R_50_FPN_1x_reg_cls_var.yaml| standard_nms.yaml | [retinanet_R_50_FPN_1x_reg_cls_var.pth](https://drive.google.com/file/d/1yMspawb66LrZZidCdY3EeQlsF6aYYBrO/view?usp=sharing)
Loss Attenuation + Dropout | retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml | mc_dropout_ensembles_pre_nms.yaml | [retinanet_R_50_FPN_1x_reg_cls_var_dropout.pth](https://drive.google.com/file/d/1ALGDsnAVJXK5zgSVbRSziYZlddUqrt2a/view?usp=sharing)
BayesOD | retinanet_R_50_FPN_1x_reg_cls_var.yaml | bayes_od.yaml | [retinanet_R_50_FPN_1x_reg_cls_var.pth](https://drive.google.com/file/d/1yMspawb66LrZZidCdY3EeQlsF6aYYBrO/view?usp=sharing)
BayesOD + Dropout | retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml | bayes_od_mc_dropout.yaml | [retinanet_R_50_FPN_1x_reg_cls_var_dropout.pth](https://drive.google.com/file/d/1ALGDsnAVJXK5zgSVbRSziYZlddUqrt2a/view?usp=sharing)
Pre-NMS Ensembles| retinanet_R_50_FPN_1x_reg_cls_var.yaml | ensembles_pre_nms.yaml | -
Post-NMS Ensembles| retinanet_R_50_FPN_1x_reg_cls_var.yaml | ensembles_post_nms.yaml | -
Black Box| retinanet_R_50_FPN_1x_dropout.yaml | mc_dropout_ensembles_post_nms.yaml | [retinanet_R_50_FPN_1x_dropout.pth](https://drive.google.com/file/d/1ZL0djkgHokfeCnkSy0_jtxOS1T3ROmlL/view?usp=sharing)
Output Redundancy| retinanet_R_50_FPN_1x.yaml | anchor_statistics.yaml |[retinanet_R_50_FPN_1x.pth](https://drive.google.com/file/d/1kOuXSjRoBLTDXYtCpjzI6Q-TBizFRaw5/view?usp=sharing)

Ensemble methods require multiple independent training runs using different random seeds. 
To do so, run the training code while adding `random-seed xx`. We test with 5 runs using seed values of 0, 1000, 2000, 3000, and 4000 in our paper.

For evaluating with PDQ, please use the official PDQ code for COCO data available [here](https://github.com/david2611/pdq_evaluation).

## Citation
If you use this code, please cite our paper:
```
@misc{feng2020review,
title={A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving}, 
author={Di Feng and Ali Harakeh and Steven Waslander and Klaus Dietmayer},
year={2020},
eprint={2011.10671},
archivePrefix={arXiv},
primaryClass={cs.CV}}
```

## License
This code is released under the [Apache 2.0 License](LICENSE.md).
