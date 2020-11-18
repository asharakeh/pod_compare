# A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving

This repository is the official implementation of [A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving](https://arxiv.org/abs/2030.12345). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

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

## Training
To train the model(s) in the paper, run this command:

``` train
python src/train_net.py
--num-gpus 2
--dataset-dir /path/to/coco/dataset
--config-file BDD-Detection/architecture_name/config_name.yaml
--random-seed xx
--resume
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python src/apply_net.py --dataset-dir /path/to/test/dataset --test-dataset test_dataset_name --config-file path/to/config.yaml --inference-config /path/to/inference/config.yaml --random-seed 0
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 