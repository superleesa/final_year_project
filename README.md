# DehazeGAN: Sand-Dust Image Restoration using Semi-supervised Adversarial Learning
## About The Project
Removing color cast for image restoration in sand and dust conditions is essential for improving the accuracy and performance of outdoor computer vision systems. Enhancements in this area are crucial for environments heavily impacted by sand and dust, where visibility and image quality are significantly degraded. Despite so, only few deep learning models are published to denoise and remove color cast from sandstorm images.

Therefore, we propose a novel deep-learning model extended from TOENet (Gao et al., 2023), DehazeGAN to remove color cast from sandstorm images through unsupervised fine-tuning (UFT) using adversarial learning and supervised fine-tuning (SFT) via paired augmentation.

## Getting Started
### Installation Prerequisites
We developed and tested our application using the following configurations. 
- OS: Linux Ubuntu 22.04
- Python Version: 3.10
- Nvidia Driver Version: 545.29.06
- Nvidia CUDA Tool-kit Version: 12.0.1 

Currently, our project only supports device with GPU. If your device does not have GPU, please consider using cloud services or remote desktop with GPU.

### Installation Guide
In the terminal, clone a repository for our project from GitHub using git command 
```sh
git clone https://github.com/superleesa/final_year_project.git
```
2. Go to the repository directory and install Python libraries required for the app, using the command 
```sh
pip install -r requirements.txt.
```

## Training Guidelines
For training of the dataset, we would need to download the SIE Dataset. The paper on the SIE Dataset can be found here: [LINK](https://link.springer.com/article/10.1007/s00371-022-02448-8). The SIE Dataset should be downloaded to the ```Data``` folder, in the format of the Data DIrectory Structure below.

### Data Directory Structure
```
Data/
└── paired/
|   ├── ground_truth/
|   │   ├── ...
|   │   ├── ...
|   │   └── ...
|   └── noisy/
|       ├── ...
|       ├── ...
|       └── ...
└── unpaired/
    ├── clear/
    │   ├── ...
    │   ├── ...
    │   └── ...
    └── noisy/
        ├── ...
        ├── ...
        └── ...

```
### Training of Model
We follow the training pipeline as illustrated below:
![Training Pipeline](docs/training_pipeline.png)

- Unsupervised Finetuning Training (UFT): Refer to [README.md](train/uft/README.md) for UFT Training.
- Supervised Finetuning Training (SFT): Refer to [README.md](train/sft/toenet_base/README.md) for SFT Training.

## Evaluation Guidelines
See [README.md](evaluation/README.md) for Evaluation Guidelines.

## Results
### Metrics Comparison with the Other Models
![Comparison with the other models](docs/results_1.png)

### Color Histogram Comparison with Other Models
![Color Histogram](docs/model_comparisons_ver0.2.png)

## Acknowledgements
The base model of our project is based on the [TOENet] (https://github.com/YuanGao-YG/TOENet).









