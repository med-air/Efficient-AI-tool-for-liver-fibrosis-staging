## Introduction

Pytorch implementation for paper **[Efficient Development and Comprehensive Evaluation of a Human-in-the-loop Artificial Intelligence Tool for Liver Fibrosis Staging in CT](https://github.com/med-air/Liver_Fibrosis_Staging_in_CT/)**



## Setup

```bash
Package                Version
---------------------- -------------------
h5py                   3.1.0
numpy                  1.15.4
opencv-python          4.5.2.52
pandas                 1.1.5
SimpleITK              2.0.2
Scikit-learn           0.24.2
torch                  1.4.0
torchvision            0.5.0
```


## Dara preparing

#### 1. Sort out the data and code.
The segmentation results (Seg.nii.gz) can be generated via our human-in-the-loop strategy. The segmentation networks are based on **[UMMKD](https://github.com/carrenD/ummkd)** and **[TransUNet](https://github.com/Beckschen/TransUNet)**.

```bash
.
├── code
    ├──datasets
            └── dataset.py
    ├──train.py
    ├──test.py
    └──...
├── models_save
    └── Liver_Fibrosis_Staging
└── data
    └──final
        ├── train
        └── test
    └──raw
        ├── train
            ├── S0
                ├── Patient-XXXX
                    ├── Img_raw.nii.gz
                    ├── Seg.nii.gz
                └──...
            ├── S1
            ├── S2
            ├── S3
            └── S4
        └── test
            ├── S0
            ├── S1
            ├── S2
            ├── S3
            └── S4
```

#### 2. Use the segmentation result to crop the regions of liver and spleen.
```bash 
python preprocessing.py 
```

## Training
```bash 
python train.py --max_epoch 35 --model_path <your model path>
```
## Testing
```bash 
python test.py --model_path <your model path>
```

## The AI-assistance software

After training the segmentation and classification models, then you can use the trained model to build an AI-assistance software. The code of UI development is in the folder "liver_fibrosis_staging_system".

The demo of this AI-assistance software can be seen as below:

<p align="center">
<img src="./demo.png" alt="intro" width="50%"/>
</p>

## Contact
For any questions, please contact 'wama@cse.cuhk.edu.hk'



