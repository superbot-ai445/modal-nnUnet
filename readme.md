# modal-nnUnet
This repository uses Modal to simplify the training process of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). This project is forked from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and mainly modifies the data preprocessing and training configuration scripts. For specific details about the original project, please refer to [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet).

## install and set up modal
```
pip install modal
python3 -m modal setup
```
## install and set up nnUnet
```
git clone https://github.com/superbot-ai445/modal-nnUnet.git
cd modal-nnUnet
```

## run the training
```
modal run train.py
```
## Optimization and Customization

- GPU Type: If your model is large or requires faster training speeds, you can change the GPU_TYPE to a more powerful model, such as H100.

- Multi-GPU Training: If your task supports multi-GPU, you can increase GPU_COUNT to 2 or more.

- Model Saving: After training is complete, the model will be saved to Modal's persistent volume or to cloud storage that you specify. The default path is nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth. You can download the model to your local machine using the command 
```
modal volume get nn_cache nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

