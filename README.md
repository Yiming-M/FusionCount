# FusionCount

Official PyTorch implementation of [*FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion*](https://arxiv.org/abs/2202.13660).

## Requirements

The code was writen in Python 3.9.10. The PyTorch version used is 1.10.0

## Datasets

ShanghaiTech A & B can be found on [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech).

## Model Structure

![FusionCount](https://user-images.githubusercontent.com/45311510/161753152-1019e96e-18da-43de-9af0-46c6ed55bd12.png)

## Training

Please refer to [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) for data preparation and training details.

## Results

| Model              | Multi-Adds | MAE (SHA) | RMSE (SHA) | MAE (SHB) | RMSE (SHB) |
|--------------------|------------|-----------|------------|-----------|------------|
| CSRNet             | 856.99 G   | 68.2      | 115.0      | 10.6      | 16.0       |
| CAN                | 908.05 G   | 62.3      | 100.0      | 7.8       | 12.2       |
| BL                 | 853.70 G   | 62.8      | 101.8      | 7.7       | 12.7       |
| DM-Count           | 853.70 G   | 59.7      | 95.7       | 7.4       | 11.8       |
| FusionCount (ours) | 815.00 G   | 62.2      | 101.2      | 6.9       | 11.8       |

![sha_img_96](https://user-images.githubusercontent.com/45311510/161753232-fe2eb9de-2d64-44fe-86ce-8566a5b09196.jpg)
![sha_img_116](https://user-images.githubusercontent.com/45311510/161753247-1fd9fc49-53bb-425f-bb31-2c6a018afde4.jpg)
![shb_img_21](https://user-images.githubusercontent.com/45311510/161753262-3a6e732d-904e-4789-b842-f3ba3c62e912.jpg)
![shb_img_30](https://user-images.githubusercontent.com/45311510/161753278-0ff04c26-4c53-4e23-afe6-1fe47df7e700.jpg)

## Citation

We have submitted the paper to IEEE ICIP 2022, and it's currently under review. Please cite from arXiv if you found our work useful.

```
@misc{https://doi.org/10.48550/arxiv.2202.13660,
  doi = {10.48550/ARXIV.2202.13660},
  
  url = {https://arxiv.org/abs/2202.13660},
  
  author = {Ma, Yiming and Sanchez, Victor and Guha, Tanaya},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
