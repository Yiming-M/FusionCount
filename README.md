# FusionCount

Official PyTorch implementation of the ICIP 2022 paper [*FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion*](https://arxiv.org/abs/2202.13660).

## Requirements

The code was writen in Python 3.9.10. The PyTorch version used is 1.10.0

## Datasets

ShanghaiTech A & B can be found on [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech).

## Model Structure

![FusionCount](https://user-images.githubusercontent.com/45311510/161753152-1019e96e-18da-43de-9af0-46c6ed55bd12.png)

## Training

Please refer to [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) for data preparation and training details. Please notice that you will also have to normalise the predicted density map, as illustrated in the code snippet from [DM-Count/models.py](https://github.com/cvlab-stonybrook/DM-Count/blob/master/models.py) below.

```python
...
mu = self.density_layer(x)
B, C, H, W = mu.size()
mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
mu_normed = mu / (mu_sum + 1e-6)
return mu, mu_normed
```

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

The paper has been accepted by ICIP 2022 and published on [IEEE Xplore](https://ieeexplore.ieee.org/document/9897322). You can also find the arXiv version [here](https://arxiv.org/abs/2202.13660). Please cite us if you find it useful!

```latex
@INPROCEEDINGS{9897322,
  author={Ma, Yiming and Sanchez, Victor and Guha, Tanaya},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  title={Fusioncount: Efficient Crowd Counting Via Multiscale Feature Fusion},
  year={2022},
  volume={},
  number={},
  pages={3256-3260},
  doi={10.1109/ICIP46576.2022.9897322}
}
```
