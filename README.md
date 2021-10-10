# U-Net for Semantic Segmentation on Unbalanced Aerial Imagery
#### Note: this repository is still developing but the colab notebook is complete for training and evaluations.

Read the article at [Towards data science](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vYZYXDMfs9hK6KvXY3v1iTPiln9OeJZE?usp=sharing)

[Kaggle Dataset](https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery)

```
UNet-AerialSegmentation
       		   ├── dataloader.py
       		   ├── losses.py
      		   ├── model.py
      		   ├── train.py
       		   └── inference.py

```

![dataset_sample1](https://user-images.githubusercontent.com/56114938/133141953-46df55be-4dfb-4084-b8d0-a63a56712ab0.png)

## Training 
```
!python train.py --num_epochs 2 --batch 2 --loss focalloss
```