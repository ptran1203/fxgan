

# Experiment results

<!-- ## 1. Accuracy on 5 classes
| Model | Accuracy |
|--|--|
| VGG16 | 0.452 |
| VGG16 + augment | 0.437 |
| VGG16 + BAGAN | 0.452 |
| VGG16 + NewGan v2 | 0.431 |
| VGG16 + NewGan v1 | 0.416 | -->


## 2. AUC score on 5 classes (augment 1000 images per class)
|  | VGG16 | VGG16 + standard aug | VGG16 + Bagan | VGG 16 + NewGAN |
|--|--|--|--|--|
| No finding | 0.713 | 0.722 | **0.726** |0.723 |
|Infiltration| 0.702 | **0.712** | 0.699 | 0.706 |
|Atelectasis| 0.738 | 0.742| **0.752** | 0.746 |
|Effusion| 0.812 | **0.818** | 0.813 |0.814 |
|Nodule| 0.728 |**0.735**| 0.732 |0.719 |


## 3. AUC score on 5 classes: (augment 3000 images per class)
|  | VGG16 + standard aug | VGG16 + Bagan | VGG 16 + NewGAN |
|--|--|--|--|
| No finding | 0.725 | 0.712 | **0.736** |
|Infiltration| **0.699** | 0.693 | 0.69 |
|Atelectasis| 0.735 | **0.747** | 0.744|
|Effusion| **0.814** | 0.81 |0.813|
|Nodule| 0.728 |0.725| **0.737** |