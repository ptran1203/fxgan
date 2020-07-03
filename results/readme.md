

# Experiment results

<!-- ## 1. Accuracy on 5 classes
| Model | Accuracy |
|--|--|
| VGG16 | 0.452 |
| VGG16 + augment | 0.437 |
| VGG16 + BAGAN | 0.452 |
| VGG16 + NewGan v2 | 0.431 |
| VGG16 + NewGan v1 | 0.416 | -->


## 2. AUC score on 5 classes
| Model | No Finding | Infiltration | Atelectasis | Effusion | Nodule |
|--|--|--|--|--|--|
| VGG16 |0.713 |0.702 |0.738 |0.812 |0.728 |
| VGG16 + augment |0.722 | **0.712** |0.742 | **0.818** |**0.735** |
| VGG16 + BAGAN | **0.726** |0.699 | **0.752** |0.813 |0.732 |
| VGG16 + NewGan v1 |0.723 |0.706 |0.746 |0.814 |0.719 |


## 3. AUC score on 5 classes: augment 3k per class
| Model | No Finding | Infiltration | Atelectasis | Effusion | Nodule |
|--|--|--|--|--|--|
|newgan |**0.736** |0.69 |0.744 |0.813 |**0.737** |
|bagan |0.712 |0.693 | **0.747** |0.81 |0.725 |
|vgg16 + augment |0.725 |**0.699** |0.735 |**0.814** |0.728 |