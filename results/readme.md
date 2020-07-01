

# Experiment results

## 1. Accuracy on 5 classes
| Model | Accuracy |
|--|--|
| VGG16 | 0.452 |
| VGG16 + BAGAN | 0.452 |
| VGG16 + NewGan v2 | 0.431 |
| VGG16 + NewGan v1 | 0.416 |


## 2. AUC score on 5 classes
| Model | No Finding | Infiltration | Atelectasis | Effusion | Nodule |
|--|--|--|--|--|--|
| VGG16 |0.724 |0.704 |0.752 | **0.81** | 0.748 |
| VGG16 + BAGAN | **0.727** | **0.711** | **0.765** | 0.802 | **0.756** |
| VGG16 + NewGan v2 |0.704 |0.689 |0.745 |0.809 |0.737 |
| VGG16 + NewGan v1 |0.708 |0.692 |0.74 |0.805 |0.745 |