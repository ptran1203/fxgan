

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
| VGG16 |0.716 |0.696 |0.764 |0.817 |0.756|
| VGG16 + augment |0.726 |0.711 |0.758 |0.808 |0.737 |
| VGG16 + BAGAN | **0.727** | **0.711** | **0.765** | 0.802 | **0.756** |
| VGG16 + NewGan v1 |0.7 |0.72 |0.724 |0.796 |0.735 |
