

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

| | VGG16 | VGG16 + standard aug | VGG16 + Bagan | VGG 16 + NewGAN |
|--|--|--|--|--|
|Infiltration| 0.702 | **`0.712`** | 0.699 | 0.706 |
|Atelectasis| 0.738 | 0.742| **`0.752`** | 0.746 |
|Effusion| 0.812 | **`0.818`** | 0.813 |0.814 |
|Nodule| 0.728 |**`0.735`**| 0.732 |0.719 |


## 3. AUC score on 5 classes: (augment 3000 images per class)

| | VGG16 + standard aug | VGG16 + Bagan | VGG 16 + NewGAN |
|--|--|--|--|
|Infiltration| 0.69  | 0.698 | **`0.71`** |
|Atelectasis| 0.735 | **`0.751`** | 0.751 |
|Effusion| 0.814 | **`0.819`** | 0.817|
|Nodule| 0.728 | **`0.738`** | 0.724 |
|Average| 0.741 | 0.7515 | **`0.750`** |



VGG16     |0.682 |0.739 |0.786 |0.721 |0.755 |0.776 |0.774 |0.683 |0.884 | - 0.755
standard  |0.661 |0.731 |0.779 |0.721 |0.771 |0.774 |0.755 |0.690 |0.865 |- 0.749
GAN v1    |0.671 |0.744 |0.774 |0.736 |0.772 |0.775 |0.757 |0.695 |0.878 | -  0.755