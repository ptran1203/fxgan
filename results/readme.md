

# Experiment results

  
## 2. AUC score on 10 classes (augment 1000 images per class)
|  | BAGAN | GAN v1 | VGG16 | VGG16 + standard augment |
|--|--|--|--|--|
| No Finding | 0.687 | **0.694** | 0.644 | 0.683 |
| Infiltration | **0.669** | 0.656 | 0.663 | 0.663 |
| Atelectasis | **0.74** | 0.733 | 0.73 | 0.729 |
| Effusion | 0.759 | **0.775** | 0.768 | 0.773 |
| Nodule | **0.736** | 0.728 | 0.713 | 0.718 |
| Pneumothorax | 0.74 | **0.746** | 0.721 | 0.743 |
| Mass | 0.752 | 0.753 | 0.736 | **0.759** |
| Consolidation | 0.754 | 0.768 | **0.772** | 0.756 |
| Pleural_Thickening | **0.685** | 0.666 | 0.658 | 0.669 |
| Cardiomegaly | 0.864 | **0.867** | 0.847 | 0.862 |
| **Average** | **0.739** | **0.739** | 0.725 | 0.736 |


## 128x128
|  | VGG16 + standard augment | GAN v2 | GAN Vnew |
|--|--|--|--|
| No Finding | 0.679 | 0.695 | **0.7** |
| Infiltration | 0.675 | **0.721** | **0.721** |
| Atelectasis | 0.735 | **0.76** | 0.752 |
| Effusion | 0.794 | 0.808 | **0.825** |
| Nodule | 0.701 | 0.725 | **0.741** |
| Pneumothorax | **0.753** | **0.753** | 0.747 |
| Mass | **0.767** | 0.766 | 0.739 |
| Consolidation | 0.64 | **0.69** | 0.668 |
| Pleural_Thickening | 0.684 | **0.724** | 0.69 |
| Cardiomegaly | 0.877 | **0.888** | 0.882 |
| Emphysema | **0.742** | 0.696 | 0.732 |
| Fibrosis | 0.638 | **0.667** | 0.628 |
| Edema | 0.486 | 0.671 | **0.683** |
| Pneumonia | 0.458 | 0.381 | **0.541** |
| Hernia | 0.685 | **0.711** | 0.538 |
| **Average** | 0.688 | **0.71** | 0.706 |