

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

|  | VGG16 | VGG16 + decay |
|--|--|--|
| No Finding | **0.72** | 0.66 |
| Infiltration | 0.651 | **0.674** |
| Atelectasis | 0.706 | **0.743** |
| Effusion | 0.767 | **0.78** |
| Nodule | 0.714 | **0.716** |
| Pneumothorax | **0.778** | 0.751 |
| Mass | **0.768** | 0.751 |
| Consolidation | 0.734 | **0.769** |
| Pleural_Thickening | **0.687** | 0.673 |
| Cardiomegaly | **0.886** | 0.87 |
| **Average** | **0.741** | 0.739 |