

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

|  | VGG16 + standard augment | GAN v2 | BAGAN 0.001 center |
|--|--|--|--|
| No Finding | 0.708 | 0.709 | **0.728** |
| Infiltration | 0.681 | 0.689 | **0.705** |
| Atelectasis | 0.749 | 0.769 | **0.783** |
| Effusion | 0.798 | 0.793 | **0.803** |
| Nodule | 0.688 | **0.71** | 0.666 |
| Pneumothorax | 0.77 | **0.798** | 0.773 |
| Mass | **0.76** | 0.733 | 0.698 |
| Consolidation | 0.593 | 0.655 | **0.69** |
| Pleural_Thickening | 0.682 | **0.693** | 0.669 |
| Cardiomegaly | 0.877 | **0.888** | 0.875 |
| Emphysema | 0.714 | **0.782** | 0.75 |
| Fibrosis | 0.717 | 0.672 | **0.729** |
| Edema | 0.75 | **0.797** | 0.754 |
| Pneumonia | **0.534** | 0.496 | 0.531 |
| Hernia | **0.894** | 0.809 | 0.53 |
| **Average** | 0.728 | **0.733** | 0.712 |

