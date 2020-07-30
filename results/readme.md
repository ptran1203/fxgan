

# Experiment results


|  | VGG16 + standard augment | Oneshot-xray GAN | BAGAN |
|--|--|--|--|
| No Finding | 0.679 | **0.704** | 0.695 |
| Infiltration | 0.675 | **0.719** | 0.702 |
| Atelectasis | 0.735 | **0.762** | 0.758 |
| Effusion | 0.794 | **0.804** | 0.786 |
| Nodule | 0.701 | **0.73** | 0.729 |
| Pneumothorax | **0.753** | 0.748 | 0.73 |
| Mass | **0.767** | 0.764 | 0.743 |
| Consolidation | 0.64 | **0.68** | 0.648 |
| Pleural_Thickening | 0.684 | **0.719** | 0.684 |
| Cardiomegaly | 0.877 | 0.887 | **0.894** |
| Emphysema | **0.742** | 0.72 | 0.696 |
| Fibrosis | 0.638 | **0.675** | 0.581 |
| Edema | 0.486 | **0.665** | 0.551 |
| Pneumonia | 0.458 | **0.546** | 0.478 |
| Hernia | 0.685 | **0.746** | 0.535 |
| **Average** | 0.688 | **0.725** | 0.681 |


|  | BAGAN | VGG16 + standard augment |
|--|--|--|
| No Finding | **0.693** | 0.676 |
| Infiltration | **0.708** | 0.677 |
| Atelectasis | **0.758** | 0.753 |
| Effusion | **0.795** | 0.782 |
| Nodule | 0.72 | **0.724** |
| Pneumothorax | 0.706 | **0.741** |
| Mass | 0.745 | **0.751** |
| Consolidation | 0.68 | **0.693** |
| Pleural_Thickening | 0.668 | **0.694** |
| Cardiomegaly | **0.883** | 0.874 |
| Emphysema | 0.722 | **0.735** |
| Fibrosis | 0.621 | **0.648** |
| Edema | 0.544 | **0.653** |
| Pneumonia | 0.502 | **0.515** |
| Hernia | 0.599 | **0.697** |
| **Average** | 0.69 | **0.708** |