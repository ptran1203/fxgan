
## The GAN project for chest-xray classification
- This repository is implemented based on https://github.com/IBM/BAGAN
- To run the experiment please check the file **try_gan.ipynb**

### History table
| commit hash | comment |
|--|--|
| [97f1d](../../commit/97f1dba8dc6f8f69acac35bdcc6588add513035f) | remove bagan_old |
| [1cee6](../../commit/1cee698d28210cd18bf6914a752857611d5ef548) | Use auxiliary label |
| [a1e22](../../commit/a1e2254378c8bae92bf89f658be629c6cc28fa9e) | Use triplet loss (newest)|
| [8f627](../../commit/8f6274d2d6f87ed87986de66ee92fbc411ae7330) | Before use K_shot |
| [f7005](../../commit/f700526fd86de862683150f7302ae8f0f900388f) | Before remove dc generator|

### Chest-xray 14 dataset


1. samples count (multi-class sample are removed)

| Category | samples |
|--|--|
| No Finding | 60361 |
| Infiltration | 9547 |
| Atelectasis | 4215 |
| Effusion | 3955 |
| Nodule | 2705 |
| Pneumothorax | 2194 |
| Mass | 2139 |
| Consolidation | 1310 |
| Pleural_Thickening | 1126 |
| Cardiomegaly | 1093 |
| Emphysema | 892 |
| Fibrosis | 727 |
| Edema | 628 |
| Pneumonia | 322 |
| Hernia | 110 |

2. Data distribution (Without No finding case)

![label_counts](images/label_counts_2.png)  
