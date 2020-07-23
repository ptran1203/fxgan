
## The GAN project for chest-xray classification
- This repository is implemented based on https://github.com/IBM/BAGAN
- To run the experiment please check the file **try_gan.ipynb**

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
