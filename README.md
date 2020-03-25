

## 2. Train relation net
- Provice 3 dataset: train, support and test
- Including 3 classes: normal, pneumonia and fake
- the problem is 3-way 5-shot

[5 normal images]
[5 pneumonia images]
[5 fake images]
[N query images]

feature encoder

Concatenate each class
feature with every query feature 

calculate relation score for each class