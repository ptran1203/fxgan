

## 2. Train relation net
- Provice 3 dataset: train, support and test
- Including 3 classes: normal, pneumonia and fake
- the problem is 3-way 5-shot

### Support set
[5 normal images]
[5 pneumonia images]
[5 fake images]
### Train set
[N query images]

feature encoder

Concatenate each class
feature with every query feature (total = train images * 15)

calculate relation score for each class

loss = mse(relations,one_hot_labels)

scores = [0.7, 0.8, 0.1]
one_hot = [1, 2, 3]