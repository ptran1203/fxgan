
bg = BatchGen(BatchGen.TRAIN, 10, 'multi', 64, prune_classes=None)
clss = 13
xx = bg.get_samples_for_class(clss)
xx = np.expand_dims(xx, axis=-1)
show_samples(xx[:10])
print(bg.per_class_count)
print([x for x,y in CATEGORIES_MAP.items() if y == clss])

x, y = pickle_load('/content/drive/My Drive/bagan/dataset/multi_chest/imgs_labels.pkl')
to_keep = [i for i, l in enumerate(y) if '|' not in l]
to_keep = np.array(to_keep)
x = x[to_keep]
y = y[to_keep]  

count = Counter(y)
print(count)
print(len(count.keys()))
print(x.shape, y.shape)