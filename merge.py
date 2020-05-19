import numpy as np

shape  = (2, 2, 100)

x = np.ones(shape)
y = np.zeros(shape)

# rand = np.random.rand(shape[-1])
rand = np.random.uniform(0, 1, shape[-1])
old = np.sum(x)
print('old: ', old)

# print(x.tolist())

# loop over the channel
for i in range(x.shape[-1]):
    if rand[i] >= 0.5:
        x[:, :, i] = y[:,:,i]

New = np.sum(x)
print('new: ', New)
# print(x.tolist())

print('Rate: ', New / old)

print(rand[rand >= 0.5].size)