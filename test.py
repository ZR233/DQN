import numpy as np

a = []
b = np.zeros((2,3))
for i in range(5):
    a.append(b)

d = a[2:3]
print(d)