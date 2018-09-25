import numpy as np


x = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
z = np.array([100,200,300])
x = np.insert(x, obj=0, values=z, axis=1)
np.random.shuffle(x)

print(x)
print()

y = x[:,0]
x = x[:,1:]

print(y)
print()
print(x)