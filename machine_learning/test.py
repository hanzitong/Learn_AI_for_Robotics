


import numpy as np


# arr = np.array([[1,2], [3,4], [4,5]])
# arr2 = np.array([[1,2,100], [3,4,200], [4,5,300]])


rng = np.random.default_rng()
arr = rng.integers(0, 10, size=(2,3,2))
# print(arr)
# print("-----")
# print(arr[:,:,0])



arr_temp = np.array([[1,2,3], [4,5,6]])
# print(arr_temp.max())

print(arr_temp.ravel())
res = arr_temp.ravel()
print(res.reshape(2,3))


print("----")
x = np.array([1,2])
y = np.array([2,3,4,5])

xx, yy = np.meshgrid(x, y)


print(xx.shape)
print(yy)
