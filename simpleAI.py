import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Check array shapes
print(arr_1d.shape)  # (5,)
print(arr_2d.shape)  # (2, 3)


# Element-wise addition
result = arr_1d + 10
print(result)  # [11 12 13 14 15]

# Matrix multiplication
matrix_result = np.dot(arr_2d, arr_2d.T)  # Transpose arr_2d for proper multiplication
print(matrix_result)
