
import numpy as np

"""`pinv` function gives the left inverse, if the matrix is tall and skinny.
The right inverse if the matrix is wide and fat.
"""
a = np.random.randn(10, 3)
a_dagger = np.linalg.pinv(a)
a_dagger2 = np.linalg.inv(a.T @ a) @ a.T
assert np.allclose(a_dagger, a_dagger2)

a = np.random.randn(3, 10)
a_dagger3 = np.linalg.pinv(a)
a_dagger4 = a.T @ np.linalg.inv(a @ a.T)
assert np.allclose(a_dagger3, a_dagger4)

