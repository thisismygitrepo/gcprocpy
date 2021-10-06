
"""
Optionally, use this to normalize.
"""

import numpy as np


def prep(x, log=False, center=False, scale=False, bias=1):
    if log:
        x = np.log(x + bias)

    if center:
        x -= np.mean(x, axis=0)  # assumes that the layout is [samples, features]

    if scale:
        x = x / np.linalg.norm(x, ord="fro") * np.prod(x.shape)

    return x

if __name__ == '__main__':
    pass
