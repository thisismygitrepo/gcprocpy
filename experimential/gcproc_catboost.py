
import numpy as np
import matplotlib.pyplot as plt
import crocodile.toolbox as tb
from gcproc import GCProc

"""
* GCProc (randomly initialized) comes first to consume the data and do `dimensionality reduction`.
* CatBoost takes in the outcome and spits out a solution.
* Another GCProc (different init) instance enhances the outcome.
"""

data = None
i_dim = 100  # But only for learning purpose, eventually, alpha will NOT process the rows.
j_dim = 50  # as requested by CatBoost.

gc1 = GCProc(i_dim=i_dim, j_dim=j_dim)
gc1.fit(data)
# create fit trasnform to return dim reduced


if __name__ == '__main__':
    pass
