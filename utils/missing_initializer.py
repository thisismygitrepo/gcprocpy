
import numpy as np
import crocodile.toolbox as tb
from sklearn.impute import KNNImputer


imputer = KNNImputer(n_neighbors=5)
# imputer.fit_transform(X)

if __name__ == '__main__':
    pass
