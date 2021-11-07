
import anndata as ad
import crocodile.toolbox as tb
import numpy as np

from gcproc import GCProc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

"""The decomposition should be reasonably close.
To increase the accuracy. 
1- manipulate the size of the latent.
2- less data?
"""

root = tb.P.home().joinpath("data/tmp/public")
adata_gex = ad.read_h5ad(root.joinpath("cite/cite_gex_processed_training.h5ad"))
adata_adt = ad.read_h5ad(root.joinpath("cite/cite_adt_processed_training.h5ad"))

X = adata_gex.X[:6000, :100].toarray()
Y = adata_adt.X[:6000, :].toarray()

ground_truth = np.copy(Y[-100:, ...])

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X[:-100], Y[:-100])
Y[-100:, ...] = knn.predict(X[-100:])


s = GCProc(verbose=False)
s.fit([X, Y])


# ================= Test Prediction =============================
def subs_func(array):
    return array[-100:]


for i in range(10):
    y_hat = s.recover(predict_idx=1, subs_func=subs_func, mode="internal")
    mse = mean_squared_error(y_hat, ground_truth)
    print(f"{mse=}")
    Y[-100:, ...] = y_hat
    s.fit([X, Y])

    y_hat = s.recover(predict_idx=1, mode="external")
    mse = mean_squared_error(y_hat[-100:], ground_truth)
    print(f"{mse=}")
    s.fit([X, y_hat])



