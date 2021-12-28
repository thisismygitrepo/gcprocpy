
import anndata as ad
import crocodile.toolbox as tb
import numpy as np

from src.gcproc.gcproc import GCProc
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

"""The decomposition should be reasonably close.
To increase the accuracy. 
1- manipulate the size of the latent.
2- less data?
"""


if __name__ == '__main__':
    root = tb.P.home().joinpath("data/tmp/public")
    adata_gex = ad.read_h5ad(root.joinpath("cite/cite_gex_processed_training.h5ad"))
    adata_adt = ad.read_h5ad(root.joinpath("cite/cite_adt_processed_training.h5ad"))

    # Generating the scene to test prediction performance.
    X = adata_gex.X[:6000, :100].toarray()
    Y = adata_adt.X[:6000, :].toarray()

    ground_truth = np.copy(Y[-100:, ...])

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X[:-100], Y[:-100])
    Y[-100:, ...] = knn.predict(X[-100:])
    # =================================================

    gcproc = GCProc(verbose=False, max_iter=1)
    gcproc.fit([X, Y])
    gcproc.max_iter = 1

    # ================= Test Prediction =============================

    def test_prediction():

        def subs_func(array):
            return array[:-100]

        for i in range(10):
            y_hat = gcproc.recover(predict_idx=1, subs_func=subs_func, mode="internal")
            y_hat_test = y_hat[-100:]
            mse = mean_squared_error(y_hat_test, ground_truth)
            print(f"{mse=}")
            gcproc.fit([X, y_hat], code=gcproc.code, encode=gcproc.encode, keep_params=True)

            y_hat = gcproc.recover(predict_idx=1, mode="external")
            mse = mean_squared_error(y_hat[-100:], ground_truth)
            print(f"{mse=}")
            gcproc.fit([X, y_hat], code=gcproc.code, encode=gcproc.encode, keep_params=True)

