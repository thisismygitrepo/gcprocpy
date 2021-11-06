
import anndata as ad
import crocodile.toolbox as tb
from gcproc import GCProc
import matplotlib.pyplot as plt

"""The decomposition should be reasonably close.
To increase the accuracy. 
1- manipulate the size of the latent.
2- less data?
"""

root = tb.P.home().joinpath("data/tmp/public")
adata_gex = ad.read_h5ad(root.joinpath("cite/cite_gex_processed_training.h5ad"))
# adata_adt = ad.read_h5ad("cite/cite_adt_processed_training.h5ad")

X = adata_gex.X[:6000, :100].toarray()
s = GCProc()
s.fit([X])

