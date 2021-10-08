
import anndata as ad
import crocodile.toolbox as tb
from gcproc import GCProc

root = tb.P.home().joinpath("data/tmp/public")
adata_gex = ad.read_h5ad(root.joinpath("cite/cite_gex_processed_training.h5ad"))
# adata_adt = ad.read_h5ad("cite/cite_adt_processed_training.h5ad")

X = adata_gex.X[:1000, :1000]
s = GCProc()
s.gcproc([X])
