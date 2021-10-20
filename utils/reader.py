
import anndata as ad
import crocodile.toolbox as tb


root = tb.P.home().joinpath("data/tmp/public")
root.tree(stats=True)

adata_gex = ad.read_h5ad(root.joinpath("cite/cite_gex_processed_training.h5ad"))
adata_adt = ad.read_h5ad(root.joinpath("cite/cite_adt_processed_training.h5ad"))

x = adata_adt


if __name__ == '__main__':
    pass
