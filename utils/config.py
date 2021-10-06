import crocodile.toolbox as tb

# ' Extract configuration parameters of gcproc
# '
# ' @param i_dim Dimension reduction for samples (assumed to be along rows)
# ' @param j_dim Dimension reduction for features (assumed to be along columns)
# ' @param min_iter Minimum iteration of gcproc
# ' @param max_iter Maximum iteration of gcproc
# ' @param tol Tolerance threshold for convergence (metric: Root Mean Squared Error)
# ' @param verbose Print statements?
# ' @param init Initialisation method for the model ("random","eigen-quick","eigen-dense","svd-quick","svd-dense")


CONFIG = tb.Struct(
    i_dim=30,
    j_dim=30,
    min_iter=2,
    max_iter=350,
    tol=1,
    verbose=True,
    init="random",
    seed=1)

# ' Extract anchor framework to put into gcproc
# ' Anchors allow the transfer of learned parameters from a pre-trained model.
# ' @param code Anchor the code
transfer = tb.Struct(E=None, D=None)  # D is Decode and E is Encode

task = ["regression"]  # ["classification","imputation"]
method = [["matrix.projection", "knn"]]  # per task (list of lists)
design = None  # specifies what to predict (i.e. where are the missing values).
               # It should be list with same length as data list (passed to the gcproc)


# Join data to improve modelling interpretability for similar axes ' @param alpha Joining the alpha parameters. A
# vector of integers, where identical integers indicate same the data axis to be joined. Axes that should not be
# shared are given NA. ' @param beta Joining the beta parameters. A vector of integers, where identical integers
# indicate same the data axis to be joined. Axes that should not be shared are given NA.

join = tb.Struct(alpha=None, beta=None)


if __name__ == '__main__':
    pass
