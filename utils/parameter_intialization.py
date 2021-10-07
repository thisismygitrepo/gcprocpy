
import numpy as np
import matplotlib.pyplot as plt
import crocodile.toolbox as tb


class ParamInit:
    @staticmethod
    def svd_quick(config, x):
        pass

    @staticmethod
    def svd_dense(config, x):
        pass

    @staticmethod
    def random_normal(config, x):
        alpha = np.random.normal(size=(config.i_dim, x.shape[0]))
        beta = np.random.normal(size=(x.shape[1], config.j_dim))
        return alpha, beta  # pivot_sample & pivot_feature

    @staticmethod
    def eigen_quick(config, x):
        pass

    @staticmethod
    def eigen_dense(config, x):
        pass


if __name__ == '__main__':
    pass
