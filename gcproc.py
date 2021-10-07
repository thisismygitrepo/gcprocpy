
from utils.parameter_intialization import ParamInit
import numpy as np
import crocodile.toolbox as tb
from numpy.linalg import pinv


class GCProc(tb.Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ====================== Model Configuration ====================================
        self.i_dim = 30
        self.j_dim = 30
        self.param_init = ParamInit.random_normal  # any method from this class is okay.
        # ============== Numerical Specs
        self.norm = tb.Struct(log=False, center=False, scale=False, bias=1)
        self.seed = 1  # for reproducibility of random initializations.
        self.verbose = True

        # ============== Optimizer Specs ============================
        self.min_iter = 2
        self.max_iter = 350
        self.count = 1  # iteration counter.
        self.score = []  # history of performance of mae.
        self.score_lag = 2  # How many previous scores kept track of?
        self.accept_score = 1  # How many scores used to calculate previous and current mean score
        self.recovered = None
        self.tol = 1e-3

        # ================== Configuration of Tasks and Inputs (Recover) ====================================
        self.task = ["regression"]  # ["classification","imputation"]
        self.method = ["matrix.projection"]  # per task (list of lists). knn another choice.
        self.design = None  # specifies what to predict (i.e. where are the missing values).
        # It should be list with same length as data list (passed to the gcproc)
        self.join = tb.Struct(alpha=None, beta=None)  #
        # alpha = [1, 1, None]  # use None to signify unique alpha or beta.

        # ================ Transfer pretrained latents (output to be) ===============================
        self.encode = None  # common for all datasets.
        self.prev_encode = None  # for convergence test.
        self.code = None  # common for all datasets.

    def init_single_dataset(self, x, update_encode_code=False, idx=None):
        x = self.prepare(x)
        alpha, beta = self.param_init(self, x)
        if update_encode_code or self.encode is None:
            self.encode = alpha @ x @ beta
            self.code = pinv(alpha @ alpha.T) @ self.encode @ pinv(beta.T @ beta)
        return tb.Struct(x=x, alpha=alpha, beta=beta, idx=idx)

    def gcproc(self, data_list):
        np.random.seed(self.seed)
        data = [self.init_single_dataset(data, idx=idx) for idx, data in enumerate(data_list)]
        while True:
            self.count += 1
            for idx, d in enumerate(data):
                self.update_set(d)

                # ====================== Joining ===============================
                indices = np.argwhere(self.join.alpha == self.join.alpha[idx])
                for tmp in data[indices]:
                    tmp.alpha = d.alpha
                indices = np.argwhere(self.join.beta == self.join.beta[idx])
                for tmp in data[indices]:
                    tmp.beta = d.beta

            mae = np.mean(abs(self.prev_encode - self.encode))
            self.score.append(mae)

            mae_current = np.mean(self.score[-self.score_lag:])
            mae_prev = np.mean(self.score[-self.score_lag - 1:-1])  # TODO: generalize this.

            if self.count < self.min_iter and abs(mae_current - mae_prev) < self.tol:
                pass
            else:
                pass



    def update_set(self, d):
        beta_dot = d.beta @ pinv(d.beta.T @ d.beta)
        code_dot_beta = self.code.T @ pinv(self.code @ self.code.T)
        d.alpha = d.x @ beta_dot @ code_dot_beta

        alpha_dot = pinv(d.alpha @ d.alpha.T) @ d.alpha
        code_dot_alpha = pinv(self.code.T @ self.code) @ self.code.T
        d.beta = code_dot_alpha @ alpha_dot @ d.x

        self.prev_encode, self.encode = self.encode, d.alpha @ d.x @ d.beta
        self.code = pinv(d.alpha @ d.alpha.T) @ self.encode @ pinv(d.beta @ d.beta.T)
        return d

    def prepare(self, x):
        """ Based on this function: https://github.com/AskExplain/gcproc/blob/main/R/prepare_data.R
        :return:
        """
        if self.norm.log:
            x = np.log(x + self.norm.bias)
        if self.norm.center:
            x -= np.mean(x, axis=0)  # assumes that the layout is [samples, features]
        if self.norm.scale:
            x = x / np.linalg.norm(x, ord="fro") * np.prod(x.shape)
        return x


if __name__ == '__main__':
    pass
