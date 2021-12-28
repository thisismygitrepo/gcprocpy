
import matplotlib.pyplot as plt
from utils.parameter_intialization import ParamInit
import numpy as np
import crocodile.toolbox as tb
from numpy.linalg import pinv, inv
from sklearn.preprocessing import StandardScaler


class GCProc(tb.Base):
    def __init__(self, i_dim=30, j_dim=10, max_iter=350, verbose=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ====================== Model Configuration ====================================
        self.i_dim = i_dim  # size of the latent space.
        self.j_dim = j_dim  # size of the latent space.
        self.param_init = ParamInit.random_normal  # initializes alpha and beta.

        # ===================== Numerical Specs =========================================
        self.norm = tb.Struct(log=False, center=False, scale=False, bias=1)
        self.seed = 1  # for reproducibility of random initializations.
        self.verbose = verbose

        # ============== Optimizer Specs ================================================
        self.max_iter = max_iter
        self.count = 1  # iteration counter.
        self.score = []  # history of performance of mae, used by stopping criterion.
        self.score_batch_size = 4
        self.converg_thresh = 1e-3

        # ================== Configuration of Tasks and Inputs (Recover) ====================================
        self.task = ["regression", "classification", "imputation"][0]
        self.method = ["matrix.projection"]  # per task (list of lists). knn another choice.
        self.design = None  # specifies what to predict (i.e. where are the missing values).
        # It should be list with same length as data list (passed to the gcproc)
        self.join = tb.Struct(alpha=[], beta=[])  # which alphas and betas are common among datasets.
        # alpha = [1, 1, None]  # use None to signify unique alpha or beta.

        # ================ Transfer pretrained latents (output to be) ===============================
        self.data = None  # list of datasets passed.
        self.encode = None  # common for all datasets. alpha_x Ex beta_x == alpha_y Y beta_y = ENCODE (1)
        self.code = None  # alpha_d (x) beta_d
        self.prev_encode = None  # for convergence test.

    def save(self, name, save_alpha_beta=False):
        dat = tb.Struct(code=self.code, encode=self.encode)
        if save_alpha_beta:
            # dat.update(alpha=self.alpha, beta=self.beta)
            pass
        dat.save_npy(path=tb.P.home().joinpath(f"GCProcData/{name}"))

    def init_single_dataset(self, x, idx=None):
        x = self.prepare(x)
        alpha, beta = self.param_init(self, x)
        return tb.Struct(x=x, alpha=alpha, beta=beta, recovered=None, idx=idx)

    def init_code_encode(self, code=None, encode=None):
        assert self.data is not None, f"Initialize the parameters first!"
        d = self.data[-1]  # any dataset is good enough.
        if encode is None:  # init from random values.
            self.encode = self.get_encode(d)
        else:
            self.encode = encode
        if code is None:
            self.code = self.get_code(d)
        else:
            self.code = code

    def check_convergenence(self) -> bool:
        if self.count < self.score_batch_size:
            return False  # too few iterations.
        else:
            mae_avg = np.mean(self.score[-self.score_batch_size:])
            if self.count < self.max_iter:
                return mae_avg < self.converg_thresh
            else:
                print(f"Failed to converge before max iteration {self.max_iter}. Latest loss = {mae_avg}")
                return True  # as if convergence check returns True (break!)

    def fit(self, data_list, code=None, encode=None, keep_params=False):
        """
        :param data_list:
        :param code:
        :param encode:
        :param keep_params:
        :return:
        """
        np.random.seed(self.seed)
        if keep_params is False:
            self.data = [self.init_single_dataset(data, idx=idx) for idx, data in enumerate(data_list)]
        else:
            tmp_data = []
            for idx, data in enumerate(data_list):
                dat = tb.Struct(x=data, alpha=self.data[idx].alpha, beta=self.data[idx].beta,
                                recovered=None, idx=idx)
                tmp_data.append(dat)
            self.data = tmp_data

        self.init_code_encode(code=code, encode=encode)
        data = self.data
        while True:
            self.count += 1
            for idx, d in enumerate(data):
                self.update_set(d)
                self.join_params(data, idx, d)

            mae = np.mean(abs(self.prev_encode - self.encode))
            self.score.append(mae)
            if self.check_convergenence():
                break

        if self.verbose:
            # d = data[0]  # one can optionally compute the recovered X for convergenence purpose.
            # d.recovered = self.recover(d)  # Optional.
            # print(f"Iteration #{self.count:3}. Loss = {np.abs(d.x - self.recover(d)).sum():1.0f}")
            self.plot_progress()

    def plot_progress(self):
        fig, ax = plt.subplots()
        ax.semilogy(self.score)
        ax.set_title(f"GCProc Convergence")
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel(f"Encode Mean Absolute Error")

    def fit_transform(self, data_list):
        """Runs the solver `fit` and then transform each of the datasets provided."""
        self.fit(data_list)
        for dat in self.data:
            self.encode(dat)

    def join_params(self, data, idx, d):
        # ====================== Joining ===============================
        if self.join.alpha:
            indices = np.argwhere(self.join.alpha == self.join.alpha[idx])
            for tmp in data[indices]:
                tmp.alpha = d.alpha
        if self.join.beta:
            indices = np.argwhere(self.join.beta == self.join.beta[idx])
            for tmp in data[indices]:
                tmp.beta = d.beta

    def update_set(self, d):
        self.prev_encode, self.encode = self.encode, self.get_encode(d)  # update encode using forward model.
        self.code = self.get_code(d)
        tmp = (self.code @ d.beta.T)
        d.alpha = (d.x @ pinv(tmp)).T  # using eq 2.
        tmp = (d.alpha.T @ self.code)
        d.beta = (pinv(tmp) @ d.x).T  # using eq 2.

        return d

    @staticmethod
    def get_encode(d):  # use eq. 1
        return d.alpha @ d.x @ d.beta

    def get_code(self, d):  # eq. (2) -> eq. (1) and solve for z:
        return inv(d.alpha @ d.alpha.T) @ self.encode @ inv(d.beta.T @ d.beta)

    def recover(self, predict_idx, subs_func=None, mode=["external", "internal"][0]):  # reconstruct x from a, b & z.
        """

        :param predict_idx:
        :param subs_func:
        :param mode:
        :return:
        """
        y = self.data[predict_idx]  # to predict (update).
        covariates = np.zeros(shape=(y.x.shape[0], y.beta.shape[1]))
        for idx in range(len(self.data)):
            if idx != predict_idx:
                d = self.data[idx]
                tmp = y.alpha.T @ self.code @ d.beta.T @ d.beta
                covariates += StandardScaler().fit_transform(tmp) / len(self.data)

        covariates = np.column_stack([np.ones(covariates.shape[0]), covariates])

        if mode == "internal":  # ==> subset
            y_x = subs_func(y.x)
            tmp = subs_func(covariates)
        else:
            y_x = y.x
            tmp = covariates

        k = pinv(tmp) @ y_x  # bootstrapping.
        y_hat = covariates @ k
        return y_hat

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
