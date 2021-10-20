
from utils.parameter_intialization import ParamInit
import numpy as np
import crocodile.toolbox as tb
from numpy.linalg import pinv


class GCProc(tb.Base):
    def __init__(self, i_dim=30, j_dim=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ====================== Model Configuration ====================================
        self.i_dim = i_dim  # size of the latent space.
        self.j_dim = j_dim  # size of the latent space.
        self.param_init = ParamInit.random_normal  # initializes alpha and beta.

        # ===================== Numerical Specs =========================================
        self.norm = tb.Struct(log=False, center=False, scale=False, bias=1)
        self.seed = 1  # for reproducibility of random initializations.
        self.verbose = True

        # ============== Optimizer Specs ================================================
        self.max_iter = 350
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
        self.code = None  # common for all datasets. alpha_x Ex beta_x == alpha_y Y beta_y = z
        self.prev_code = None  # for convergence test.

    def init_single_dataset(self, x, update_encode_code=False, idx=None):
        x = self.prepare(x)
        alpha, beta = self.param_init(self, x)
        if update_encode_code or self.code is None:
            self.code = alpha @ x @ beta
        return tb.Struct(x=x, alpha=alpha, beta=beta, recovered=pinv(alpha) @ self.code @ pinv(beta), idx=idx)

    def check_convergenence(self) -> bool:
        if self.count < self.score_batch_size:
            return False  # too few iterations.
        else:
            mae_avg = np.mean(self.score[-self.score_batch_size:-1])
            return mae_avg < self.converg_thresh

    def gcproc(self, data_list):
        np.random.seed(self.seed)
        self.data = [self.init_single_dataset(data, idx=idx) for idx, data in enumerate(data_list)]
        data = self.data
        while True:
            self.count += 1
            for idx, d in enumerate(data):
                self.update_set(d)
                self.join_params(data, idx, d)

            mae = np.mean(abs(self.prev_code - self.code))
            self.score.append(mae)
            if self.check_convergenence():
                break
            if self.verbose:
                d = data[0]
                print(f"Iteration #{self.count:3}. Loss = {np.square(d.x - d.recovered).sum():12.0f}")

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
        d.alpha = (d.x @ d.beta @ pinv(self.code)).T  # update alpha using backward model (cheap)
        d.beta = (pinv(self.code) @ d.alpha @ d.x).T  # update beta using backward model (cheap)
        self.prev_code, self.code = self.code, d.alpha @ d.x @ d.beta  # update z using forward model.
        d.recovered = self.recover(d)
        return d

    def recover(self, d):  # reconstruct from a, b & z.
        return pinv(d.alpha) @ self.code @ pinv(d.beta)

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
