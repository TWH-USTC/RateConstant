import numpy as np
import torch
import gpytorch
from gpytorch.constraints import Interval
from tqdm import tqdm
import math


def update_lr_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_lr_epoch(optimizer, epoch, total_epochs=100, base_lr=1e-3, min_lr=5e-5, warmup_epochs=10):
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    epoch = max(0, min(epoch, total_epochs))

    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr = base_lr * (epoch / float(warmup_epochs))
    else:
        t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        t = min(max(t, 0.0), 1.0)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def standardize_y(y: torch.Tensor, eps: float = 1e-8):
    mean = y.mean()
    std = y.std().clamp_min(eps)
    return (y - mean) / std, mean, std


def load_gpytorch_checkpoint(path, X_train_like, y_train_like, device=None):
    ckpt = torch.load(path, map_location="cpu")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_like = torch.as_tensor(X_train_like, dtype=torch.float32, device=device)
    y_train_like = torch.as_tensor(y_train_like, dtype=torch.float32, device=device).view(-1)

    # 重建 likelihood 和 model（结构必须与训练时一致）
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPOnEmbeddings(X_train_like, y_train_like, likelihood).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])

    y_mean = float(ckpt["y_mean"])
    y_std  = float(ckpt["y_std"])

    model.eval()
    likelihood.eval()
    return model, likelihood, y_mean, y_std, ckpt.get("extra", {})


def save_gpytorch_checkpoint(path, model, likelihood, y_mean, y_std, extra=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "y_mean": float(y_mean.detach().cpu().item()) if torch.is_tensor(y_mean) else float(y_mean),
        "y_std":  float(y_std.detach().cpu().item())  if torch.is_tensor(y_std)  else float(y_std),
        "D": model.train_inputs[0].shape[-1],  # 特征维度（便于校验/重建）
        "extra": extra or {},                  # 可选：保存iters/lr/seed等
    }
    torch.save(ckpt, path)


def build_covar_and_likelihood(D: int):
    ls_c = Interval(1e-3, 1e2)      # lengthscale bounds
    var_c = Interval(1e-3, 1e2)     # LinearKernel variance bounds
    noise_c = Interval(1e-6, 1e0)   # noise bounds

    matern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_constraint=ls_c))

    rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=D, lengthscale_constraint=ls_c))

    # 额外加一条显式线性核（可让线性部分单独学习尺度）
    lin = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(variance_constraint=var_c),) # outputscale_constraint=os_c

    covar_module = matern + rbf + lin

    # WhiteKernel -> likelihood noise
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_c)

    # 初始化（对齐你 sklearn 的初值）
    matern.initialize(outputscale=1.0)
    matern.base_kernel.initialize(lengthscale=torch.ones(1, D))

    rbf.initialize(outputscale=1.0)
    rbf.base_kernel.initialize(lengthscale=torch.ones(1, D))

    lin.initialize(outputscale=1.0)
    lin.base_kernel.initialize(variance=1.0)

    likelihood.initialize(noise=1e-3)

    return covar_module, likelihood


class ExactGPOnEmbeddings(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        D = train_x.size(-1)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module, _ = build_covar_and_likelihood(D)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_and_predict_gpytorch(
    embeddings_train, labels_train,
    embeddings_test,
    iters=100,
    lr=0.05,
    seed=42,
    device=None,
    save_path=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- numpy -> torch ---
    X_train = torch.as_tensor(embeddings_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(labels_train, dtype=torch.float32, device=device).view(-1)
    X_test  = torch.as_tensor(embeddings_test,  dtype=torch.float32, device=device)

    y_train_s, y_mean, y_std = standardize_y(y_train)
    # likelihood（噪声别省；可以初始化一个较小噪声）
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.noise_covar.initialize(noise=1e-3)

    model = ExactGPOnEmbeddings(X_train, y_train_s, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in tqdm(range(iters)):
        optimizer.zero_grad(set_to_none=True)
        output = model(X_train)
        loss = -mll(output, y_train_s)
        update_lr_epoch(optimizer, i+1, iters, base_lr=lr)
        loss.backward()
        optimizer.step()

    # --- 预测：返回 mean/std（在标准化空间）---
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(X_test))
        mean_s = pred_dist.mean          # 标准化后的均值
        std_s = pred_dist.stddev        # 标准化后的标准差

    mean = mean_s * y_std + y_mean
    std = std_s * y_std

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist_train = likelihood(model(X_train))
        mean_s_train = pred_dist_train.mean  # 标准化后的均值
        std_s_train = pred_dist_train.stddev  # 标准化后的标准差

    mean_train = mean_s_train * y_std + y_mean
    std_train = std_s_train * y_std

    if save_path is not None:
        save_gpytorch_checkpoint(
            save_path,
            model, likelihood,
            y_mean=y_mean, y_std=y_std,
            extra={"iters": iters, "lr": lr, "seed": seed}
        )

    return mean_train.detach().cpu().numpy(), std_train.detach().cpu().numpy(),\
           mean.detach().cpu().numpy(), std.detach().cpu().numpy(), model, likelihood


