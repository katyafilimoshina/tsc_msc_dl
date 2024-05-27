import jax
from flax import struct
from jax import numpy as jnp
from scipy.stats import gaussian_kde
import ot
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import trange



def compute_metrics(
    xs_true,
    xs_pred,
    name=None,
    n_samples=1000,
    n_steps = 50,
    scale=1.0,
    trunc_chain_len=None,
    ess_rar=1,
):
    """
    Calculates metrics: 
    ESS (Effective sample size), 
    EMD (Earth mover’s distance), 
    ESTV mean and std (Empirical sliced total variation distance)
    """

    torch.manual_seed(926)
    metrics = dict()
    key = jax.random.PRNGKey(0)
    
    # ESS
    ess = ESS(
        acl_spectrum(
            xs_pred[::ess_rar] - xs_pred[::ess_rar].mean(0)[None, ...],
        ),
    ).mean()
    metrics["ess"] = ess

    # ESTV
    xs_pred = xs_pred[-trunc_chain_len:]
    try:
      tracker = average_total_variation(
          key,
          xs_true,
          xs_pred,
          n_steps=n_steps,
          n_samples=n_samples,
      )
      metrics["tv_mean"] = tracker.mean()
      metrics["tv_conf_sigma"] = tracker.std_of_mean()
      mean = tracker.mean()
      std = tracker.std()

      # EMD
      metrics["emd"] = 0
      for b in range(xs_pred.shape[1]):
          M = ot.dist(xs_true / scale, xs_pred[:, b,:] / scale)
          emd = ot.lp.emd2([], [], M, numItermax = 1e6)
          metrics["emd"] += emd / xs_pred.shape[1]

      # Print results
      mean = metrics["tv_mean"]
      std = metrics["tv_conf_sigma"]
      ess = metrics["ess"]
      emd = metrics["emd"]

      if name is not None:
          print(f"===={name}====")
      print(
          f"TV distance. Mean: {mean:.3f}, Std: {std:.3f}. \nESS: {ess:.3f} \nEMD: {emd:.3f}",
      )

    except:
      print("During this try, only one distinct point is generated.")

    return metrics



def plot_metrics(metrics_list):
  """
  Plots metrics (ESS, ESTV, EMD) values history from training
  """
  num_of_iters = [d['num_of_iter'] for d in metrics_list]
  ess_values = [d['ess'] for d in metrics_list]
  emd_values = [d['emd'] for d in metrics_list]
  tv_mean_values = [d['tv_mean'] for d in metrics_list]
  tv_std_values = [d['tv_conf_sigma'] for d in metrics_list]

  ess_values = np.nan_to_num(ess_values, nan=0.0)
  emd_values = np.nan_to_num(emd_values, nan=0.0)
  tv_mean_values = np.nan_to_num(tv_mean_values, nan=0.0)
  tv_std_values = np.nan_to_num(tv_std_values, nan=0.0)

  fig, axs = plt.subplots(2, 2, figsize=(12, 8))
  
  # Plot ESS
  axs[0, 0].plot(num_of_iters, ess_values, marker='o')
  axs[0, 0].set_title('ESS vs Num of training iterations')
  axs[0, 0].set_xlabel('Number of training iterations')
  axs[0, 0].set_ylabel('ESS')
  
  # Plot EMD
  axs[0, 1].plot(num_of_iters, emd_values, marker='o')
  axs[0, 1].set_title('EMD vs Num of training iterations')
  axs[0, 1].set_xlabel('Number of training iterations')
  axs[0, 1].set_ylabel('EMD')
  
  # Plot TV Mean
  axs[1, 0].plot(num_of_iters, tv_mean_values, marker='o')
  axs[1, 0].set_title('TV Mean vs Num of training iterations')
  axs[1, 0].set_xlabel('Number of training iterations')
  axs[1, 0].set_ylabel('TV Mean')
  
  # Plot TV Std
  axs[1, 1].plot(num_of_iters, tv_std_values, marker='o')
  axs[1, 1].set_title('TV Std vs Num of training iterations')
  axs[1, 1].set_xlabel('Number of training iterations')
  axs[1, 1].set_ylabel('TV Std')
  
  plt.tight_layout()
  plt.show()
  




def ESS(A):
    # ess = ESS(acl_spectrum((trunc_sample - trunc_sample.mean(0)[None, ...]))).mean()
    A = A * (A > 0.05)
    ess = 1.0 / (1.0 + 2 * np.sum(A[1:, ...], axis=0))
    return ess


def autocovariance(X, tau=0):
    # dT, dX = np.shape(X)
    dT = X.shape[0]
    s = 0.0
    dN = 1
    if tau > 0:
        x1 = X[:-tau, ...]
    else:
        x1 = X
    x2 = X[tau:, ...]
    s = np.sum(x1 * x2, axis=0) / dN

    return s / (dT - tau)



def acl_spectrum(X, n=150, scale=None):
    scale = (
        np.array(scale)
        if scale is not None
        else np.sqrt(autocovariance(X, tau=0))
    )
    return np.stack(
        [
            autocovariance(X / (scale[None, ...] + 1e-7), tau=t)
            for t in range(n - 1)
        ],
        axis=0,
    )


class MeanTracker:
    def __init__(self):
        self.values = []

    def update(self, value: float) -> None:
        self.values.append(value)

    def __len__(self):
        return len(self.values)

    def mean(self) -> float:
        return jnp.mean(jnp.array(self.values))

    def std(self) -> float:
        return jnp.std(jnp.array(self.values), ddof=1)

    def std_of_mean(self) -> float:
        return jnp.std(jnp.array(self.values)) / jnp.sqrt(len(self))

    def last(self) -> float:
        return self.values[-1]
    

@struct.dataclass
class Projector:
    x0: jnp.ndarray
    v: jnp.ndarray

    def project(self, xs: jnp.ndarray) -> jnp.ndarray:
        return (xs - self.x0[None]) @ self.v


def create_random_projection(key: jnp.ndarray, xs: jnp.ndarray) -> Projector:
    x0 = jnp.mean(xs, 0)
    v = jax.random.normal(key, [len(x0)])
    v = v / jnp.linalg.norm(v)

    return Projector(x0, v)


def total_variation_1d(xs_true, xs_pred, n_samples):
    true_density = gaussian_kde(xs_true)
    pred_density = gaussian_kde(xs_pred)

    x_min = min(xs_true.min(), xs_pred.min())
    x_max = max(xs_true.max(), xs_pred.max())

    points = np.linspace(x_min, x_max, n_samples)

    return (
        0.5
        * np.abs(true_density(points) - pred_density(points)).mean()
        * (x_max - x_min)
    )


def total_variation(
    key: jnp.ndarray,
    xs_true: jnp.ndarray,
    xs_pred: jnp.ndarray,
    n_samples: int,
):
    proj = create_random_projection(key, xs_true)
    return total_variation_1d(
        proj.project(xs_true),
        proj.project(xs_pred),
        n_samples,
    )


def average_total_variation(
    key: jnp.ndarray,
    true: jnp.ndarray,
    other: jnp.ndarray,
    n_samples: int,
    n_steps: int,
) -> MeanTracker:
    tracker = MeanTracker()
    keys = jax.random.split(key, n_steps)

    for b in range(other.shape[1]):
        for i in trange(n_steps, leave=False):
            tracker.update(total_variation(keys[i], true, other[:, b], n_samples))
    return tracker

