import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.linalg import qr as pivoted_qr

CSV_PATH = "/Users/sam/Desktop/ml/Bio-reactor model/bioreactor_pinn_data_noisy.csv"

SEED = 42
N_RUNS = 1              
WARMUP_EPOCHS = 3000
ADAM_EPOCHS = 10000
PRINT_EVERY = 1000
WARM_LOG_EVERY = 10     
N_COL = 2001
WIDTH, DEPTH = 64, 4
EPS = 1e-14

USE_LINEAR_INIT = False     
                            

# ---- Q-DEIM settings ----
T_DIV = 20              # number of non-overlapping time blocks 
EPS_THR = 1e-6          # energy criterion
ADD_DATA_TO_COLLOC = True   # also enforce the ODE at the selected sample times

DATA_LOSS_FIG = "pinn_data_loss_curve.png"
PHYS_LOSS_FIG = "pinn_physics_loss_curve.png"
PARAM_FIG = "pinn_parameter_estimation_curves.png"
QDEIM_FIG = "pinn_qdeim_samples.png"

torch.set_default_dtype(torch.float64)

# Fixed kinetic parameters
FIXED = dict(Pm=50.0, Ki=22.0, mum=0.48, Km=1.2)

# Parameters to estimate
NAMES = ["Yxs", "beta", "alpha", "Sf"]
GUESS = np.array([0, 0, 0, 0])

REFERENCE = np.array([0.40, 0.20, 2.2, 20.0])

# Known initial state [X, S, P]
X0 = np.array([5.9956, 5.0109, 19.1267])


df = pd.read_csv(CSV_PATH).sort_values("time", kind="stable")

t = df["time"].to_numpy(dtype=float)
D = df["D0"].to_numpy(dtype=float)
Y = df[["X", "S", "P"]].to_numpy(dtype=float)

if len(t) < 2:
    raise ValueError("At least two observations are required.")
if not (np.all(np.isfinite(t)) and np.all(np.isfinite(D)) and np.all(np.isfinite(Y))):
    raise ValueError("The input data contain NaN or infinite values.")
if np.any(np.diff(t) <= 0):
    raise ValueError("Time must be strictly increasing (remove duplicate times).")

Nt = len(t)
T0 = t[0]
TSPAN = t[-1] - t[0]

# State and rate normalisation (from the full snapshot matrix)
scale = np.maximum(Y.std(axis=0), 1e-8)
rate = scale * max(abs(D.mean()), 1e-8)

scale_t = torch.tensor(scale)
rate_t = torch.tensor(rate)

Y0 = torch.tensor(X0).reshape(1, 3)
tau_all = torch.tensor(((t - T0) / TSPAN)[:, None])      # all measurement times

T_GRID = torch.tensor(t)
D_GRID = torch.tensor(D)


def D_fn(real_time):
    """Zero-order hold of the supplied D0 schedule."""
    q = real_time.detach().reshape(-1).contiguous()
    idx = (torch.searchsorted(T_GRID, q, right=True) - 1).clamp(0, len(D_GRID) - 1)
    return D_GRID[idx].reshape(real_time.shape)


def qdeim_select(Y, t_div, eps_thr):
    """
    Two-way Q-DEIM on each time block of the (states x time) snapshot matrix.
    Selected observations = (state pivots) x (time pivots) per block.
    Returns a boolean mask of shape (Nt, 3) and the rank used in each block.
    """
    U = ((Y - Y.mean(axis=0)) / scale).T                  
    n, m = U.shape
    mask = np.zeros((m, n), dtype=bool)
    ranks = []

    for cols in np.array_split(np.arange(m), t_div):
        Z, S, Vt = np.linalg.svd(U[:, cols], full_matrices=False)
        tail = 1.0 - np.cumsum(S) / S.sum()               
        r = int(np.argmax(tail < eps_thr)) + 1            
        Zr, Vtr = Z[:, :r], Vt[:r, :]

        _, _, px = pivoted_qr(Zr.T, mode="economic", pivoting=True)   
        _, _, pt = pivoted_qr(Vtr, mode="economic", pivoting=True)    

        ix = np.sort(px[:r])
        it = np.sort(cols[pt[:r]])                       
        mask[np.ix_(it, ix)] = True
        ranks.append(r)

    return mask, ranks


T_DIV = max(1, min(T_DIV, Nt // 3))                       
obs_mask, ranks = qdeim_select(Y, T_DIV, EPS_THR)

rows = np.where(obs_mask.any(axis=1))[0] # selected time indices
tau_d = tau_all[rows] # selected sample times
Yd = torch.tensor(Y[rows]) # measurements at those times
Md = torch.tensor(obs_mask[rows].astype(float))# which states are observed

N_SAMPLES = int(obs_mask.sum())
print(f"Q-DEIM: t_div={T_DIV}, eps_thr={EPS_THR:g}, ranks per block={ranks}\n"
      f"        {len(rows)} sample times, N={N_SAMPLES} values "
      f"({100 * N_SAMPLES / Y.size:.1f}% of {Y.size} measurements)\n")

# Collocation points
tau_c = torch.linspace(0.0, 1.0, N_COL).reshape(-1, 1)
if ADD_DATA_TO_COLLOC:
    tau_c = torch.cat([tau_c, tau_d])


#Parameters
LOWER = np.array([0.05, 1e-4, 0.01, X0[1] + 1e-3])
UPPER = np.array([1.50, 1.50, 10.0, 100.0])
LO, HI = np.log(LOWER), np.log(UPPER)


class Params(nn.Module):


    def __init__(self, guess):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(4))
        self.register_buffer("lo", torch.tensor(LO))
        self.register_buffer("hi", torch.tensor(HI))
        self.set(guess)

    def set(self, values):
        v = np.clip(values, LOWER * 1.001, UPPER * 0.999)
        f = np.clip((np.log(v) - LO) / (HI - LO), 1e-3, 1 - 1e-3)
        with torch.no_grad():
            self.raw.copy_(torch.tensor(np.log(f / (1 - f))))

    def forward(self):
        return torch.exp(self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw))


# NETWORK
class Net(nn.Module):
    """PINN with the known initial condition enforced exactly."""

    def __init__(self):
        super().__init__()
        layers, n_in = [], 1
        for _ in range(DEPTH):
            layers += [nn.Linear(n_in, WIDTH), nn.Tanh()]
            n_in = WIDTH
        layers.append(nn.Linear(n_in, 3))
        self.f = nn.Sequential(*layers)

    def forward(self, tau):
        tau0 = torch.zeros((1, 1), dtype=tau.dtype, device=tau.device)
        return Y0 + scale_t * (self.f(tau) - self.f(tau0))


# BIOREACTOR ODE 
def mu_fn(S, P):
    Sp = S.clamp_min(0.0)
    return (FIXED["mum"] * (1.0 - P / FIXED["Pm"]) * Sp
            / (FIXED["Km"] + Sp + Sp**2 / FIXED["Ki"]))


def rhs(y, d, p):
    Yxs, beta, alpha, Sf = p
    X, S, P = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    mu = mu_fn(S, P)

    dX = (mu - d) * X
    dS = d * (Sf - S) - mu * X / Yxs
    dP = -d * P + (alpha * mu + beta) * X
    return torch.cat([dX, dS, dP], dim=1)


def state_and_derivative(net, tau):
    """Network states and d/dt (real time) at the given tau points."""
    tau = tau.clone().requires_grad_(True)
    y = net(tau)
    grads = [torch.autograd.grad(y[:, j].sum(), tau, create_graph=True)[0]
             for j in range(3)]
    return y, torch.cat(grads, dim=1) / TSPAN


# loss
def data_loss(net):
    """Masked MSE on the Q-DEIM-selected observations only."""
    res = (net(tau_d) - Yd) / scale_t
    return torch.sum(Md * res**2) / Md.sum()


def phys_loss(net, p):
    y, dydt = state_and_derivative(net, tau_c)
    d = D_fn(T0 + TSPAN * tau_c)

    loss = torch.mean(((dydt - rhs(y, d, p)) / rate_t) ** 2)

    # Known initial steady state: rhs(Y0) = 0 at the first D0 value
    d0 = D_GRID[0].reshape(1, 1)
    loss = loss + torch.mean((rhs(Y0, d0, p) / rate_t) ** 2)
    return loss


def objective(net, gp):
    ld = data_loss(net)
    lp = phys_loss(net, gp())
    return torch.log(ld + EPS) + torch.log(lp + EPS), ld, lp


def linear_init(net):
    """
    The ODEs are linear in the unknowns, given states and derivatives:
        dS + d*S = d*Sf - (mu*X) * (1/Yxs)
        dP + d*P = alpha*(mu*X) + beta*X
    Solve both by least squares using the warmed-up network.
    Returns [Yxs, beta, alpha, Sf] or None if it fails.
    Only used when USE_LINEAR_INIT = True.
    """
    y, dydt = state_and_derivative(net, tau_c)
    d = D_fn(T0 + TSPAN * tau_c)
    X, S, P = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    muX = mu_fn(S, P) * X

    def to_np(a):
        return a.detach().cpu().numpy().ravel()

    d, S, P, X, muX = map(to_np, (d, S, P, X, muX))
    dS, dP = to_np(dydt[:, 1:2]), to_np(dydt[:, 2:3])
    rS, rP = rate[1], rate[2]

    A_S = np.column_stack([d, -muX]) / rS
    b_S = (dS + d * S) / rS
    Sf, theta = np.linalg.lstsq(A_S, b_S, rcond=None)[0]

    A_P = np.column_stack([muX, X]) / rP
    b_P = (dP + d * P) / rP
    alpha, beta = np.linalg.lstsq(A_P, b_P, rcond=None)[0]

    if not np.isfinite([Sf, theta, alpha, beta]).all() or theta <= 1e-6:
        return None
    return np.array([1.0 / theta, beta, alpha, Sf])


#train
def train(run_id, seed):
    torch.manual_seed(seed)
    net, gp = Net(), Params(GUESS)


    warm = {"iter": [], "data": [], "physics": []}
    p_fixed = gp().detach()                      
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for w in range(WARMUP_EPOCHS):
        opt.zero_grad(set_to_none=True)
        ld = data_loss(net)

        if w % WARM_LOG_EVERY == 0 or w == WARMUP_EPOCHS - 1:
            lp = phys_loss(net, p_fixed)        
            warm["iter"].append(w)
            warm["data"].append(ld.item())
            warm["physics"].append(lp.item())
            if w % PRINT_EVERY == 0:
                print(f"[run {run_id}] warm-up {w:6d} | data {ld.item():.3e} | "
                      f"phys@GUESS {lp.item():.3e}")

        ld.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
    print(f"[run {run_id}] warm-up done   | data {data_loss(net).item():.3e}")

    if USE_LINEAR_INIT:
        cand = linear_init(net)
        if cand is not None:
            cand = np.clip(cand, LOWER * 1.001, UPPER * 0.999)
            l_guess = phys_loss(net, torch.tensor(GUESS)).item()
            l_cand = phys_loss(net, torch.tensor(cand)).item()
            if l_cand < l_guess:
                gp.set(cand)
            print(f"[run {run_id}] linear init: {np.round(cand, 4)} "
                  f"(physics {l_cand:.2e} vs guess {l_guess:.2e}) -> "
                  f"{'used' if l_cand < l_guess else 'rejected'}")
    else:
        print(f"[run {run_id}] starting parameters from GUESS: {np.round(GUESS, 4)}")

    opt = torch.optim.Adam([
        {"params": net.parameters(), "lr": 1e-3},
        {"params": gp.parameters(), "lr": 3e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=ADAM_EPOCHS, eta_min=1e-5)

    hist = {"epoch": [], "objective": [], "data": [], "physics": [], "params": []}
    best_loss, best_state, best_epoch = np.inf, None, 0

    for ep in range(ADAM_EPOCHS + 1):
        opt.zero_grad(set_to_none=True)
        total, ld, lp = objective(net, gp)

        if not (torch.isfinite(total) and torch.isfinite(ld) and torch.isfinite(lp)):
            print(f"[run {run_id}] non-finite loss at epoch {ep}; stopping this run.")
            break

        hist["epoch"].append(ep)
        hist["objective"].append(total.item())
        hist["data"].append(ld.item())
        hist["physics"].append(lp.item())
        hist["params"].append(gp().detach().cpu().numpy().copy())

        if total.item() < best_loss:
            best_loss, best_epoch = total.item(), ep
            best_state = (copy.deepcopy(net.state_dict()),
                          copy.deepcopy(gp.state_dict()))

        if ep % PRINT_EVERY == 0:
            p = hist["params"][-1]
            print(f"[run {run_id}] epoch {ep:6d} | obj {total.item():8.3f} | "
                  f"data {ld.item():.3e} | phys {lp.item():.3e} | "
                  + " ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, p)))

        if ep == ADAM_EPOCHS:
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(net.parameters()) + list(gp.parameters()), 10.0)
        opt.step()
        sched.step()

    if best_state is None:
        return None

    net.load_state_dict(best_state[0])
    gp.load_state_dict(best_state[1])
    return dict(params=gp().detach().cpu().numpy(), loss=best_loss,
                epoch=best_epoch, hist=hist, warm=warm, net=net)

runs = []
for r in range(N_RUNS):
    res = train(r + 1, SEED + r)
    if res is not None:
        runs.append(res)
    print()

if not runs:
    raise RuntimeError("All training runs failed.")

print("Per-run results (best objective, parameters):")
for i, res in enumerate(runs, 1):
    print(f"  run {i}: obj {res['loss']:.3f} | "
          + " ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, res['params'])))

best = min(runs, key=lambda r: r["loss"])
p_est = best["params"]
best_epoch = best["epoch"]
hist = best["hist"]
warm = best["warm"]
best_net = best["net"]

report = pd.DataFrame({
    "Parameter": NAMES,
    "Initial": np.asarray(hist["params"][0]),
    "Estimate": p_est,
    "Reference": REFERENCE,
    "Error_%": 100.0 * np.abs(p_est - REFERENCE) / np.abs(REFERENCE),
})

print("\nEstimated bioreactor parameters (best run)")
print(report.to_string(index=False, formatters={
    "Initial": "{:.6f}".format,
    "Estimate": "{:.6f}".format,
    "Reference": "{:.6f}".format,
    "Error_%": "{:.4f}".format,
}))
print(f"\nBest checkpoint: Adam epoch {best_epoch}")

with torch.no_grad():
    Y_pred = best_net(tau_all).cpu().numpy()
rel_l2 = np.linalg.norm(Y_pred - Y, axis=0) / np.linalg.norm(Y, axis=0)
print("Relative L2 error vs all noisy measurements: "
      + " ".join(f"{n}={e:.3e}" for n, e in zip(["X", "S", "P"], rel_l2)))

sq = ((Y_pred - Y) / scale) ** 2
mse_all = float(sq.mean())
mse_sel = float(sq[obs_mask].mean())
mse_unsel = float(sq[~obs_mask].mean())
print(f"Normalised MSE | selected samples: {mse_sel:.4e} | "
      f"unselected measurements: {mse_unsel:.4e} | all: {mse_all:.4e}")


W = WARMUP_EPOCHS
warm_x = np.asarray(warm["iter"])                      
joint_x = W + np.asarray(hist["epoch"])                

plt.figure(figsize=(9, 5))
plt.semilogy(warm_x, np.maximum(warm["data"], EPS), color="C0", linewidth=1.3,
             label="Data loss (warm-up)")
plt.semilogy(joint_x, np.maximum(hist["data"], EPS), color="C2", linewidth=1.3,
             label="Data loss (joint training)")
plt.axhline(mse_all, color="gray", linestyle="--",
            label=f"Noise floor estimate (error on all measurements = {mse_all:.3f})")
plt.axvline(W, color="gray", linestyle=":", label="Warm-up -> joint training")
plt.axvline(W + best_epoch, color="black", linestyle=":", label="Selected checkpoint")
plt.xlabel("Iteration (warm-up, then Adam)")
plt.ylabel("Data loss (log scale)")
plt.title("Data loss")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(DATA_LOSS_FIG, dpi=200, bbox_inches="tight")

plt.figure(figsize=(9, 5))
plt.semilogy(warm_x, np.maximum(warm["physics"], EPS), color="C3", linewidth=1.3,
             label="Physics loss at GUESS parameters (warm-up, not trained)")
plt.semilogy(joint_x, np.maximum(hist["physics"], EPS), color="C1", linewidth=1.3,
             label="Physics loss (joint training)")
plt.axvline(W, color="gray", linestyle=":", label="Warm-up -> joint training")
plt.axvline(W + best_epoch, color="black", linestyle=":", label="Selected checkpoint")
plt.xlabel("Iteration (warm-up, then Adam)")
plt.ylabel("Physics loss (log scale)")
plt.title("Physics loss")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PHYS_LOSS_FIG, dpi=200, bbox_inches="tight")


epochs = np.asarray(hist["epoch"])
P_hist = np.asarray(hist["params"])
P_start = P_hist[0]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for i, ax in enumerate(axes.ravel()):
    ax.plot(epochs, P_hist[:, i], label="Estimated", linewidth=1.5)
    ax.axhline(REFERENCE[i], color="red", linestyle="--", label="Reference")
    ax.axhline(P_start[i], color="gray", linestyle=":", label="Initial value")
    ax.axvline(best_epoch, color="black", linestyle=":", label="Selected checkpoint")

    y_lo = min(0.6 * REFERENCE[i], 0.95 * p_est[i], 0.95 * P_start[i])
    y_hi = max(1.4 * REFERENCE[i], 1.05 * p_est[i], 1.05 * P_start[i])
    ax.set_ylim(y_lo, y_hi)

    ax.set_title(f"{NAMES[i]}  (start = {P_start[i]:.4f}, final = {p_est[i]:.4f}, "
                 f"ref = {REFERENCE[i]:.4f})", fontsize=10)
    ax.set_xlabel("Adam epoch")
    ax.set_ylabel("Parameter value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PARAM_FIG, dpi=200, bbox_inches="tight")



fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for j, (ax, name) in enumerate(zip(axes, ["X", "S", "P"])):
    ax.plot(t, Y[:, j], ".", color="lightgray", markersize=4, label="All measurements")
    ax.plot(t, Y_pred[:, j], "-", color="C0", linewidth=1.3, label="PINN fit")
    sel = obs_mask[:, j]
    ax.plot(t[sel], Y[sel, j], "rx", markersize=6, label="Q-DEIM samples")
    lo = min(Y[:, j].min(), Y_pred[:, j].min())
    hi = max(Y[:, j].max(), Y_pred[:, j].max())
    pad = 0.15 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(f"{name}(t)")
    ax.set_xlabel("time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(QDEIM_FIG, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nSaved plots: {DATA_LOSS_FIG}, {PHYS_LOSS_FIG}, {PARAM_FIG} and {QDEIM_FIG}")
