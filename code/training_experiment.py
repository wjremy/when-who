"""Part B of the original experiment notebook (personalised_forecasting_experiments.ipynb, cells 8-21):
training with pooled vs personalised losses on synthetic data. Run on Kaggle in Sep 2026; outputs in results/kaggle_runs/.
Extracted unchanged on 9 Oct 2026 (Wesley approved) so that Supplement S9 has a reproducible source. Synthetic data only."""

# ## Part B. Training: does a personalised loss give better personalised forecasts?
# 
# Pooled loss is dominated by learning each patient's base rate. We compare losses that remove that part, stopping rules, and whether pooled AUC picks the right model.
# 
# | loss | what it does |
# |---|---|
# | `bce_day` | standard binary cross-entropy, patients sampled by number of days (usual practice) |
# | `bce_patient` | same loss, every patient weighted equally |
# | `bce_offset` | the patient's run-in base rate is a fixed offset; the network learns only the deviation |
# | `pairwise` | logistic ranking loss over event vs event-free days of the **same** patient |
# | `offset+pairwise` | both |
# 
# DGP: patient base rates with between-patient SD τ (log-odds); a nonlinear, lagged, partly cyclic daily signal of strength ω, learnable only relative to the patient's own usual level. GRU on a 30-day window; base rate from a fixed 60-day run-in (an expanding-window rate would be biased, see C1).
# Run on GPU: Settings → Accelerator → GPU T4 ×2. Smoke test first; the grids are resumable.

# B1 Setup and configuration
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

QUICK = False            # True = tiny smoke test only
CFG = dict(
    K_train=300, K_val=100, K_test=200,   # patients per split
    T_min=120, T_max=365,                 # diary length per patient (days), uniform
    d=6,                                  # daily features
    W=30,                                 # input window (days of history fed to the GRU)
    RUNIN=60,                             # run-in days: base rate estimated here, not evaluated
    mu=0.15,                              # target mean event rate
    epochs=30, P=32, D=32,                # batch = P patients x D days
    hidden=64, lr=1e-3, wd=1e-5, dropout=0.1,
    prior_strength=10,                    # pseudo-days for the history base-rate estimate
)
GRID = dict(tau=[0.5, 1.0, 1.5], omega=[0.5, 1.0], seed=[0, 1, 2],
            loss=["bce_day", "bce_patient", "bce_offset", "pairwise", "offset+pairwise"])
if QUICK:
    CFG.update(K_train=80, K_val=40, K_test=60, epochs=4)
    GRID = dict(tau=[1.0], omega=[1.0], seed=[0], loss=GRID["loss"])

# B2 Data-generating process
# logit P(y_kt = 1) = alpha_k + omega * h_kt
#   alpha_k ~ N(m, tau^2): patient base rate (between-patient heterogeneity)
#   h_kt: standardised within-patient signal built from the patient's OWN deviations
#         (lagged triggers, a threshold, an interaction, a ~28-day cycle); learnable only
#         relative to the patient's usual level, so the model must use history.
# Features also carry patient 'traits' correlated with alpha_k, so a model can read the base
# rate from feature levels, just as real diaries reveal who is a frequent-attack patient.

def _signal(z, t_idx, phase, amp):
    """z: (K, T, d) standardised deviations; returns raw within-patient signal (K, T)."""
    K, T, d = z.shape
    lag1 = np.roll(z, 1, axis=1); lag2 = np.roll(z, 2, axis=1)
    lag1[:, :1] = 0; lag2[:, :2] = 0
    h = (0.8 * lag1[..., 0]
         + 0.7 * np.maximum(lag1[..., 1] + lag2[..., 1] - 1.0, 0)
         + 0.6 * lag1[..., 2] * lag1[..., 3]
         + 0.6 * (lag2[..., 4] > 1.5)
         + amp[:, None] * np.sin(2 * np.pi * (t_idx[None, :] + phase[:, None]) / 28.0))
    return h


def simulate_cohort(K, tau, omega, m, rng, cfg=CFG, h_scale=None):
    T, d = cfg["T_max"], cfg["d"]
    alpha = rng.normal(m, tau, K)
    za = (alpha - m) / max(tau, 1e-6)
    trait = rng.normal(0, 1, (K, d))
    trait[:, :3] = 0.8 * za[:, None] + 0.6 * trait[:, :3]        # traits reveal base rate
    scale = np.exp(rng.normal(0, 0.2, (K, d)))
    eps = np.zeros((K, T, d)); e = rng.normal(0, 1, (K, d))
    for t in range(T):                                            # AR(1) daily noise
        e = 0.5 * e + math.sqrt(1 - 0.25) * rng.normal(0, 1, (K, d)); eps[:, t] = e
    x = trait[:, None, :] + scale[:, None, :] * eps
    z = eps                                                       # true standardised deviation
    phase = rng.uniform(0, 28, K); amp = rng.uniform(0, 1, K)
    h = _signal(z, np.arange(T), phase, amp)
    if h_scale is None:
        h_scale = (h[:, 2:].mean(), h[:, 2:].std())
    h = (h - h_scale[0]) / h_scale[1]
    eta = alpha[:, None] + omega * h
    risk = expit(eta)
    y = (rng.random((K, T)) < risk).astype(np.float32)
    lengths = rng.integers(cfg["T_min"], cfg["T_max"] + 1, K)
    valid = np.arange(T)[None, :] < lengths[:, None]
    y[~valid] = 0
    return dict(x=x.astype(np.float32), y=y, risk=risk.astype(np.float32), alpha=alpha,
                lengths=lengths, valid=valid, h_scale=h_scale)


def calibrate_m(tau, omega, cfg=CFG, seed=123):
    """Intercept m giving mean event rate cfg['mu']; also returns the shared signal scaling."""
    probe = simulate_cohort(400, tau, omega, 0.0, np.random.default_rng(seed), cfg)
    eta0 = logit(np.clip(probe["risk"], 1e-7, 1 - 1e-7))[probe["valid"]]
    lo, hi = -12.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(eta0 + mid).mean() < cfg["mu"]: lo = mid
        else: hi = mid
    return (lo + hi) / 2, probe["h_scale"]


def runin_rate(y, prior_mu, strength, runin):
    """Each patient's event rate over a run-in period (the first `runin` days), shrunk to the
    prior, held CONSTANT afterwards. Forecasts are only evaluated after the run-in.
    (An expanding-window rate would bias within-patient AUC downward in finite diaries,
    because within a patient's realised sequence past and future event counts are negatively
    related; a fixed run-in estimate avoids that.)"""
    r = (y[:, :runin].sum(1) + strength * prior_mu) / (runin + strength)
    return np.repeat(r[:, None], y.shape[1], axis=1)

# B3 Model, losses, training loop (k_train overrides the number of training patients)
class Cohort:
    """Holds one simulated cohort on the device, with the causal history base rate."""
    def __init__(self, c, prior_mu):
        W, R = CFG["W"], CFG["RUNIN"]
        hr = runin_rate(c["y"], prior_mu, CFG["prior_strength"], CFG["RUNIN"])
        self.hist = torch.tensor(logit(np.clip(hr, 1e-4, 1 - 1e-4)), dtype=torch.float32, device=DEVICE)
        feats = np.concatenate([c["x"], c["y"][..., None]], axis=-1)
        self.X = torch.tensor(feats, dtype=torch.float32, device=DEVICE)       # (K, T, d+1)
        self.y = torch.tensor(c["y"], dtype=torch.float32, device=DEVICE)
        self.risk = c["risk"]; self.alpha = c["alpha"]
        self.K = c["x"].shape[0]; self.len = c["lengths"]
        self.n_valid = np.maximum(self.len - R, 0)
        # every evaluable (patient, day): after the run-in (R >= W, so a full window exists)
        kk, tt = [], []
        for k in range(self.K):
            t = np.arange(R, self.len[k]); kk.append(np.full(len(t), k)); tt.append(t)
        self.kk = torch.tensor(np.concatenate(kk), device=DEVICE)
        self.tt = torch.tensor(np.concatenate(tt), device=DEVICE)
        self.offs = torch.arange(-W, 0, device=DEVICE)

    def batch(self, kk, tt):
        win = self.X[kk[:, None], tt[:, None] + self.offs]                    # (B, W, d+1)
        return win, self.hist[kk, tt], self.y[kk, tt]


class Net(nn.Module):
    def __init__(self, d_in, H, p, offset):
        super().__init__()
        self.gru = nn.GRU(d_in, H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(H + 1, H), nn.ReLU(), nn.Dropout(p), nn.Linear(H, 1))
        self.offset = offset
    def forward(self, win, hist):
        _, h = self.gru(win)
        out = self.head(torch.cat([h[-1], hist[:, None]], dim=1)).squeeze(-1)
        return out + hist if self.offset else out


def pairwise_within(logits, y):
    """Mean logistic loss over (event day, event-free day) pairs of the SAME patient."""
    diff = logits[:, :, None] - logits[:, None, :]                            # (P, D, D)
    mask = (y[:, :, None] == 1) & (y[:, None, :] == 0)
    n = mask.sum()
    return (F.softplus(-diff) * mask).sum() / n.clamp(min=1)


LOSSES = {  # name: (offset in model, patient sampling, bce weight, pairwise weight)
    "bce_day":         (False, "day",     1.0, 0.0),
    "bce_patient":     (False, "patient", 1.0, 0.0),
    "bce_offset":      (True,  "day",     1.0, 0.0),
    "pairwise":        (False, "patient", 0.0, 1.0),
    "offset+pairwise": (True,  "patient", 1.0, 1.0),
}


@torch.no_grad()
def predict(net, co, bs=8192):
    net.eval(); out = []
    for i in range(0, len(co.kk), bs):
        win, hist, _ = co.batch(co.kk[i:i + bs], co.tt[i:i + bs])
        out.append(net(win, hist))
    return torch.cat(out).float().cpu().numpy()


def evaluate(net, co):
    s = predict(net, co)
    y = co.y[co.kk, co.tt].cpu().numpy(); pid = co.kk.cpu().numpy()
    return dict(pooled_auc=roc_auc_score(y, s), within_auc=within_auc(y, s, pid),
                bce=bce(y, expit(s))), s


def train_run(tau, omega, seed, loss, keep_preds=False, k_train=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m, hs = calibrate_m(tau, omega)
    rng = np.random.default_rng(1000 + seed)
    K = dict(train=k_train or CFG["K_train"], val=CFG["K_val"], test=CFG["K_test"])
    sims = {s: simulate_cohort(K[s], tau, omega, m, rng, h_scale=hs) for s in ("train", "val", "test")}
    prior = float(sims["train"]["y"][sims["train"]["valid"]].mean())  # population prior for shrinkage
    tr, va, te = (Cohort(sims[s], prior) for s in ("train", "val", "test"))
    offset, sampling, w_bce, w_pair = LOSSES[loss]
    net = Net(CFG["d"] + 1, CFG["hidden"], CFG["dropout"], offset).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    P, D = CFG["P"], CFG["D"]
    prob = tr.n_valid / tr.n_valid.sum() if sampling == "day" else np.full(tr.K, 1 / tr.K)
    steps = max(1, int(tr.n_valid.sum() // (P * D)))
    logs, best = [], (-1, None)
    for ep in range(1, CFG["epochs"] + 1):
        net.train()
        for _ in range(steps):
            ks = rng.choice(tr.K, P, p=prob)
            ts = np.stack([rng.integers(CFG["RUNIN"], tr.len[k], D) for k in ks])
            kk = torch.tensor(np.repeat(ks, D), device=DEVICE); tt = torch.tensor(ts.ravel(), device=DEVICE)
            win, hist, yb = tr.batch(kk, tt)
            logits = net(win, hist)
            l = 0.0
            if w_bce: l = l + w_bce * F.binary_cross_entropy_with_logits(logits, yb)
            if w_pair: l = l + w_pair * pairwise_within(logits.view(P, D), yb.view(P, D))
            opt.zero_grad(); l.backward(); opt.step()
        v, _ = evaluate(net, va); t, s_te = evaluate(net, te)
        logs.append(dict(tau=tau, omega=omega, seed=seed, loss=loss, k_train=K["train"], epoch=ep,
                         **{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}))
        if keep_preds and v["within_auc"] > best[0]:
            best = (v["within_auc"], s_te.copy())
    # reference forecasters on the test cohort
    y_te = te.y[te.kk, te.tt].cpu().numpy(); pid = te.kk.cpu().numpy(); tt = te.tt.cpu().numpy()
    r = te.risk[pid, tt]; hist_p = expit(te.hist[te.kk, te.tt].cpu().numpy())
    ref = dict(tau=tau, omega=omega, seed=seed, k_train=K["train"],
               oracle_pooled=roc_auc_score(y_te, r), oracle_within=within_auc(y_te, r, pid),
               baserate_pooled=roc_auc_score(y_te, hist_p), baserate_within=within_auc(y_te, hist_p, pid))
    extra = dict(y=y_te, pid=pid, risk=r, hist=hist_p, pred=best[1], thr=prior) if keep_preds else None
    return logs, ref, extra

# B4 Grid runner: worker threads per GPU (two per GPU keeps both T4s busy); resumable via the CSVs
import threading, queue
def run_grid(runs, keys, ep_csv, ref_csv, threads_per_gpu=2):
    done = set()
    if os.path.exists(ep_csv):
        done = set(map(tuple, pd.read_csv(ep_csv)[keys].drop_duplicates().values.tolist()))
    runs = [r for r in runs if tuple(r) not in done]
    n_gpu = max(torch.cuda.device_count(), 1); total = len(done) + len(runs)
    print(f"{len(done)} done, {len(runs)} to go, {n_gpu} GPU(s)")
    todo, lock, t0, counter = queue.Queue(), threading.Lock(), time.time(), [len(done)]
    for r in runs: todo.put(r)
    def worker(g):
        if DEVICE == "cuda": torch.cuda.set_device(g)
        while True:
            try: r = todo.get_nowait()
            except queue.Empty: return
            a = dict(zip(keys, r))
            try: logs, ref, _ = train_run(a["tau"], a["omega"], a["seed"], a["loss"], k_train=a.get("k_train"))
            except Exception as e:
                print(f"gpu{g} FAILED {a}: {e}"); continue
            best = max(logs, key=lambda x: x["val_within_auc"])
            with lock:
                pd.DataFrame(logs).to_csv(ep_csv, mode="a", header=not os.path.exists(ep_csv), index=False)
                pd.DataFrame([ref]).to_csv(ref_csv, mode="a", header=not os.path.exists(ref_csv), index=False)
                counter[0] += 1
                print(f"[{counter[0]}/{total}] gpu{g} {a}  test within {best['test_within_auc']:.3f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    th = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu) for _ in range(threads_per_gpu)]
    for t in th: t.start()
    for t in th: t.join()
    print("grid finished")

# B5 Smoke test (about a minute): one short run per loss
if RUN["B_smoke"]:
    _old = CFG["epochs"]; CFG["epochs"] = 2
    for loss in LOSSES:
        t0 = time.time()
        logs, ref, _ = train_run(1.0, 1.0, 0, loss)
        print(f"{loss:16s} test within AUC {logs[-1]['test_within_auc']:.3f}  pooled {logs[-1]['test_pooled_auc']:.3f}  ({time.time() - t0:.0f}s)")
    print("oracle within", round(ref["oracle_within"], 3), "| base rate: pooled", round(ref["baserate_pooled"], 3),
          "within", round(ref["baserate_within"], 3))
    CFG["epochs"] = _old

# B6 Main grid: tau x omega x seed x loss = 90 runs
EP_CSV, REF_CSV = f"{OUT}/epoch_logs.csv", f"{OUT}/reference.csv"
KEYS = ["tau", "omega", "seed", "loss"]
if RUN["B_grid"]:
    run_grid(list(itertools.product(*[GRID[k] for k in KEYS])), KEYS, EP_CSV, REF_CSV)

# B7 Analysis: does a personalised loss help, and which stopping rule picks the best model?
if os.path.exists(EP_CSV):
    ep = pd.read_csv(EP_CSV)
    ref = pd.read_csv(REF_CSV).drop_duplicates(["tau", "omega", "seed"])
    CRIT = {"val loss (pooled BCE)": ("val_bce", "min"), "val pooled AUC": ("val_pooled_auc", "max"),
            "val within-patient AUC": ("val_within_auc", "max"), "last epoch": ("epoch", "max")}
    rows = []
    for key, g in ep.groupby(["tau", "omega", "seed", "loss"]):
        for name, (col, how) in CRIT.items():
            r = g.loc[g[col].idxmin()] if how == "min" else g.loc[g[col].idxmax()]
            rows.append(dict(zip(["tau", "omega", "seed", "loss"], key), criterion=name, epoch=int(r.epoch),
                             test_within=r.test_within_auc, test_pooled=r.test_pooled_auc))
    sel = pd.DataFrame(rows).merge(ref, on=["tau", "omega", "seed"])
    sel["share_of_oracle_skill"] = (sel.test_within - 0.5) / (sel.oracle_within - 0.5)

    print("A. Test within-patient AUC by loss (model selected on val within-patient AUC), mean over seeds")
    a = sel[sel.criterion == "val within-patient AUC"].pivot_table(index=["tau", "omega"], columns="loss", values="test_within")
    a["oracle"] = ref.groupby(["tau", "omega"]).oracle_within.mean()
    display(a.round(3))

    print("B. Same, as share of the oracle's within-patient skill (AUC - 0.5), mean over all settings")
    display(sel[sel.criterion == "val within-patient AUC"].groupby("loss").share_of_oracle_skill.agg(["mean", "std"]).round(3))

    print("C. Stopping rule: test within-patient AUC of the selected epoch, standard loss (bce_day)")
    display(sel[sel.loss == "bce_day"].pivot_table(index=["tau", "omega"], columns="criterion", values="test_within").round(3))
    display(sel[sel.loss == "bce_day"].pivot_table(index=["tau", "omega"], columns="criterion", values="epoch").round(1))

    print("D. Paired differences vs bce_day (same data, same seed), val-within selection")
    b = sel[sel.criterion == "val within-patient AUC"].pivot_table(index=["tau", "omega", "seed"], columns="loss", values="test_within")
    diff = b.sub(b["bce_day"], axis=0).drop(columns="bce_day")
    display(pd.DataFrame({"mean diff": diff.mean(), "sd": diff.std(), "n": diff.count(),
                          "share > 0": (diff > 0).mean()}).round(4))

    print("E. Pooled vs within for the reference forecasters")
    display(ref.groupby(["tau", "omega"])[["oracle_pooled", "oracle_within", "baserate_pooled", "baserate_within"]].mean().round(3))
    sel.to_csv(f"{OUT}/selected_models.csv", index=False)

# B8 Model selection: do pooled and within-patient AUC pick the same model?
if os.path.exists(EP_CSV):
    ep = pd.read_csv(EP_CSV)
    g = ep.loc[ep.groupby(KEYS).val_within_auc.idxmax()]        # each run at its best within-AUC epoch
    out = []
    for k, d in g.groupby(["tau", "omega", "seed"]):
        a, b = d.loc[d.val_pooled_auc.idxmax()], d.loc[d.val_within_auc.idxmax()]
        out.append(dict(tau=k[0], pooled_pick=a.loss, within_pick=b.loss, agree=a.loss == b.loss,
                        cost_within_auc=a.test_within_auc - b.test_within_auc))
    o = pd.DataFrame(out)
    print(f"pooled and within AUC pick the same loss in {o.agree.sum()} of {len(o)} settings")
    display(o.groupby("tau")[["agree", "cost_within_auc"]].mean().round(4))
    print("Spearman correlation of val pooled vs val within AUC over all (loss, epoch) candidates:")
    display(ep.groupby("tau").apply(lambda d: d[["val_pooled_auc", "val_within_auc"]].corr("spearman").iloc[0, 1]).round(2))
    o.to_csv(f"{OUT}/model_selection.csv", index=False)

# B9 Metric families: pooled vs within for the GRU, the base-rate forecaster and the oracle
KEEP = (1.0, 0.5, 0)
kept = globals().get("kept", {})
if RUN["B_grid"] or os.path.exists(EP_CSV):
    for loss in ("bce_day", "bce_offset"):
        if loss not in kept:
            kept[loss] = train_run(*KEEP, loss, keep_preds=True)[2]
    e = kept["bce_day"]; fam = []
    for name, p in [("GRU, standard loss", expit(e["pred"])), ("GRU, base-rate offset", expit(kept["bce_offset"]["pred"])),
                    ("Base rate from history", e["hist"]), ("Oracle (true risk)", e["risk"])]:
        t = metric_families(e["y"], p, e["pid"], e["thr"], e["hist"]); t.insert(0, "forecaster", name); fam.append(t)
    fam = pd.concat(fam); fam["pooled_minus_within"] = fam.pooled - fam.within_patient
    display(fam.round(3)); fam.to_csv(f"{OUT}/metric_families.csv", index=False)

# B10 Training curves: pooled vs within AUC on validation, with validation loss
if os.path.exists(EP_CSV):
    ep = pd.read_csv(EP_CSV)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, loss in zip(axes, ["bce_day", "offset+pairwise"]):
        g = ep[(ep.tau == 1.0) & (ep.omega == 0.5) & (ep.loss == loss)].groupby("epoch")[["val_pooled_auc", "val_within_auc", "val_bce"]].mean()
        ax.plot(g.index, g.val_pooled_auc, label="pooled AUC"); ax.plot(g.index, g.val_within_auc, label="within-patient AUC")
        ax2 = ax.twinx(); ax2.plot(g.index, g.val_bce, "k:", label="val BCE"); ax2.set_ylabel("BCE")
        ax.set_title(f"{loss}, tau=1.0, omega=0.5"); ax.set_xlabel("epoch"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png", dpi=150); plt.show()

# ### B11 Follow-up: heterogeneity × training-set size
# Does the personalised-loss gain grow with between-patient heterogeneity and with more training patients?

# B11a Follow-up grid: tau x k_train x seed x loss = 108 runs
GRID2 = dict(tau=[1.5, 2.0, 2.5], omega=[0.5], k_train=[300, 100, 50], seed=[0, 1, 2],
             loss=["bce_day", "bce_offset", "pairwise", "offset+pairwise"])
EP2, REF2 = f"{OUT}/followup_epoch_logs.csv", f"{OUT}/followup_reference.csv"
KEYS2 = ["tau", "omega", "k_train", "seed", "loss"]
if RUN["B_followup"]:
    run_grid(list(itertools.product(*[GRID2[k] for k in KEYS2])), KEYS2, EP2, REF2)

# B11b Follow-up analysis: test within AUC at the best validation-within epoch, and gain over bce_day
if os.path.exists(EP2):
    f = pd.read_csv(EP2)
    gf = f.loc[f.groupby(KEYS2).val_within_auc.idxmax()]
    display(gf.pivot_table(index=["tau", "k_train"], columns="loss", values="test_within_auc").round(3))
    w = gf.pivot_table(index=["tau", "k_train", "seed"], columns="loss", values="test_within_auc")
    gain = w.sub(w["bce_day"], axis=0).drop(columns="bce_day")
    print("gain in test within AUC over bce_day, mean over seeds:")
    display(gain.groupby(["tau", "k_train"]).mean().round(4))
