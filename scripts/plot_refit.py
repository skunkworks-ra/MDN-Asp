"""Before/after refit for one major cycle, from the arrays saved by --diff_save_atoms.

Usage: python scripts/plot_refit.py RUN_DIR [MAJOR]. Writes RUN_DIR/refit_before_after_majorNN.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(sys.argv[1])
N = f"major{int(sys.argv[2]) if len(sys.argv) > 2 else 0:02d}"
cy, cx = slice(113, 347), slice(108, 343)   # 3C391 crop from STATE.md figure style

res_b = np.load(D / f"{N}_before_refit_residual.npy")
res_a = np.load(D / f"{N}_residual.npy")
mod_b = np.load(D / f"{N}_before_refit_model.npy")
mod_a = np.load(D / f"{N}_model.npy")
z = np.load(D / f"atoms_refit_{N}.npz")
ab, aa = z["a_before"], z["a_after"]

fig, ax = plt.subplots(2, 3, figsize=(16, 10))
r = max(np.abs(res_b[cy, cx]).max(), np.abs(res_a[cy, cx]).max())
for a, im, t in ((ax[0, 0], res_b, "residual before refit"), (ax[0, 1], res_a, "residual after refit")):
    h = a.imshow(im[cy, cx], origin="lower", cmap="RdBu_r", vmin=-r, vmax=r)
    a.set_title(f"{t}  (+-{r:.2e} Jy/beam)")
    plt.colorbar(h, ax=a, fraction=0.046)
d = (mod_a - mod_b)[cy, cx]
dm = np.abs(d).max()
h = ax[0, 2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-dm, vmax=dm)
ax[0, 2].set_title(f"model change (after - before), sum {(mod_a - mod_b).sum():.3f} Jy")
plt.colorbar(h, ax=ax[0, 2], fraction=0.046)

m = max(mod_b[cy, cx].max(), mod_a[cy, cx].max())
for a, im, t in ((ax[1, 0], mod_b, f"model before refit, {mod_b.sum():.4f} Jy"),
                 (ax[1, 1], mod_a, f"model after refit, {mod_a.sum():.4f} Jy")):
    h = a.imshow(im[cy, cx], origin="lower", cmap="inferno", vmin=0, vmax=m)
    a.set_title(t)
    plt.colorbar(h, ax=a, fraction=0.046)

lim = max(ab.max(), aa.max())
ax[1, 2].scatter(ab, aa, s=6, alpha=0.5)
ax[1, 2].plot([0, lim], [0, lim], "k--", lw=0.8)
ax[1, 2].set_xlabel("atom amplitude before refit")
ax[1, 2].set_ylabel("after refit")
ax[1, 2].set_title(f"{len(ab)} atoms: {(aa < ab).sum()} down, {(aa > ab).sum()} up, {(aa == 0).sum()} zeroed, {(aa == ab).sum()} same")
plt.tight_layout()
plt.savefig(D / f"refit_before_after_{N}.png", dpi=90)
print("n atoms", len(ab), "a_before sum", ab.sum(), "a_after sum", aa.sum())
