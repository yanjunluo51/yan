"""Figure 1 of the revised main text: the T = 6 instance of Proposition 9
(former Table 1). Row caps (7, 6), flip contributions (1,1,1,1,1,1) and
(1,0,0,0,0,0), flip budget b_H = 6, exact stopping."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from certlib import Instance, bounds, capacity

c = np.array([[7] * 6, [6] * 6])
v = np.array([[1] * 6, [1, 0, 0, 0, 0, 0]])
inst = Instance(c, v, 6)

def obs_from(strings):
    return np.array([[ch != "." for ch in s] for s in strings])

cases = [
    ("(a) Minimum safety: 5 entries", ["......", ".00000"]),
    ("(b) Best separate union: 12 entries", ["111111", "100000"]),
    ("(c) Joint certificate: 7 entries", ["111111", "1....."]),
]
XMAX = 14
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, axes = plt.subplots(3, 1, figsize=(3.05, 2.55), sharex=True)
for ax, (title, strings) in zip(axes, cases):
    obs = obs_from(strings)
    lo, up = bounds(inst, obs)
    vm, vp = capacity(up, inst.b), capacity(lo, inst.b)
    for row in range(2):
        y = 1 - row
        u = min(up[row], XMAX)
        ax.plot([lo[row], u], [y, y], color="#3b6ea5", lw=4, solid_capstyle="butt")
        ax.plot([lo[row]], [y], marker="|", color="#1d3c60", ms=8, mew=1.5)
        if up[row] > XMAX:
            ax.annotate("", xy=(XMAX + 0.6, y), xytext=(XMAX - 0.4, y),
                        arrowprops=dict(arrowstyle="-|>", color="#3b6ea5", lw=1.2))
            ax.text(XMAX + 0.9, y, f"u={up[row]}", va="center", fontsize=6)
        else:
            ax.plot([u], [y], marker="|", color="#1d3c60", ms=8, mew=1.5)
        mark = "  safe" if up[row] <= inst.b and row == int(np.argmin(up)) and vm >= 1 else ""
        ax.text(-0.5, y, f"row {row + 1}  {strings[row]}", ha="right", va="center", family="monospace", fontsize=6.3)
    ax.axvline(inst.b, color="#b03a2e", ls="--", lw=0.9)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.text(-0.62, 1.32, title, transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
    ax.text(1.0, 1.32, f"$V^-$={vm}, $V^+$={vp}", transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom")
    if title.startswith("(c)"):
        ax.text(7.0, 0.5, r"$\ell_1+\ell_2=7>b_H$", fontsize=6, va="center", color="#444444")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[-1].set_xlim(0, XMAX + 3)
axes[-1].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
axes[-1].set_xlabel(r"cost interval $[\ell_i,u_i]$ (flips); dashed: $b_H=6$", fontsize=6.5)
fig.subplots_adjust(left=0.36, right=0.93, top=0.885, bottom=0.14, hspace=0.95)
fig.savefig("../paper/figures/fig1_joint_example.pdf")
print("saved")
