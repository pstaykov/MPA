"""NVDA Handelsrichtung: Verteilung von C(tau) über alle Lags vs. Rauschglocke 1/sqrt(N)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from trade_signs import acf_days, label, load_days

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 15,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False,
})

TAU_MAX = 1000
days = load_days()
N = sum(len(d) for d in days)
lags = np.arange(1, TAU_MAX + 1)
c_nvda = acf_days(days, lags)

s = 1 / np.sqrt(N)
fig, ax = plt.subplots(figsize=(9, 5.5))
bins = np.arange(-5 * s, c_nvda.max() + s, s / 2)
ax.hist(c_nvda, bins=bins, density=True, color="C0", alpha=0.6,
        label="NVDA Kauf/Verkauf")
xg = np.linspace(-4.5 * s, 4.5 * s, 400)
ax.plot(xg, norm.pdf(xg, 0, s), "--", color="C3", lw=1.5,
        label=r"Rauschen, $\sigma = 1/\sqrt{N}$")
for sgn in (-1, 1):
    ax.axvline(sgn * 2 * s, color="gray", ls="--", lw=1.2)
ax.set_yscale("log")
ax.set_ylim(0.1, None)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel(rf"Autokorrelation $C(\tau)$,  $\tau = 1 \dots {TAU_MAX}$ Trades")
ax.set_ylabel("Dichte")
ax.set_title(f"NVIDIA (NVDA), {label(days)}", loc="left")
ax.legend(loc="upper right", fontsize=12, handletextpad=0.3)
fig.tight_layout()
fig.savefig("nvda_noise_bell_curve.png", dpi=200)
print("2/sqrt(N) = %.4f" % (2 * s))
print("NVDA     außerhalb ±2/√N: %.1f%%" % (100 * np.mean(np.abs(c_nvda) > 2 * s)))
