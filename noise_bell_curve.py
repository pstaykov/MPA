"""Warum 2/sqrt(N)? Verteilung von C(tau) für eine zufällige Kauf/Verkauf-Folge."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from trade_signs import acf, load_days

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 15,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False,
})

N = sum(len(d) for d in load_days())  # so viele Trades wie die NVDA-Daten
rng = np.random.default_rng(0)
lags = np.arange(1, 1001)

# 50 zufällige Folgen aus Käufen (+1) und Verkäufen (-1), je C(tau) für 1000 Lags
vals = np.concatenate([acf(rng.choice([-1.0, 1.0], N), lags) for _ in range(50)])

s = 1 / np.sqrt(N)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.hist(vals, bins=80, density=True, color="C0", alpha=0.6,
        label=r"$C(\tau)$ zufälliger Kauf/Verkauf-Folgen")
xg = np.linspace(-4.5 * s, 4.5 * s, 400)
ax.plot(xg, norm.pdf(xg, 0, s), "--", color="C3", lw=1.5,
        label=r"Normalverteilung, $\sigma = 1/\sqrt{N}$")
m = np.abs(xg) <= 2 * s
ax.fill_between(xg[m], norm.pdf(xg[m], 0, s), color="C3", alpha=0.12,
                label="95 % der Werte")
for sgn in (-1, 1):
    ax.axvline(sgn * 2 * s, color="gray", ls="--", lw=1.2)
ax.text(2 * s, ax.get_ylim()[1] * 0.55, "  $+2/\\sqrt{N}$\n  ≈ %.4f" % (2 * s),
        color="gray", fontsize=13)
ax.text(-2 * s, ax.get_ylim()[1] * 0.55, r"$-2/\sqrt{N}$  ", color="gray",
        fontsize=13, ha="right")
ax.set_xlabel(r"Autokorrelation $C(\tau)$,  $\tau$ in Trades")
ax.set_ylabel("Dichte")
ax.set_title("Reines Rauschen, N = " + f"{N:,}".replace(",", ".") + " Trades", loc="left")
ax.legend(loc="upper left", fontsize=12, handletextpad=0.3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
fig.tight_layout()
fig.savefig("noise_bell_curve.png", dpi=200)
print(f"außerhalb ±2/√N: {np.mean(np.abs(vals) > 2*s):.3%}")
