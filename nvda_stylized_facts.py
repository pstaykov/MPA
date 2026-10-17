"""Stilisierte Fakten am Beispiel der NVIDIA-Aktie (NVDA).

(a) Fat Tails: Verteilung der Tagesrenditen vs. Normalverteilung
(b) Autokorrelation der Handelsrichtung (Kauf/Verkauf), tau in Trades
"""
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

from trade_signs import acf_days, label as data_label, load_days

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})

TICKER = "NVDA"


def log_returns(interval, period):
    px = yf.download(TICKER, interval=interval, period=period,
                     auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    return np.diff(np.log(px.values))


daily = log_returns("1d", "max")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) Fat Tails: Histogramm der normierten Tagesrenditen, Bins der Breite 0.5
z = (daily - daily.mean()) / daily.std()
lim = np.ceil(np.abs(z).max() / 0.5) * 0.5
bins = np.arange(-lim, lim + 0.5, 0.5)
ax1.hist(z, bins=bins, density=True, color="C0", alpha=0.7,
         edgecolor="white", lw=0.5, label="Tagesrenditen")
xg = np.linspace(-5, 5, 400)
ax1.plot(xg, norm.pdf(xg), "--", color="C3", lw=1.5, label="Normalverteilung")
ax1.set_yscale("log")
ax1.set_ylim(1e-4, 2)
ax1.set_xlim(-lim, lim)
ax1.set_xlabel(r"Normierte Rendite  $(r-\mu)/\sigma$")
ax1.set_ylabel("Dichte")
ax1.set_title("(a) Fat Tails", loc="left")
ax1.text(0.97, 0.95, f"{daily.size:,}".replace(",", ".") + " Handelstage",
         transform=ax1.transAxes, fontsize=12, color="gray", ha="right", va="top")
ax1.legend(loc="upper left", handletextpad=0.3)

# (b) Autokorrelation der Handelsrichtung
days = load_days()
n_trades = sum(len(d) for d in days)
lags = np.unique(np.logspace(0, 4, 40).astype(int))
c = acf_days(days, lags)
ok = c > 0  # Log-Achse: nur positive Werte
ax2.plot(lags[ok], c[ok], "o", color="C0", ms=7, label="NVDA Kauf/Verkauf")
ax2.axhline(2 / np.sqrt(n_trades), color="gray", ls="--", lw=1.2,
            label="95 %-Rauschgrenze")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylim(1e-4, 10)
ax2.set_xlabel(r"Verzögerung $\tau$ (ausgeführte Trades)")
ax2.set_ylabel(r"Autokorrelation $C(\tau)$")
ax2.set_title("(b) Autokorrelation", loc="left")
ax2.text(0.03, 0.04, f"Nasdaq, {data_label(days)}", transform=ax2.transAxes,
         fontsize=12, color="gray")
ax2.legend(loc="upper right", handletextpad=0.3)

fig.suptitle(f"NVIDIA ({TICKER})", fontsize=16)
fig.tight_layout()
fig.savefig("nvda_stylized_facts.png", dpi=200)
print(len(daily), sum(len(d) for d in days))
