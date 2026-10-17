"""Gemeinsame Hilfsfunktionen: NVDA-Handelsrichtungen laden und C(tau) berechnen."""
import glob

import numpy as np

DATA = "trades/nvda_*.npz"  # je Handelstag eine Datei (aus itch_trade_signs.py)

OPEN_NS, CLOSE_NS = 9.5 * 3600e9, 16 * 3600e9  # fortlaufender Handel


def load_signs(path):
    """Vorzeichen eps_n = +1 (Kauf) / -1 (Verkauf) je Trade eines Tages.

    Ausführungen mit identischem Zeitstempel und gleicher Richtung stammen von
    einem einzigen aggressiven Auftrag, der mehrere Limit-Orders abräumt; sie
    werden zu einem Trade zusammengefasst.
    """
    d = np.load(path)
    ts, s = d["ts"], d["sign"].astype(np.int64)
    m = (ts >= OPEN_NS) & (ts < CLOSE_NS)
    ts, s = ts[m], s[m]
    new = np.ones(len(s), bool)
    new[1:] = (ts[1:] != ts[:-1]) | (s[1:] != s[:-1])
    return s[new].astype(float)


def load_days(pattern=DATA):
    return [load_signs(p) for p in sorted(glob.glob(pattern))]


def label(days):
    n = sum(len(d) for d in days)
    tage = "Handelstag" if len(days) == 1 else "Handelstage"
    return f"{len(days)} {tage}, N = " + f"{n:,}".replace(",", ".") + " Trades"


def acf(x, lags):
    """C(tau) = <x_n x_{n+tau}> - <x>^2, normiert auf C(0) = 1 (via FFT)."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:n]
    return c[np.asarray(lags)] / c[0]


def acf_days(days, lags):
    """C(tau) je Tag berechnet (keine Paare über Nacht), gewichtet gemittelt."""
    w = np.array([len(d) for d in days], float)
    return np.average([acf(d, lags) for d in days], axis=0, weights=w)

