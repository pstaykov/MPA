"""Extrahiert Trades mit Richtung (Kauf/Verkauf) aus einer Nasdaq TotalView-ITCH 5.0 Datei.

Richtung = Seite des aggressiven (initiierenden) Auftrags. ITCH meldet bei
Ausführungen (E/C) die Seite des *ruhenden* Limit-Auftrags; wird ein ruhender
Kaufauftrag ausgeführt, war der Initiator ein Verkäufer (-1) und umgekehrt.

Aufruf:  python itch_trade_signs.py <datei.NASDAQ_ITCH50.gz | -> NVDA <ausgabe.npz>
"""
import gzip
import sys

import numpy as np
from numba import njit, types
from numba.typed import Dict

CHUNK = 256 * 1024 * 1024


@njit(inline="always")
def be(buf, off, n):
    v = 0
    for i in range(n):
        v = (v << 8) | buf[off + i]
    return v


@njit
def parse(buf, n, target, st, sides, ts, sign, qty, px, k):
    """st[0] = stock locate des Ziel-Tickers (-1 = noch unbekannt)."""
    p = 0
    while p + 2 <= n:
        ln = (buf[p] << 8) | buf[p + 1]
        if p + 2 + ln > n:
            break
        m = p + 2
        t = buf[m]
        loc = (buf[m + 1] << 8) | buf[m + 2]
        if t == 82 and st[0] < 0:  # 'R' Stock Directory
            ok = True
            for i in range(8):
                if buf[m + 11 + i] != target[i]:
                    ok = False
                    break
            if ok:
                st[0] = loc
        elif loc == st[0]:
            if t == 65 or t == 70:  # 'A'/'F' Add Order
                ref = be(buf, m + 11, 8)
                sides[ref] = 1 if buf[m + 19] == 66 else -1  # 'B'
            elif t == 85:  # 'U' Replace
                old = be(buf, m + 11, 8)
                if old in sides:
                    sides[be(buf, m + 19, 8)] = sides[old]
            elif t == 69 or t == 67:  # 'E'/'C' Execution
                ref = be(buf, m + 11, 8)
                if ref in sides and k < ts.shape[0]:
                    if t == 67 and buf[m + 31] != 89:  # nicht druckbar
                        p += 2 + ln
                        continue
                    ts[k] = be(buf, m + 5, 6)
                    sign[k] = -sides[ref]
                    qty[k] = be(buf, m + 19, 4)
                    px[k] = be(buf, m + 32, 4) if t == 67 else -1
                    k += 1
        p += 2 + ln
    return p, k


def main(path, symbol, out):
    target = np.frombuffer(symbol.ljust(8).encode(), dtype=np.uint8).copy()
    st = np.array([-1], dtype=np.int64)
    sides = Dict.empty(key_type=types.int64, value_type=types.int64)
    cap = 5_000_000
    ts = np.zeros(cap, np.int64)
    sign = np.zeros(cap, np.int8)
    qty = np.zeros(cap, np.int64)
    px = np.zeros(cap, np.int64)
    k = 0
    rest = b""
    total = 0
    src = gzip.GzipFile(fileobj=sys.stdin.buffer) if path == "-" else gzip.open(path, "rb")
    with src as f:
        while True:
            try:
                data = f.read(CHUNK)
            except EOFError:  # unvollständige Datei
                print("Datei endet unerwartet, verarbeite bis hier")
                break
            if not data:
                break
            buf = np.frombuffer(rest + data, dtype=np.uint8)
            used, k = parse(buf, buf.shape[0], target, st, sides, ts, sign, qty, px, k)
            rest = bytes(buf[used:])
            total += len(data)
            print(f"{total / 1e9:6.1f} GB gelesen, {k} Ausführungen", flush=True)
    np.savez_compressed(out, ts=ts[:k], sign=sign[:k], qty=qty[:k], px=px[:k])
    print(f"locate={st[0]}  Ausführungen={k}  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
