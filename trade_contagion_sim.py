"""
Visualisation of the trader-interaction model from
Wilinski & Kanniainen, "Agent-based model of information diffusion in the
limit order book trading" (arXiv:2508.20672).

Model (as in the paper):
  * N = 1000 traders, M ~ 4000 links (<k> = 8) on three networks:
      - modified square lattice (Moore neighbourhood on a torus, k = 8)
      - Erdos-Renyi G(N, M)
      - Barabasi-Albert (m = 4)
  * Every trader places "source" market orders on its own (Poisson clock).
  * When a trader places a source order, every neighbour copies it with
    probability q = N / (4M); the copy is placed after an Exp(lambda_f) delay
    and is then forwarded to the copier's neighbours (except the sender).
  * Market orders = trades, so this script simulates the market-order cascade
    process, which is independent of the limit/cancel order streams.

Visual-only change: the source-order rate is lowered so individual trades and
cascades can be followed by eye. q, lambda_f and the networks are unchanged.

Output: trade_contagion.mp4 next to this script.
"""

import heapq
import os
import subprocess

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- parameters
N = 1000
M = 4000
Q = N / (4 * M)              # spreading probability (paper: q = N / 4M)
DELAY_MEAN = 1000.0          # follower delay, paper lambda_f = 1000
FPS = 30
FRAME_DT = 150.0             # model time units per video frame
INTRO_S, SIM_S, OUTRO_S = 1, 30, 2
SOURCES_PER_S = 1.6          # own-decision trades per second per network (visual)
HUB_EVENT_S = 7.0            # when the best-connected trader acts on its own
MAX_TRADES = 60000

W, H, DPI = 1920, 1080, 100

BG = "white"
EDGE = "#b0b0b0"
NODE = "#9a9a9a"
OWN = "#D55E00"              # own-decision trade
FOL = "#0072B2"              # trade copied from a neighbour
TXT = "black"


# ---------------------------------------------------------------- networks
def lattice(rows=25, cols=40):
    G = nx.Graph()
    idx = lambda r, c: (r % rows) * cols + (c % cols)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                G.add_edge(idx(r, c), idx(r + dr, c + dc))
    pos = {idx(r, c): (c / (cols - 1), 0.6 - r / (cols - 1))
           for r in range(rows) for c in range(cols)}
    # wrap-around links exist in the model but are not drawn
    drawn = [(u, v) for u, v in G.edges()
             if abs(pos[u][0] - pos[v][0]) < 0.1 and abs(pos[u][1] - pos[v][1]) < 0.1]
    return G, pos, drawn


def normalise(pos):
    p = np.array(list(pos.values()))
    p = (p - p.min(0)) / (p.max(0) - p.min(0))
    return {k: tuple(v) for k, v in zip(pos.keys(), p)}


def build_networks():
    Gl, pl, el = lattice()
    Ge = nx.gnm_random_graph(N, M, seed=7)
    pe = normalise(nx.spring_layout(Ge, seed=3, iterations=120, k=0.06))
    Gb = nx.barabasi_albert_graph(N, 4, seed=11)
    pb = normalise(nx.spring_layout(Gb, seed=5, iterations=200, k=0.05))
    return [
        dict(name="Quadratgitter", G=Gl, pos=pl, edges=el),
        dict(name="Erdős–Rényi", G=Ge, pos=pe, edges=list(Ge.edges())),
        dict(name="Skalenfrei (Barabási–Albert)", G=Gb, pos=pb,
             edges=list(Gb.edges())),
    ]


# ---------------------------------------------------------------- simulation
def simulate(G, rng, t_end, hub_time):
    """Event-driven cascade simulation. Returns list of trades
    (time, node, is_own, parent, cascade_id)."""
    nbrs = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
    heap, seq = [], 0
    rate = SOURCES_PER_S / (FPS * FRAME_DT)
    t = rng.exponential(1 / rate)
    cid = 0
    while t < t_end:
        heap.append((t, seq, int(rng.integers(N)), -1, cid)); seq += 1; cid += 1
        t += rng.exponential(1 / rate)
    hub = max(range(N), key=lambda i: len(nbrs[i]))
    heap.append((hub_time, seq, hub, -1, cid)); seq += 1
    hub_cid = cid
    heapq.heapify(heap)

    trades = []
    while heap and len(trades) < MAX_TRADES:
        t, _, i, parent, c = heapq.heappop(heap)
        if t > t_end:
            break
        trades.append((t, i, parent < 0, parent, c))
        for j in nbrs[i]:
            if j != parent and rng.random() < Q:
                heapq.heappush(heap, (t + rng.exponential(DELAY_MEAN), seq, j, i, c)); seq += 1
    return trades, hub, hub_cid


def cascade_sizes(trades):
    sizes = {}
    for _, _, _, _, c in trades:
        sizes[c] = sizes.get(c, 0) + 1
    return sizes


def pick_seed(nets, t_end, hub_time):
    """Find a seed where the BA hub cascade erupts and burns out before the end."""
    for s in range(400):
        rng = np.random.default_rng(1000 + s)
        tr, _, hc = simulate(nets[2]["G"], rng, t_end, hub_time)
        hub_tr = [x for x in tr if x[4] == hc]
        if not hub_tr:
            continue
        last = hub_tr[-1][0]
        others = len(tr) - len(hub_tr)
        if 600 < len(hub_tr) < 6000 and last < t_end * 0.85 and others < 250:
            return 1000 + s, len(hub_tr)
    raise RuntimeError("no suitable seed")


# ---------------------------------------------------------------- rendering
def main():
    nets = build_networks()
    sim_frames = SIM_S * FPS
    t_end = sim_frames * FRAME_DT
    hub_time = HUB_EVENT_S * FPS * FRAME_DT
    seed, hub_size = pick_seed(nets, t_end, hub_time)
    print(f"seed {seed}: BA hub cascade {hub_size} trades")

    for k, net in enumerate(nets):
        rng = np.random.default_rng(seed if k == 2 else seed + 17 * k + 1)
        tr, hub, hc = simulate(net["G"], rng, t_end, hub_time)
        net.update(trades=tr, hub=hub, hub_cid=hc)
        sz = cascade_sizes(tr)
        print(f"{net['name']}: {len(tr)} trades, hub cascade {sz[hc]}, max cascade {max(sz.values())}")

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)

    col_x = [0.02, 0.35, 0.68]
    col_w = 0.30
    for k, net in enumerate(nets):
        ax = fig.add_axes([col_x[k], 0.12, col_w, 0.74], facecolor=BG)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.axis("off")
        if k == 0:
            ax.set_ylim(-0.05, 0.65)
        ax.set_aspect("equal", adjustable="datalim")
        pos = net["pos"]
        P = np.array([pos[i] for i in range(N)])
        segs = [(pos[u], pos[v]) for u, v in net["edges"]]
        ax.add_collection(LineCollection(segs, colors=EDGE, linewidths=0.35,
                                         alpha=0.25 if k else 0.5, zorder=1))
        deg = np.array([net["G"].degree(i) for i in range(N)])
        if k == 0:
            base = np.full(N, 9.0)
        elif k == 1:
            base = 3 + 1.4 * deg
        else:
            base = 4 + 0.9 * deg ** 1.25
        ax.scatter(P[:, 0], P[:, 1], s=base, c=NODE, linewidths=0, zorder=2)
        fig.text(col_x[k] + col_w / 2, 0.92, net["name"], ha="center", va="center", color=TXT, fontsize=22)

        tt = np.array([t for t, *_ in net["trades"]])
        net.update(ax=ax, P=P, base=base,
                   own=np.array([o for _, _, o, _, _ in net["trades"]], bool),
                   frame_of=(tt / FRAME_DT).astype(int),
                   node=np.array([n for _, n, *_ in net["trades"]]),
                   parent=np.array([p for _, _, _, p, _ in net["trades"]]))

    # legend
    handles = [
        Line2D([], [], marker="o", ls="none", markersize=9, markerfacecolor=OWN, markeredgecolor=OWN,
               label="Handel aus eigener Entscheidung"),
        Line2D([], [], marker="o", ls="-", lw=1.6, markersize=9, color=FOL,
               label="Vom Nachbarn kopierter Handel"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=18,
               bbox_to_anchor=(0.5, 0.02), handlelength=2.2, columnspacing=3)

    # dynamic artists
    for net in nets:
        ax = net["ax"]
        net["glow"] = ax.scatter(net["P"][:, 0], net["P"][:, 1], s=np.zeros(N), c=np.zeros((N, 4)),
                                 linewidths=0, zorder=4)
        net["rings"] = ax.scatter([], [], s=[], facecolors="none", edgecolors=OWN, linewidths=2.0, zorder=5)
        net["traces"] = LineCollection([], linewidths=1.3, zorder=3)
        ax.add_collection(net["traces"])

    own_rgb = np.array(matplotlib.colors.to_rgb(OWN))
    fol_rgb = np.array(matplotlib.colors.to_rgb(FOL))
    TAU_OWN, TAU_FOL, TAU_RING, TAU_LINE = 14.0, 9.0, 16, 10  # frames

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = os.path.join(OUT_DIR, "trade_contagion.mp4")
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", mp4],
                            stdin=subprocess.PIPE)

    intro_f, total_f = INTRO_S * FPS, (INTRO_S + SIM_S + OUTRO_S) * FPS
    for f in range(total_f):
        sf = min(max(f - intro_f, -1), sim_frames - 1)  # simulation frame (-1 = not started)
        for net in nets:
            if sf < 0:
                continue
            fo = net["frame_of"]
            idx = np.nonzero((fo <= sf) & (fo > sf - 3 * TAU_OWN))[0]
            a_own, a_fol = np.zeros(N), np.zeros(N)
            for i in idx:
                age = sf - fo[i]
                if net["own"][i]:
                    a_own[net["node"][i]] = max(a_own[net["node"][i]], np.exp(-age / TAU_OWN))
                else:
                    a_fol[net["node"][i]] = max(a_fol[net["node"][i]], np.exp(-age / TAU_FOL))
            amp = np.maximum(a_own, a_fol)
            col = np.zeros((N, 4))
            col[:, :3] = np.where((a_own >= a_fol)[:, None], own_rgb, fol_rgb)
            col[:, 3] = np.clip(amp * 1.2, 0, 1)
            net["glow"].set_facecolors(col)
            net["glow"].set_sizes(amp * (net["base"] * 3 + 40))

            ri = [i for i in idx if net["own"][i] and sf - fo[i] < TAU_RING]
            ages = np.array([sf - fo[i] for i in ri])
            net["rings"].set_offsets(net["P"][net["node"][ri]] if ri else np.empty((0, 2)))
            net["rings"].set_sizes(120 + 110 * ages if ri else [])
            ec = np.tile(np.append(own_rgb, 1.0), (len(ri), 1))
            if ri:
                ec[:, 3] = 1 - 0.8 * ages / TAU_RING
            net["rings"].set_edgecolors(ec)

            li = [i for i in idx if not net["own"][i] and sf - fo[i] < TAU_LINE]
            segs = [(net["P"][net["parent"][i]], net["P"][net["node"][i]]) for i in li]
            lc = np.tile(np.append(fol_rgb, 1.0), (len(li), 1))
            if li:
                lc[:, 3] = 0.9 * (1 - np.array([sf - fo[i] for i in li]) / TAU_LINE)
            net["traces"].set_segments(segs); net["traces"].set_colors(lc)

        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
        if f % 150 == 0:
            print(f"frame {f}/{total_f}")
    proc.stdin.close(); proc.wait()
    print("wrote", mp4)


if __name__ == "__main__":
    main()
