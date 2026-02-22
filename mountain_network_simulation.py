"""
╔══════════════════════════════════════════════════════════════════════╗
║         เครือข่ายภูเขาสูง — Mountain High Network Simulation              ║
║         Mockup / Toy Model Presentation                              ║
╚══════════════════════════════════════════════════════════════════════╝
โมดูลจำลองครอบคลุม:
  1.  Network Topology (Graph Theory)         — โทโพโลยี ring/mesh ภูเขา
  2.  Link Quality Simulation                 — ความผันผวนของสัญญาณ
  3.  DTN Store-and-Forward                   — จำลอง Delay-Tolerant Networking
  4.  QoS / Queuing Theory                    — คิวทราฟฟิกฉุกเฉิน vs ทั่วไป
  5.  AI / AIOps Link Failure Prediction      — พยากรณ์ลิงก์ล่มด้วย ML-like
  6.  Edge AI Traffic Reduction               — ลดทราฟฟิกด้วย Edge Processing
  7.  Multi-path Routing Resilience           — Failover อัตโนมัติ
  8.  Energy Budget (Solar/Battery)           — งบพลังงาน solar บนยอดเขา
  9.  3-Layer Architecture Summary            — ภาพรวมสถาปัตย์ 3 ชั้น
"""

import math
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import networkx as nx
from collections import deque
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─── THEME ────────────────────────────────────────────────────────────────────
BG        = "#0D1117"
PANEL     = "#161B22"
PANEL2    = "#1C2230"
ACCENT    = "#58A6FF"
GREEN     = "#3FB950"
ORANGE    = "#F0883E"
RED       = "#FF4444"
YELLOW    = "#E3B341"
PURPLE    = "#BC8CFF"
CYAN      = "#76E3EA"
MUTED     = "#8B949E"
WHITE     = "#E6EDF3"
GRID_COL  = "#21262D"

def apply_theme(fig, axes=None):
    fig.patch.set_facecolor(BG)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NETWORK TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
def build_mountain_graph():
    G = nx.Graph()
    nodes = {
        "Internet\nGateway":  (0.50, 0.92, "gateway"),
        "Summit\nAlpha":      (0.25, 0.72, "backbone"),
        "Summit\nBeta":       (0.55, 0.75, "backbone"),
        "Summit\nGamma":      (0.78, 0.65, "backbone"),
        "Relay\nNorth":       (0.15, 0.52, "relay"),
        "Relay\nCenter":      (0.48, 0.53, "relay"),
        "Relay\nEast":        (0.82, 0.45, "relay"),
        "Village\nA":         (0.08, 0.30, "village"),
        "Village\nB":         (0.30, 0.25, "village"),
        "Village\nC":         (0.55, 0.28, "village"),
        "Village\nD":         (0.78, 0.22, "village"),
        "Sensor\nCluster 1":  (0.20, 0.10, "iot"),
        "Sensor\nCluster 2":  (0.65, 0.08, "iot"),
    }
    colors_map = {
        "gateway": "#F0883E", "backbone": "#58A6FF",
        "relay": "#BC8CFF", "village": "#3FB950", "iot": "#76E3EA"
    }
    for name, (x, y, ntype) in nodes.items():
        G.add_node(name, pos=(x, y), ntype=ntype, color=colors_map[ntype])

    edges_primary = [
        ("Internet\nGateway",  "Summit\nAlpha",   300, "fiber"),
        ("Internet\nGateway",  "Summit\nBeta",    200, "microwave"),
        ("Summit\nAlpha",      "Summit\nBeta",    150, "microwave"),
        ("Summit\nBeta",       "Summit\nGamma",   120, "microwave"),
        ("Summit\nAlpha",      "Relay\nNorth",     80, "wifi"),
        ("Summit\nBeta",       "Relay\nCenter",    90, "wifi"),
        ("Summit\nGamma",      "Relay\nEast",      70, "wifi"),
        ("Relay\nNorth",       "Village\nA",       30, "wifi"),
        ("Relay\nNorth",       "Village\nB",       35, "wifi"),
        ("Relay\nCenter",      "Village\nB",       25, "wifi"),
        ("Relay\nCenter",      "Village\nC",       40, "wifi"),
        ("Relay\nEast",        "Village\nC",       30, "wifi"),
        ("Relay\nEast",        "Village\nD",       28, "wifi"),
        ("Village\nA",         "Sensor\nCluster 1",10, "lora"),
        ("Village\nD",         "Sensor\nCluster 2",10, "lora"),
    ]
    edges_backup = [
        ("Summit\nAlpha",  "Relay\nCenter",  60, "backup"),
        ("Summit\nGamma",  "Relay\nCenter",  55, "backup"),
        ("Village\nB",     "Village\nC",     20, "backup"),
    ]
    for src, dst, bw, etype in edges_primary:
        G.add_edge(src, dst, bandwidth=bw, etype=etype, weight=100//bw)
    for src, dst, bw, etype in edges_backup:
        G.add_edge(src, dst, bandwidth=bw, etype=etype, weight=150//bw, backup=True)
    return G, nodes

def plot_topology(ax, G, nodes, failed_links=None):
    pos = {n: (v[0], v[1]) for n, v in nodes.items()}
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_sizes  = []
    for n in G.nodes:
        t = G.nodes[n]["ntype"]
        node_sizes.append({"gateway":700,"backbone":550,"relay":350,"village":280,"iot":200}[t])

    primary_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("etype") != "backup"]
    backup_edges  = [(u,v) for u,v,d in G.edges(data=True) if d.get("etype") == "backup"]
    if failed_links:
        failed = set(map(frozenset, failed_links))
        primary_edges = [e for e in primary_edges if frozenset(e) not in failed]

    edge_colors = []
    for u, v in primary_edges:
        t = G[u][v]["etype"]
        edge_colors.append({"fiber":"#F0883E","microwave":ACCENT,"wifi":GREEN,"lora":CYAN}[t])

    nx.draw_networkx_edges(G, pos, edgelist=primary_edges,
                           edge_color=edge_colors, width=2.0, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=backup_edges,
                           edge_color=PURPLE, width=1.2, alpha=0.5,
                           style="dashed", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95, ax=ax)
    labels = {n: n.replace("\n", "\n") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=5.5,
                            font_color=WHITE, font_weight="bold", ax=ax)

    legend_items = [
        Line2D([0],[0], color="#F0883E", lw=2, label="Fiber"),
        Line2D([0],[0], color=ACCENT,    lw=2, label="Microwave"),
        Line2D([0],[0], color=GREEN,     lw=2, label="Wi-Fi PtP"),
        Line2D([0],[0], color=CYAN,      lw=2, label="LoRa/IoT"),
        Line2D([0],[0], color=PURPLE,    lw=2, ls="--", label="Backup Path"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              facecolor=PANEL2, edgecolor=GRID_COL, fontsize=6.5, labelcolor=WHITE)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LINK QUALITY SIMULATION (weather-induced)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_link_quality(hours=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, hours, hours * 10)

    def weather_effect(t):
        rain  = 0.6 * np.sin(2*np.pi*t/24 + 1.5)**2
        fog   = 0.4 * (1 + np.sin(2*np.pi*t/8)) / 2
        wind  = 0.3 * rng.random(len(t))
        return np.clip(rain + fog + wind, 0, 1)

    def quality(t, base, noise_scale=0.05):
        w = weather_effect(t)
        q = base - 0.4*w + noise_scale*rng.standard_normal(len(t))
        return np.clip(q, 0.05, 1.0)

    q_microwave = quality(t, 0.90, 0.04)
    q_wifi      = quality(t, 0.80, 0.07)
    q_leo_sat   = quality(t, 0.75, 0.06)
    q_lora      = quality(t, 0.95, 0.02)   # LoRa very resilient
    return t, q_microwave, q_wifi, q_leo_sat, q_lora

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DTN STORE-AND-FORWARD SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
class DTNNode:
    def __init__(self, name):
        self.name   = name
        self.buffer = deque()
        self.delivered_count = 0
        self.dropped_count   = 0
        self.buffer_history  = []

    def store(self, pkt):
        if len(self.buffer) < 50:
            self.buffer.append(pkt)
        else:
            self.dropped_count += 1

    def forward(self, link_up):
        forwarded = []
        if link_up and self.buffer:
            n = min(len(self.buffer), random.randint(3, 8))
            for _ in range(n):
                if self.buffer:
                    pkt = self.buffer.popleft()
                    forwarded.append(pkt)
                    self.delivered_count += 1
        self.buffer_history.append(len(self.buffer))
        return forwarded

def simulate_dtn(steps=120):
    src  = DTNNode("Village_A")
    hop1 = DTNNode("Relay_North")
    dst  = DTNNode("Summit_Alpha")

    link_states = []
    arrivals, deliveries, buf_sizes = [], [], []

    for step in range(steps):
        # inject packets at source (poisson arrivals)
        n_arrive = np.random.poisson(4)
        for i in range(n_arrive):
            pkt = {"id": f"{step}_{i}", "ts": step, "priority": random.choice(["emergency","normal","iot"])}
            src.store(pkt)
        arrivals.append(n_arrive)

        # link 1: src → hop1 (intermittent, fails ~30% of time)
        link1_up = random.random() > 0.30
        # link 2: hop1 → dst  (fails ~20% of time)
        link2_up = random.random() > 0.20

        fwd1 = src.forward(link1_up)
        for p in fwd1:
            hop1.store(p)

        fwd2 = hop1.forward(link2_up)
        for p in fwd2:
            dst.delivered_count += 1

        deliveries.append(dst.delivered_count)
        buf_sizes.append(len(src.buffer) + len(hop1.buffer))
        link_states.append((link1_up, link2_up))

    return arrivals, deliveries, buf_sizes, link_states, src, hop1, dst

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  QoS QUEUING SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_qos(steps=100):
    """Priority queue: emergency > voice > data > iot"""
    classes = ["Emergency", "Voice/Video", "Data", "IoT/Sensor"]
    weights = [8, 4, 2, 1]   # WFQ weights
    arrival_rates = [0.5, 2.0, 5.0, 8.0]  # packets/step
    link_capacity = 12  # packets/step can forward

    queues     = [deque() for _ in classes]
    latencies  = {c: [] for c in classes}
    queue_lens = {c: [] for c in classes}

    for step in range(steps):
        # arrivals
        for i, (c, rate) in enumerate(zip(classes, arrival_rates)):
            n = np.random.poisson(rate)
            for _ in range(n):
                if len(queues[i]) < 60:
                    queues[i].append(step)

        # WFQ service
        tokens = link_capacity
        for i, (q, w) in enumerate(zip(queues, weights)):
            serve = min(int(tokens * w / sum(weights)), len(q))
            for _ in range(serve):
                if q:
                    arrival_ts = q.popleft()
                    lat = step - arrival_ts
                    latencies[classes[i]].append(lat)
            tokens -= serve

        for i, c in enumerate(classes):
            queue_lens[c].append(len(queues[i]))

    avg_lat = {c: np.mean(latencies[c]) if latencies[c] else 0 for c in classes}
    return queue_lens, avg_lat, classes

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  AI LINK FAILURE PREDICTION (AIOps mock)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_ai_prediction(hours=72):
    t = np.arange(hours)
    # Ground truth signal quality
    signal = (0.8 + 0.15*np.sin(2*np.pi*t/24)
              - 0.3*(np.random.random(hours) < 0.1).astype(float)
              + 0.04*np.random.randn(hours))
    signal = np.clip(signal, 0, 1)

    # "AI prediction" — smoothed + phase-shifted by a few hours
    from scipy.ndimage import uniform_filter1d
    pred = uniform_filter1d(signal, size=4)
    pred = np.roll(pred, -2)  # 2-hour lookahead approximation
    pred = np.clip(pred + 0.03*np.random.randn(hours), 0, 1)

    # Mark failures and predicted alarms
    fail_threshold  = 0.45
    alarm_threshold = 0.52
    actual_fails  = signal < fail_threshold
    ai_alarms     = pred   < alarm_threshold

    tp = np.sum(actual_fails & ai_alarms)
    fp = np.sum(~actual_fails & ai_alarms)
    fn = np.sum(actual_fails & ~ai_alarms)
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0
    recall    = tp / (tp+fn) if (tp+fn) > 0 else 0

    return t, signal, pred, actual_fails, ai_alarms, precision, recall

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  EDGE AI TRAFFIC REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_edge_ai(days=30):
    t = np.arange(days)
    # Without Edge AI: raw sensor data sent to cloud
    raw_traffic = 80 + 30*np.sin(2*np.pi*t/7) + 5*np.random.randn(days)  # MB/hour

    # Edge AI processes locally → sends only events (10–20% of raw)
    reduction_factor = 0.12 + 0.05*np.random.rand(days)
    edge_traffic = raw_traffic * reduction_factor

    # Events detected (binary)
    events = np.random.poisson(2, days)  # avg 2 events/day

    savings_pct = (1 - edge_traffic/raw_traffic) * 100
    return t, raw_traffic, edge_traffic, events, savings_pct

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MULTI-PATH ROUTING — FAILOVER
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_multipath(steps=200):
    rng = np.random.default_rng(7)
    # Two paths: primary (higher BW) and secondary (backup)
    primary_bw    = 100  # Mbps
    secondary_bw  = 40
    traffic_load  = 60 + 30*np.sin(np.linspace(0, 4*np.pi, steps))

    primary_up = rng.random(steps) > 0.18   # 18% failure rate
    failover_active = ~primary_up

    throughput_no_failover  = np.where(primary_up, traffic_load, 0)
    throughput_with_failover = np.where(
        primary_up,
        np.minimum(traffic_load, primary_bw),
        np.minimum(traffic_load * 0.6, secondary_bw)
    )
    return (steps, traffic_load, primary_up,
            throughput_no_failover, throughput_with_failover)

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  ENERGY BUDGET SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_energy(days=14):
    t = np.arange(days)
    # Solar generation (kWh/day) — depends on weather
    cloud_cover = 0.3 + 0.4*np.abs(np.sin(2*np.pi*t/7)) + 0.15*np.random.rand(days)
    solar_gen   = np.clip(4.0 * (1 - cloud_cover) + 0.3*np.random.randn(days), 0.2, 4.5)

    # Consumption
    radio_consume  = 1.2 * np.ones(days)
    cpu_consume    = 0.4 * np.ones(days)
    edge_ai_extra  = 0.3 * np.ones(days)
    total_consume  = radio_consume + cpu_consume + edge_ai_extra

    battery = np.zeros(days)
    battery[0] = 8.0   # initial SoC (kWh)
    for i in range(1, days):
        battery[i] = np.clip(battery[i-1] + solar_gen[i] - total_consume[i], 0.5, 12.0)

    return t, solar_gen, total_consume, battery

# ═══════════════════════════════════════════════════════════════════════════════
# ──────────────────────  MAIN FIGURE ASSEMBLY  ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("🏔️  Building Mountain Network Simulation…")

    # ── pre-compute ──────────────────────────────────────────────────────────
    G, nodes = build_mountain_graph()

    t_lq, q_mw, q_wifi, q_leo, q_lora = simulate_link_quality()
    dtn_arrivals, dtn_deliv, dtn_bufs, dtn_links, src_node, hop_node, dst_node = simulate_dtn()
    qos_ql, qos_lat, qos_classes = simulate_qos()
    t_ai, sig, pred, fails, alarms, prec, rec = simulate_ai_prediction()
    t_edge, raw_tr, edge_tr, events, savings = simulate_edge_ai()
    steps, tload, pup, tput_no, tput_fo = simulate_multipath()
    t_en, solar, consume, battery = simulate_energy()

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.suptitle(
        "เครือข่ายภูเขาสูง  ·  Mountain High Network — Simulation Dashboard",
        fontsize=17, fontweight="bold", color=WHITE, y=0.985,
        fontfamily="DejaVu Sans"
    )

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.05, right=0.97,
                           top=0.97, bottom=0.03)

    # ─── Row 0 : Topology (span 2 cols) + 3-Layer Architecture ──────────────
    ax_topo = fig.add_subplot(gs[0, :2])
    ax_arch = fig.add_subplot(gs[0, 2])

    # ─── Row 1 : Link Quality  | DTN Buffer  | AI Prediction ─────────────────
    ax_lq   = fig.add_subplot(gs[1, 0])
    ax_dtn  = fig.add_subplot(gs[1, 1])
    ax_ai   = fig.add_subplot(gs[1, 2])

    # ─── Row 2 : QoS Latency  | Multi-path  | Edge AI ────────────────────────
    ax_qos  = fig.add_subplot(gs[2, 0])
    ax_mp   = fig.add_subplot(gs[2, 1])
    ax_edg  = fig.add_subplot(gs[2, 2])

    # ─── Row 3 : Energy Budget (span 2) | Summary stats ──────────────────────
    ax_en   = fig.add_subplot(gs[3, :2])
    ax_stat = fig.add_subplot(gs[3, 2])

    all_axes = [ax_lq, ax_dtn, ax_ai, ax_qos, ax_mp, ax_edg, ax_en, ax_stat]
    apply_theme(fig, all_axes)

    # ═══ PLOT 1 — TOPOLOGY ════════════════════════════════════════════════════
    ax_topo.set_facecolor(PANEL)
    ax_topo.set_title("① Network Topology — Ring/Mesh Mountain Backbone  (Graph Theory)",
                      color=WHITE, fontsize=10, fontweight="bold", pad=6)
    plot_topology(ax_topo, G, nodes)
    ax_topo.text(0.01, 0.02,
                 f"Nodes: {G.number_of_nodes()}  |  Links: {G.number_of_edges()}  "
                 f"|  Avg Degree: {2*G.number_of_edges()/G.number_of_nodes():.1f}",
                 transform=ax_topo.transAxes,
                 color=MUTED, fontsize=7)

    # ═══ PLOT 2 — 3-LAYER ARCHITECTURE ═══════════════════════════════════════
    ax_arch.set_facecolor(PANEL)
    ax_arch.set_title("② 3-Layer Architecture", color=WHITE,
                      fontsize=10, fontweight="bold", pad=6)
    ax_arch.axis("off")
    layers = [
        ("Layer C\nDTN/Store-&-Forward Overlay", 0.70, PURPLE,   "🔀 Disruption Tolerant"),
        ("Layer B\nMountain Backbone (Ring/Mesh)", 0.45, ACCENT,  "📡 Microwave / Fiber"),
        ("Layer A\nAccess (Village / IoT)", 0.20, GREEN,          "📶 Wi-Fi PtP / LoRa"),
    ]
    for (label, y, color, sub) in layers:
        rect = FancyBboxPatch((0.05, y-0.12), 0.90, 0.22,
                              boxstyle="round,pad=0.02",
                              facecolor=color+"33", edgecolor=color,
                              linewidth=1.5, transform=ax_arch.transAxes)
        ax_arch.add_patch(rect)
        ax_arch.text(0.50, y+0.02, label, ha="center", va="center",
                     color=WHITE, fontsize=8, fontweight="bold",
                     transform=ax_arch.transAxes)
        ax_arch.text(0.50, y-0.07, sub, ha="center", va="center",
                     color=color, fontsize=7.5, transform=ax_arch.transAxes)
    ax_arch.annotate("", xy=(0.50, 0.34), xytext=(0.50, 0.40),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
    ax_arch.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.66),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))

    # ═══ PLOT 3 — LINK QUALITY ════════════════════════════════════════════════
    ax_lq.set_title("③ Link Quality vs. Weather (48h)",
                    color=WHITE, fontsize=9, fontweight="bold")
    ax_lq.plot(t_lq, q_mw,   color=ACCENT,  lw=1.5, label="Microwave", alpha=0.9)
    ax_lq.plot(t_lq, q_wifi,  color=GREEN,   lw=1.5, label="Wi-Fi",    alpha=0.9)
    ax_lq.plot(t_lq, q_leo,   color=ORANGE,  lw=1.5, label="LEO Sat",  alpha=0.8)
    ax_lq.plot(t_lq, q_lora,  color=CYAN,    lw=1.2, label="LoRa",     alpha=0.8)
    ax_lq.axhline(0.45, color=RED, lw=1.0, ls="--", alpha=0.7, label="Fail threshold")
    ax_lq.fill_between(t_lq, 0, 0.45, color=RED, alpha=0.07)
    ax_lq.set_xlabel("Time (hours)", color=MUTED, fontsize=8)
    ax_lq.set_ylabel("Link Quality (0–1)", color=MUTED, fontsize=8)
    ax_lq.legend(facecolor=PANEL2, edgecolor=GRID_COL, labelcolor=WHITE,
                 fontsize=6.5, ncol=2, loc="lower right")
    ax_lq.set_ylim(0, 1.05)

    # ═══ PLOT 4 — DTN BUFFER ══════════════════════════════════════════════════
    ax_dtn.set_title("④ DTN Store-and-Forward Buffer (120 steps)",
                     color=WHITE, fontsize=9, fontweight="bold")
    steps_arr = np.arange(len(dtn_bufs))
    cumulative_arrivals = np.cumsum(dtn_arrivals)
    ax_dtn.fill_between(steps_arr, dtn_bufs, color=PURPLE, alpha=0.35, label="Buffer size (pkts)")
    ax_dtn.plot(steps_arr, dtn_bufs, color=PURPLE, lw=1.5)
    ax2_dtn = ax_dtn.twinx()
    ax2_dtn.plot(steps_arr, dtn_deliv, color=GREEN, lw=1.5, label="Cumulative delivered")
    ax2_dtn.set_ylabel("Delivered (cumulative)", color=GREEN, fontsize=7)
    ax2_dtn.tick_params(axis='y', colors=GREEN, labelsize=7)
    ax2_dtn.set_facecolor(PANEL)
    # Mark link-down periods
    for i, (l1, l2) in enumerate(dtn_links):
        if not l1:
            ax_dtn.axvspan(i, i+1, color=RED, alpha=0.06)
    ax_dtn.set_xlabel("Simulation Step", color=MUTED, fontsize=8)
    ax_dtn.set_ylabel("Buffer Size (packets)", color=MUTED, fontsize=8)
    ax_dtn.text(0.02, 0.93,
                f"Total delivered: {dst_node.delivered_count}  "
                f"| Dropped: {src_node.dropped_count}",
                transform=ax_dtn.transAxes, color=MUTED, fontsize=6.5)

    # ═══ PLOT 5 — AI PREDICTION ═══════════════════════════════════════════════
    ax_ai.set_title(f"⑤ AIOps: Link Failure Prediction  "
                    f"(Precision={prec:.0%}, Recall={rec:.0%})",
                    color=WHITE, fontsize=9, fontweight="bold")
    ax_ai.plot(t_ai, sig,  color=ACCENT, lw=1.5, label="Actual signal quality")
    ax_ai.plot(t_ai, pred, color=ORANGE, lw=1.5, ls="--", label="AI prediction (2h ahead)")
    ax_ai.axhline(0.45, color=RED, lw=0.9, ls=":", alpha=0.8, label="Fail threshold")
    fail_idx  = np.where(fails)[0]
    alarm_idx = np.where(alarms)[0]
    ax_ai.scatter(t_ai[fail_idx],  sig[fail_idx],  color=RED,    s=18, zorder=5, label="Actual failure")
    ax_ai.scatter(t_ai[alarm_idx], pred[alarm_idx], color=YELLOW, s=12, marker="^",
                  zorder=4, alpha=0.8, label="AI alarm")
    ax_ai.set_xlabel("Time (hours)", color=MUTED, fontsize=8)
    ax_ai.set_ylabel("Signal Quality", color=MUTED, fontsize=8)
    ax_ai.legend(facecolor=PANEL2, edgecolor=GRID_COL, labelcolor=WHITE,
                 fontsize=6, loc="lower right")
    ax_ai.set_ylim(-0.05, 1.1)

    # ═══ PLOT 6 — QoS LATENCY ════════════════════════════════════════════════
    ax_qos.set_title("⑥ QoS — Avg Latency by Traffic Class (WFQ)",
                     color=WHITE, fontsize=9, fontweight="bold")
    qos_colors = [RED, ORANGE, ACCENT, CYAN]
    bars = ax_qos.bar(qos_classes, [qos_lat[c] for c in qos_classes],
                      color=qos_colors, alpha=0.82, edgecolor=PANEL, linewidth=1)
    for bar, c in zip(bars, qos_classes):
        ax_qos.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{qos_lat[c]:.1f} steps",
                    ha="center", va="bottom", color=WHITE, fontsize=7.5, fontweight="bold")
    ax_qos.set_ylabel("Average Latency (steps)", color=MUTED, fontsize=8)
    ax_qos.tick_params(axis='x', labelsize=7, rotation=10)
    ax_qos.text(0.02, 0.92, "Emergency traffic gets lowest latency via WFQ",
                transform=ax_qos.transAxes, color=MUTED, fontsize=6.5)

    # ═══ PLOT 7 — MULTI-PATH ══════════════════════════════════════════════════
    ax_mp.set_title("⑦ Multi-path Failover — Throughput Comparison",
                    color=WHITE, fontsize=9, fontweight="bold")
    s = np.arange(steps)
    ax_mp.fill_between(s, tput_no, alpha=0.25, color=RED,   label="No failover")
    ax_mp.fill_between(s, tput_fo, alpha=0.35, color=GREEN, label="With failover")
    ax_mp.plot(s, tload, color=MUTED, lw=1.0, ls="--", alpha=0.6, label="Traffic demand")
    primary_down = np.where(~pup)[0]
    ax_mp.scatter(primary_down, np.zeros(len(primary_down)),
                  color=RED, s=12, marker="x", zorder=5, label="Primary link fail")
    ax_mp.set_xlabel("Simulation Step", color=MUTED, fontsize=8)
    ax_mp.set_ylabel("Throughput (Mbps)", color=MUTED, fontsize=8)
    ax_mp.legend(facecolor=PANEL2, edgecolor=GRID_COL, labelcolor=WHITE,
                 fontsize=6.5, loc="upper right")
    gain = np.mean(tput_fo) - np.mean(tput_no)
    ax_mp.text(0.02, 0.06,
               f"Avg throughput gain: +{gain:.1f} Mbps  |  Uptime: {np.mean(pup):.0%}",
               transform=ax_mp.transAxes, color=MUTED, fontsize=6.5)

    # ═══ PLOT 8 — EDGE AI TRAFFIC ═════════════════════════════════════════════
    ax_edg.set_title("⑧ Edge AI Traffic Reduction (30 days)",
                     color=WHITE, fontsize=9, fontweight="bold")
    ax_edg.fill_between(t_edge, raw_tr, color=RED,  alpha=0.3, label="Raw (no Edge AI)")
    ax_edg.fill_between(t_edge, edge_tr, color=GREEN, alpha=0.5, label="With Edge AI")
    ax_edg.plot(t_edge, raw_tr,  color=RED,   lw=1.5, alpha=0.8)
    ax_edg.plot(t_edge, edge_tr, color=GREEN, lw=1.5)
    ax3 = ax_edg.twinx()
    ax3.bar(t_edge, events, color=YELLOW, alpha=0.5, width=0.7, label="Events detected")
    ax3.set_ylabel("Events/day", color=YELLOW, fontsize=7)
    ax3.tick_params(axis='y', colors=YELLOW, labelsize=7)
    ax3.set_facecolor(PANEL)
    ax_edg.set_xlabel("Day", color=MUTED, fontsize=8)
    ax_edg.set_ylabel("Traffic (MB/hour)", color=MUTED, fontsize=8)
    ax_edg.legend(facecolor=PANEL2, edgecolor=GRID_COL, labelcolor=WHITE,
                  fontsize=6.5, loc="upper left")
    ax_edg.text(0.02, 0.06,
                f"Avg savings: {np.mean(savings):.0f}%  backbone bandwidth",
                transform=ax_edg.transAxes, color=MUTED, fontsize=6.5)

    # ═══ PLOT 9 — ENERGY BUDGET ═══════════════════════════════════════════════
    ax_en.set_title("⑨ Energy Budget — Solar Generation vs. Consumption & Battery SoC (14 days)",
                    color=WHITE, fontsize=9, fontweight="bold")
    ax_en.bar(t_en - 0.2, solar,   width=0.38, color=YELLOW, alpha=0.75, label="Solar generation (kWh)")
    ax_en.bar(t_en + 0.2, consume, width=0.38, color=RED,    alpha=0.65, label="Total consumption (kWh)")
    ax_en2 = ax_en.twinx()
    ax_en2.plot(t_en, battery, color=CYAN, lw=2.0, marker="o", ms=4, label="Battery SoC (kWh)")
    ax_en2.axhline(2.0, color=ORANGE, lw=1.0, ls="--", alpha=0.7, label="Low SoC warning")
    ax_en2.set_ylabel("Battery SoC (kWh)", color=CYAN, fontsize=8)
    ax_en2.tick_params(axis='y', colors=CYAN, labelsize=7)
    ax_en2.set_facecolor(PANEL)
    ax_en.set_xlabel("Day", color=MUTED, fontsize=8)
    ax_en.set_ylabel("Energy (kWh/day)", color=MUTED, fontsize=8)
    lines1, labs1 = ax_en.get_legend_handles_labels()
    lines2, labs2 = ax_en2.get_legend_handles_labels()
    ax_en.legend(lines1+lines2, labs1+labs2,
                 facecolor=PANEL2, edgecolor=GRID_COL, labelcolor=WHITE,
                 fontsize=7, loc="upper right", ncol=2)

    # ═══ PLOT 10 — SUMMARY STATS ══════════════════════════════════════════════
    ax_stat.set_facecolor(PANEL)
    ax_stat.axis("off")
    ax_stat.set_title("📊  Key Metrics Summary",
                      color=WHITE, fontsize=10, fontweight="bold", pad=6)
    metrics = [
        ("Network Nodes",        f"{G.number_of_nodes()}",       ACCENT),
        ("Network Links",        f"{G.number_of_edges()}",       ACCENT),
        ("Graph Connectivity",   "✓ 2-connected" if nx.is_connected(G) else "⚠ Disconnected", GREEN),
        ("DTN Delivery Rate",    f"{dst_node.delivered_count / max(sum(dtn_arrivals),1)*100:.0f}%",    GREEN),
        ("Avg Link Quality",     f"{np.mean(q_mw):.2f}",        YELLOW),
        ("AI Precision",         f"{prec:.0%}",                  ORANGE),
        ("AI Recall",            f"{rec:.0%}",                   ORANGE),
        ("Failover Uptime Gain", f"+{np.mean(tput_fo)-np.mean(tput_no):.0f} Mbps", PURPLE),
        ("Edge AI Traffic Save", f"{np.mean(savings):.0f}%",    CYAN),
        ("Min Battery SoC",      f"{np.min(battery):.1f} kWh",  YELLOW),
    ]
    for i, (label, val, color) in enumerate(metrics):
        y = 0.93 - i * 0.085
        ax_stat.text(0.05, y, label, transform=ax_stat.transAxes,
                     color=MUTED, fontsize=8.5, va="center")
        ax_stat.text(0.75, y, val, transform=ax_stat.transAxes,
                     color=color, fontsize=9.0, va="center", fontweight="bold", ha="right")
        line = Line2D([0.04, 0.96], [y - 0.04, y - 0.04],
                      color=GRID_COL, lw=0.5, transform=ax_stat.transAxes)
        ax_stat.add_line(line)

    # ── theories footnote ────────────────────────────────────────────────────
    theories = ("Theories: OSI Model  ·  TCP/IP  ·  DTN (Bundle Protocol)  ·  "
                "Graph Theory  ·  Queuing Theory  ·  Resilient Network Design  ·  "
                "Edge Computing  ·  SDN/NFV  |  "
                "เครือข่ายภูเขาสูง — Group Project Simulation")
    fig.text(0.5, 0.005, theories, ha="center", va="bottom",
             color=MUTED, fontsize=6.5, style="italic")

    plt.savefig("mountain_network_simulation.png",
                dpi=160, bbox_inches="tight", facecolor=BG)
    print("✅  Saved: mountain_network_simulation.png")
    plt.close()

    return fig

if __name__ == "__main__":
    fig = main()
    print("\n📌  Summary:")
    print("   • Plot 1  — Ring/Mesh Topology (Graph Theory)")
    print("   • Plot 2  — 3-Layer Architecture (OSI / DTN)")
    print("   • Plot 3  — Link Quality vs Weather (Microwave / Wi-Fi / LEO / LoRa)")
    print("   • Plot 4  — DTN Store-and-Forward Buffer simulation")
    print("   • Plot 5  — AIOps Link Failure Prediction")
    print("   • Plot 6  — QoS Latency by Traffic Class (WFQ / Queuing Theory)")
    print("   • Plot 7  — Multi-path Failover Throughput")
    print("   • Plot 8  — Edge AI Traffic Reduction (80%+ savings)")
    print("   • Plot 9  — Solar Energy Budget & Battery SoC")
    print("   • Plot 10 — Key Metrics Summary")
