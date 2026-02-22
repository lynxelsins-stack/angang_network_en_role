"""
══════════════════════════════════════════════════════════════════════════
เครือข่ายภูเขาสูง — Interactive Network Simulator                           
ก่อนอื่นเบื้องต้น ขอให้แน่ใจว่าคุณได้ติดตั้ง pygame แล้ว (pip install pygame) 
จากนั้นรันสคริปต์นี้เพื่อเปิดตัวจำลองเครือข่ายภูเขาสูงแบบอินเทอร์แอคทีฟ 
ที่ซึ่งคุณสามารถส่งข้อความผ่านเครือข่ายที่มีสัญญาณขาดและดูการทำงานของ DTN ได้แบบเรียลไทม์นะจ๊ะ!


ติดตั้ง:  python -m pip install pygame-ce
       python -m pip install --upgrade setuptools wheel
                                          
รัน    :  python mountain_network_interactive.py                       
══════════════════════════════════════════════════════════════════════════
วิธีใช้:                                                                
• คลิกโหนด           — เลือกโหนดปลายทาง                                  
• พิมพ์ข้อความ         — กด Enter เพื่อส่ง packet                           
• คลิกขวาโหนด        — ตัด/ต่อสัญญาณลิงก์ (จำลองสัญญาณขาด)             
• กด [S]            — สุ่มสัญญาณขาดอัตโนมัติ                            
• กด [R]            — รีเซ็ตทุกลิงก์ให้ปกติ                             
• กด [C]            — ล้าง log                                          
"""

import pygame
import sys
import math
import random
import time
from collections import deque

# ─── ติดตั้ง pygame ก่อนรัน ──────────────────────────────────────────────────
try:
    import pygame
except ImportError:
    print("❌  program didn't found pygame  →  run:  pip install pygame-ce  and try again")
    sys.exit(1)

pygame.init()
pygame.font.init()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
W, H         = 1280, 780
FPS          = 60
PANEL_W      = 340   # right-side panel width

# Colors
BG           = (13,  17,  23)
PANEL_BG     = (22,  27,  34)
PANEL_BORDER = (33,  38,  45)
C_GATEWAY    = (240, 136,  62)
C_BACKBONE   = ( 88, 166, 255)
C_RELAY      = (188, 140, 255)
C_VILLAGE    = ( 63, 185,  80)
C_IOT        = (118, 227, 234)
C_LINK_OK    = ( 50, 120, 200)
C_LINK_FAIL  = (180,  40,  40)
C_LINK_WEAK  = (200, 160,  40)
C_PACKET     = (255, 220,  80)
C_PACKET_EMG = (255,  80,  80)
C_PACKET_DTN = (180, 100, 255)
C_TEXT       = (230, 237, 243)
C_MUTED      = (139, 148, 158)
C_GREEN      = ( 63, 185,  80)
C_RED        = (255,  68,  68)
C_YELLOW     = (227, 179,  65)
C_INPUT_BG   = ( 30,  36,  44)
C_INPUT_BD   = ( 88, 166, 255)
C_SEL        = (255, 220,  80)

# ═══════════════════════════════════════════════════════════════════════════════
# NODE & LINK DEFINITIONS  (เครือข่ายภูเขาสูง)
# ═══════════════════════════════════════════════════════════════════════════════
NET_W = W - PANEL_W   # canvas width for network

RAW_NODES = [
    # id, label,            rx,   ry,   type
    ("GW",    "Internet\nGateway",  0.50, 0.06, "gateway"),
    ("SA",    "Summit\nAlpha",      0.22, 0.22, "backbone"),
    ("SB",    "Summit\nBeta",       0.52, 0.20, "backbone"),
    ("SG",    "Summit\nGamma",      0.78, 0.24, "backbone"),
    ("RN",    "Relay\nNorth",       0.12, 0.44, "relay"),
    ("RC",    "Relay\nCenter",      0.46, 0.46, "relay"),
    ("RE",    "Relay\nEast",        0.82, 0.46, "relay"),
    ("VA",    "Village\nA",         0.06, 0.68, "village"),
    ("VB",    "Village\nB",         0.28, 0.72, "village"),
    ("VC",    "Village\nC",         0.54, 0.74, "village"),
    ("VD",    "Village\nD",         0.80, 0.70, "village"),
    ("S1",    "Sensor\nCluster 1",  0.16, 0.90, "iot"),
    ("S2",    "Sensor\nCluster 2",  0.66, 0.90, "iot"),
]

RAW_LINKS = [
    # src, dst, bw(Mbps), type
    ("GW","SA",  300, "fiber"),
    ("GW","SB",  200, "microwave"),
    ("SA","SB",  150, "microwave"),
    ("SB","SG",  120, "microwave"),
    ("SA","RN",   80, "wifi"),
    ("SB","RC",   90, "wifi"),
    ("SG","RE",   70, "wifi"),
    ("RN","VA",   30, "wifi"),
    ("RN","VB",   35, "wifi"),
    ("RC","VB",   25, "wifi"),
    ("RC","VC",   40, "wifi"),
    ("RE","VC",   30, "wifi"),
    ("RE","VD",   28, "wifi"),
    ("VA","S1",   10, "lora"),
    ("VD","S2",   10, "lora"),
    # backup
    ("SA","RC",   60, "backup"),
    ("SG","RC",   55, "backup"),
    ("VB","VC",   20, "backup"),
]

NODE_COLOR = {
    "gateway": C_GATEWAY, "backbone": C_BACKBONE,
    "relay": C_RELAY, "village": C_VILLAGE, "iot": C_IOT
}
NODE_RADIUS = {
    "gateway": 22, "backbone": 19, "relay": 15, "village": 13, "iot": 10
}
LINK_COLOR = {
    "fiber": C_GATEWAY, "microwave": C_BACKBONE,
    "wifi": C_VILLAGE, "lora": C_IOT, "backup": C_RELAY
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
class Node:
    def __init__(self, nid, label, rx, ry, ntype):
        self.id     = nid
        self.label  = label
        self.x      = int(rx * NET_W)
        self.y      = int(ry * (H - 80))  + 40
        self.type   = ntype
        self.color  = NODE_COLOR[ntype]
        self.radius = NODE_RADIUS[ntype]
        self.buffer = deque(maxlen=20)   # DTN buffer
        self.selected = False
        self.pulse    = 0.0   # animation

    def draw(self, surf, font_sm, font_xs, selected):
        # glow effect when selected
        if selected:
            glow_r = self.radius + 8
            glow_surf = pygame.Surface((glow_r*2+4, glow_r*2+4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*C_SEL, 60), (glow_r+2, glow_r+2), glow_r)
            surf.blit(glow_surf, (self.x - glow_r - 2, self.y - glow_r - 2))

        # pulse ring if has buffered packets (DTN)
        if self.buffer:
            pulse_r = self.radius + 4 + int(4 * abs(math.sin(self.pulse)))
            pygame.draw.circle(surf, C_PACKET_DTN, (self.x, self.y), pulse_r, 2)

        # shadow
        pygame.draw.circle(surf, (0,0,0), (self.x+2, self.y+2), self.radius)
        # fill
        pygame.draw.circle(surf, self.color, (self.x, self.y), self.radius)
        # border
        border_col = C_SEL if selected else (255,255,255)
        pygame.draw.circle(surf, border_col, (self.x, self.y), self.radius, 2)

        # label
        for i, line in enumerate(self.label.split("\n")):
            txt = font_xs.render(line, True, C_TEXT)
            surf.blit(txt, (self.x - txt.get_width()//2,
                            self.y + self.radius + 2 + i*11))

        # buffer badge
        if self.buffer:
            badge = font_xs.render(f"▣{len(self.buffer)}", True, C_PACKET_DTN)
            surf.blit(badge, (self.x - badge.get_width()//2, self.y - self.radius - 13))

    def hit(self, mx, my):
        return math.hypot(mx - self.x, my - self.y) <= self.radius + 6


class Link:
    def __init__(self, src, dst, bw, ltype):
        self.src   = src
        self.dst   = dst
        self.bw    = bw
        self.type  = ltype
        self.alive = True
        self.quality = 1.0       # 0.0–1.0
        self.flicker_timer = 0
        self.backup = (ltype == "backup")

    def color(self):
        if not self.alive:
            return C_LINK_FAIL
        if self.quality < 0.5:
            return C_LINK_WEAK
        return LINK_COLOR.get(self.type, C_LINK_OK)

    def width(self):
        if self.backup:
            return 1
        return max(1, int(self.bw / 60))

    def toggle(self):
        self.alive = not self.alive

    def tick(self, dt):
        # random quality fluctuation
        self.quality += random.uniform(-0.02, 0.02)
        self.quality  = max(0.1, min(1.0, self.quality))
        if self.flicker_timer > 0:
            self.flicker_timer -= dt
            if self.flicker_timer <= 0:
                self.alive = True

    def draw(self, surf):
        n1, n2 = self.src, self.dst
        col = self.color()
        w   = self.width()
        if self.backup:
            # dashed
            dx = n2.x - n1.x; dy = n2.y - n1.y
            dist = max(1, math.hypot(dx, dy))
            dash = 8; step = dash * 2
            for i in range(0, int(dist), step):
                t0 = i / dist; t1 = min((i+dash)/dist, 1.0)
                x0 = int(n1.x + dx*t0); y0 = int(n1.y + dy*t0)
                x1 = int(n1.x + dx*t1); y1 = int(n1.y + dy*t1)
                pygame.draw.line(surf, col, (x0,y0), (x1,y1), 1)
        else:
            pygame.draw.line(surf, col, (n1.x, n1.y), (n2.x, n2.y), w)


class Packet:
    """A visual packet travelling along a path of nodes."""
    RADIUS = 6

    def __init__(self, msg, path, ptype="normal"):
        self.msg      = msg
        self.path     = path       # list of Node
        self.ptype    = ptype      # "normal" | "emergency" | "dtn"
        self.seg      = 0          # current segment index
        self.progress = 0.0        # 0.0 → 1.0 along current segment
        self.speed    = 0.004 if ptype == "dtn" else 0.006
        self.done     = False
        self.dead     = False      # dropped
        self.trail    = deque(maxlen=12)

    @property
    def color(self):
        return {
            "normal":    C_PACKET,
            "emergency": C_PACKET_EMG,
            "dtn":       C_PACKET_DTN,
        }[self.ptype]

    @property
    def pos(self):
        if self.seg >= len(self.path) - 1:
            return (self.path[-1].x, self.path[-1].y)
        n1 = self.path[self.seg]
        n2 = self.path[self.seg + 1]
        x  = n1.x + (n2.x - n1.x) * self.progress
        y  = n1.y + (n2.y - n1.y) * self.progress
        return (int(x), int(y))

    def tick(self, link_map):
        if self.done or self.dead:
            return
        if self.seg >= len(self.path) - 1:
            self.done = True
            return

        n1 = self.path[self.seg]
        n2 = self.path[self.seg + 1]
        key = (n1.id, n2.id)
        lnk = link_map.get(key) or link_map.get((n2.id, n1.id))

        if lnk and not lnk.alive:
            if self.ptype == "dtn":
                # store at current node, wait
                n2.buffer.append(self)
                self.dead = True
            else:
                self.dead = True   # dropped
            return

        self.trail.append(self.pos)
        self.progress += self.speed * (lnk.quality if lnk else 1.0)
        if self.progress >= 1.0:
            self.progress = 0.0
            self.seg += 1

    def draw(self, surf):
        if self.done or self.dead:
            return
        # trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(180 * i / max(1, len(self.trail)))
            r = max(1, self.RADIUS - 3)
            s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (r, r), r)
            surf.blit(s, (tx - r, ty - r))
        # body
        x, y = self.pos
        pygame.draw.circle(surf, (0,0,0), (x+1,y+1), self.RADIUS)
        pygame.draw.circle(surf, self.color, (x, y), self.RADIUS)
        pygame.draw.circle(surf, (255,255,255), (x, y), self.RADIUS, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
class MountainNetSim:
    def __init__(self):
        self.screen  = pygame.display.set_mode((W, H))
        pygame.display.set_caption("🏔️  เครือข่ายภูเขาสูง — Interactive Simulator")
        self.clock   = pygame.time.Clock()

        # fonts
        self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_md = pygame.font.SysFont("consolas", 13)
        self.font_sm = pygame.font.SysFont("consolas", 11)
        self.font_xs = pygame.font.SysFont("consolas", 10)

        self._build_network()

        self.packets   = []
        self.log       = deque(maxlen=18)
        self.selected  = None       # selected destination node id
        self.input_txt = ""
        self.input_active = True
        self.sim_time  = 0.0
        self.auto_chaos = False     # random link failure mode
        self.stats     = {"sent":0, "delivered":0, "dropped":0, "dtn_stored":0}

    def _build_network(self):
        self.nodes = {nid: Node(nid, lbl, rx, ry, nt)
                      for nid, lbl, rx, ry, nt in RAW_NODES}
        self.links = []
        self.link_map = {}
        for src_id, dst_id, bw, lt in RAW_LINKS:
            lnk = Link(self.nodes[src_id], self.nodes[dst_id], bw, lt)
            self.links.append(lnk)
            self.link_map[(src_id, dst_id)] = lnk

    def _find_path(self, src_id, dst_id):
        """BFS respecting only alive links (with DTN fallback)."""
        from collections import deque as dq
        visited = {src_id}
        queue   = dq([[src_id]])
        while queue:
            path = queue.popleft()
            cur  = path[-1]
            if cur == dst_id:
                return [self.nodes[n] for n in path]
            for (a, b), lnk in self.link_map.items():
                nxt = None
                if a == cur and b not in visited:
                    nxt = b
                elif b == cur and a not in visited:
                    nxt = a
                if nxt:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None  # no path

    def _find_path_any(self, src_id, dst_id):
        """BFS ignoring link state (for DTN routing)."""
        from collections import deque as dq
        visited = {src_id}
        queue   = dq([[src_id]])
        all_adj = {}
        for (a, b) in self.link_map:
            all_adj.setdefault(a, []).append(b)
            all_adj.setdefault(b, []).append(a)
        while queue:
            path = queue.popleft()
            cur  = path[-1]
            if cur == dst_id:
                return [self.nodes[n] for n in path]
            for nxt in all_adj.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def send_message(self, msg, dst_id, ptype="normal"):
        src_id = "GW"
        path = self._find_path(src_id, dst_id)
        if path:
            pkt = Packet(msg, path, ptype)
            self.packets.append(pkt)
            self.stats["sent"] += 1
            self._log(f"📤 [{ptype.upper()}] '{msg[:20]}' → {dst_id}", C_PACKET)
        else:
            # DTN: route through dead links, packet will buffer
            path = self._find_path_any(src_id, dst_id)
            if path:
                pkt = Packet(msg, path, "dtn")
                self.packets.append(pkt)
                self.stats["sent"] += 1
                self._log(f"🔀 [DTN] '{msg[:20]}' → {dst_id} (buffering)", C_PACKET_DTN)
            else:
                self._log(f"❌ not found the path to {dst_id}", C_RED)

    def _log(self, msg, color=None):
        self.log.append((msg, color or C_MUTED, time.time()))

    def _toggle_link_at(self, node):
        """Toggle all links connected to this node."""
        nid = node.id
        for (a, b), lnk in self.link_map.items():
            if a == nid or b == nid:
                lnk.toggle()
                state = "✅ normal" if lnk.alive else "❌ lost"
                self._log(f"🔌 {a}↔{b}: {state}", C_YELLOW if lnk.alive else C_RED)

    def _random_chaos(self):
        for lnk in self.links:
            if not lnk.backup and random.random() < 0.25:
                lnk.alive = False
                lnk.flicker_timer = random.uniform(3.0, 8.0)
        self._log("⚡ random lost! DTN will help forwarding", C_RED)

    def _reset_links(self):
        for lnk in self.links:
            lnk.alive = True
            lnk.quality = 1.0
        self._log("🔄 reset all links", C_GREEN)

    # ──────────────────────────────────────────────────────────────────────────
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                elif ev.key == pygame.K_RETURN:
                    if self.input_txt.strip() and self.selected:
                        msg  = self.input_txt.strip()
                        ptype = "emergency" if msg.startswith("!") else "normal"
                        self.send_message(msg, self.selected, ptype)
                        self.input_txt = ""
                    elif not self.selected:
                        self._log("⚠️  click a destination node first", C_YELLOW)

                elif ev.key == pygame.K_BACKSPACE:
                    self.input_txt = self.input_txt[:-1]

                elif ev.key == pygame.K_s:
                    self._random_chaos()

                elif ev.key == pygame.K_r:
                    self._reset_links()

                elif ev.key == pygame.K_c:
                    self.log.clear()

                elif ev.key == pygame.K_1:
                    self.send_message("Hello there!", "VA", "normal")
                elif ev.key == pygame.K_2:
                    self.send_message("!forest fire emergency!", "VD", "emergency")
                elif ev.key == pygame.K_3:
                    self.send_message("sensor_data_batch", "S1", "normal")

                else:
                    if len(self.input_txt) < 40:
                        self.input_txt += ev.unicode

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if mx > NET_W:   # panel click — ignore
                    continue
                for node in self.nodes.values():
                    if node.hit(mx, my):
                        if ev.button == 1:   # left click → select
                            self.selected = node.id
                            self._log(f"🎯 selected destination: {node.id} ({node.type})", C_SEL)
                        elif ev.button == 3:  # right click → toggle link
                            self._toggle_link_at(node)
                        break

    # ──────────────────────────────────────────────────────────────────────────
    def update(self, dt):
        self.sim_time += dt

        # tick links
        for lnk in self.links:
            lnk.tick(dt)

        # tick packets
        for pkt in self.packets:
            pkt.tick(self.link_map)
            if pkt.done:
                self.stats["delivered"] += 1
                self._log(f"✅ sent to {pkt.path[-1].id}: '{pkt.msg[:24]}'", C_GREEN)
                pkt.done = False   # mark consumed
                pkt.dead = True
            elif pkt.dead and pkt.ptype == "dtn":
                self.stats["dtn_stored"] += 1
            elif pkt.dead:
                self.stats["dropped"] += 1

        self.packets = [p for p in self.packets if not p.dead]

        # pulse animation
        for node in self.nodes.values():
            node.pulse += 0.08

        # try to flush DTN buffers
        for node in self.nodes.values():
            if node.buffer:
                # check if any outgoing link is alive
                for (a, b), lnk in self.link_map.items():
                    if (a == node.id or b == node.id) and lnk.alive:
                        pkt = node.buffer.popleft()
                        pkt.dead = False
                        self.packets.append(pkt)
                        self._log(f"🔀 DTN flush @ {node.id}", C_PACKET_DTN)
                        break

    # ──────────────────────────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(BG)

        # ── network canvas ────────────────────────────────────────────────────
        # draw links
        for lnk in self.links:
            lnk.draw(self.screen)

        # draw nodes
        sel_id = self.selected
        for node in self.nodes.values():
            node.draw(self.screen, self.font_sm, self.font_xs, node.id == sel_id)

        # draw packets
        for pkt in self.packets:
            pkt.draw(self.screen)

        # divider
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W, 0), (NET_W, H), 2)

        # ── right panel ───────────────────────────────────────────────────────
        self._draw_panel()

        # ── top bar ───────────────────────────────────────────────────────────
        self._draw_topbar()

        pygame.display.flip()

    def _draw_topbar(self):
        bar_h = 32
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, NET_W, bar_h))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, bar_h), (NET_W, bar_h))
        title = self.font_lg.render("🏔  High Mountain Network — Interactive Simulator", True, C_TEXT)
        self.screen.blit(title, (10, 7))

    def _draw_panel(self):
        px = NET_W + 8
        pygame.draw.rect(self.screen, PANEL_BG, (NET_W, 0, PANEL_W, H))

        y = 10
        # ── Title ──
        t = self.font_lg.render("📡  Control Panel", True, C_BACKBONE)
        self.screen.blit(t, (px, y)); y += 22
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W+4, y), (W-4, y)); y += 8

        # ── Stats ──
        t = self.font_md.render("Statistics", True, C_MUTED)
        self.screen.blit(t, (px, y)); y += 16
        stats = [
            (f"Sent     : {self.stats['sent']}",      C_TEXT),
            (f"Delivered: {self.stats['delivered']}",  C_GREEN),
            (f"Dropped  : {self.stats['dropped']}",    C_RED),
            (f"DTN buf  : {self.stats['dtn_stored']}", C_PACKET_DTN),
        ]
        for s, c in stats:
            self.screen.blit(self.font_sm.render(s, True, c), (px, y)); y += 14
        y += 4
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W+4, y), (W-4, y)); y += 8

        # ── Destination selector ──
        t = self.font_md.render("Destination Node", True, C_MUTED)
        self.screen.blit(t, (px, y)); y += 16
        sel_lbl = self.selected or "(click a node on the map)"
        col = C_SEL if self.selected else C_MUTED
        self.screen.blit(self.font_sm.render(f"▶ {sel_lbl}", True, col), (px, y)); y += 18
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W+4, y), (W-4, y)); y += 8

        # ── Input box ──
        t = self.font_md.render("Message  (Enter=send, !=emergency)", True, C_MUTED)
        self.screen.blit(t, (px, y)); y += 16
        box_rect = pygame.Rect(NET_W+4, y, PANEL_W-12, 26)
        pygame.draw.rect(self.screen, C_INPUT_BG, box_rect, border_radius=4)
        pygame.draw.rect(self.screen, C_INPUT_BD, box_rect, 1, border_radius=4)
        cursor = "█" if int(time.time()*2) % 2 == 0 else " "
        txt = self.font_md.render(self.input_txt + cursor, True, C_TEXT)
        self.screen.blit(txt, (box_rect.x+5, box_rect.y+5))
        y += 32
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W+4, y), (W-4, y)); y += 8

        # ── Quick send buttons hint ──
        hints = [
            ("[1] send greeting message → Village A",   C_MUTED),
            ("[2] !emergency forest fire  → Village D",   C_PACKET_EMG),
            ("[3] sensor batch    → Sensor 1",     C_IOT),
            ("[S] random link loss (chaos)",           C_RED),
            ("[R] reset all links",                   C_GREEN),
            ("[C] clear log",                         C_MUTED),
            ("click right on node = toggle signal",         C_YELLOW),
        ]
        for h, c in hints:
            self.screen.blit(self.font_xs.render(h, True, c), (px, y)); y += 13
        y += 4
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W+4, y), (W-4, y)); y += 8

        # ── Legend ──
        t = self.font_md.render("Legend", True, C_MUTED)
        self.screen.blit(t, (px, y)); y += 15
        legend = [
            ("● Gateway",    C_GATEWAY),
            ("● Backbone",   C_BACKBONE),
            ("● Relay",      C_RELAY),
            ("● Village",    C_VILLAGE),
            ("● IoT/Sensor", C_IOT),
            ("— Fiber",      C_GATEWAY),
            ("— Microwave",  C_BACKBONE),
            ("— Wi-Fi",      C_VILLAGE),
            ("-- Backup",    C_RELAY),
            ("● Packet",     C_PACKET),
            ("● Emergency",  C_PACKET_EMG),
            ("● DTN buf",    C_PACKET_DTN),
        ]
        col2 = False
        for i, (lbl, c) in enumerate(legend):
            cx = px + (PANEL_W//2 - 10 if i % 2 else 0)
            self.screen.blit(self.font_xs.render(lbl, True, c), (px + (160 if i%2 else 0), y))
            if i % 2:
                y += 13
        if len(legend) % 2:
            y += 13
        y += 4
        pygame.draw.line(self.screen, PANEL_BORDER, (NET_W+4, y), (W-4, y)); y += 8

        # ── Log ──
        t = self.font_md.render("Event Log", True, C_MUTED)
        self.screen.blit(t, (px, y)); y += 15
        log_area_h = H - y - 10
        for msg, col, ts in list(self.log)[-int(log_area_h//13):]:
            elapsed = time.time() - ts
            alpha_f = max(0.3, 1.0 - elapsed / 30.0)
            r,g,b   = col
            fade    = (int(r*alpha_f), int(g*alpha_f), int(b*alpha_f))
            txt = self.font_xs.render(msg[:42], True, fade)
            self.screen.blit(txt, (px, y)); y += 13
            if y > H - 15:
                break

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        self._log("🏔  WELLCOME TO HIGH MOUNTAIN NETWORK!", C_BACKBONE)
        self._log("click a node and type message, then press Enter to send", C_MUTED)
        self._log("right-click a node = toggle signal", C_YELLOW)
        self._log("press [S] for random link loss, [R] to reset all links", C_MUTED)

        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sim = MountainNetSim()
    sim.run()
