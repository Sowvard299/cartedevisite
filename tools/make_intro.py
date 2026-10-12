#!/usr/bin/env python3
"""Génère l'animation de la carte (128x64, 1 bit, pixel natif).

Le personnage est un petit pantin 3D (capsules et sphères) rendu en
sur-échantillonnage, éclairé, puis tramé (Bayer 8x8) avec contour lumineux
et traits intérieurs. Cela permet une vraie marche, un demi-tour fluide et
un bras tendu en perspective.

Scénario :
  allumage façon tube cathodique -> arrivée en marchant depuis la droite
  (poussière aux pas) -> demi-tour -> clin d'œil -> tend la carte (éclat) ->
  la carte s'envole en tournant et remplit l'écran (le personnage se dissout)
  -> flash + lignes de balayage -> les modules du QR tombent en place ->
  balayage de lecture -> textes -> QR fixe QR_HOLD_MS.

Sorties :
  preview/intro.gif                      aperçu x4, rendu "pixels d'OLED", vraies durées
  preview/intro_sheet.png                planche de toutes les images
  preview/qr_128x64.png                  image finale (QR), taille réelle
  firmware/src/assets/intro_frames.h     images au format buffer u8g2 (SSD1306)

Usage : python tools/make_intro.py
"""
from math import cos, pi, sin
from pathlib import Path

import numpy as np
import segno
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.linkedin.com/in/nathan-le-fol-ab7b3b230"
QR_HOLD_MS = 10_000
W, H = 128, 64
FPS_MS = 40             # 25 images/s ; un envoi I2C 400 kHz prend ~23 ms
SS = 4                  # sur-échantillonnage du rendu du personnage
FLOOR_Y = 61            # ligne de contact des pieds (écran)

BAYER8 = (np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]]) + 0.5) / 64


def bayer(ox=0, oy=0):
    yy, xx = np.indices((H, W))
    return BAYER8[(yy - oy) % 8, (xx - ox) % 8]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def v3(x, y, z):
    return np.array([x, y, z], dtype=float)


def unit(v):
    return v / np.linalg.norm(v)


# ================================================================ personnage
# Repère local : x = côté droit du personnage, y = vers le haut, z = vers l'avant.
# yaw = 0 : face à l'écran ; yaw = -90° : de profil, tourné vers la gauche.
SKIN, HAIR, JACKET, PANTS, SHOES = 1.5, 0.3, 0.9, 0.55, 0.22
LIGHT = unit(v3(-0.55, -0.6, 0.6))      # repère écran (y vers le bas), lumière haut-gauche-avant
G_BODY, G_HEAD, G_HAIR, G_LLEG, G_RLEG, G_LARM, G_RARM = range(7)
THIGH, SHIN, UPPER, FORE = 9.5, 9.5, 8.0, 6.5
HEAD_R = 7.0
HIP_Y = 22.2       # hauteur des hanches debout


def jacket_shader(nl, t):
    """Veste avec col en V (chemise claire) et cravate sombre sur le devant."""
    tone = np.full(t.shape, JACKET)
    front = nl[2] > 0.55
    shirt = front & (np.abs(nl[0]) < np.clip((t - 0.45) * 0.9, 0, None))
    tone[shirt] = 1.0
    tie = front & (np.abs(nl[0]) < 0.09) & (t > 0.35)
    tone[tie] = 0.0
    return tone


def flat(tone):
    return lambda nl, t: np.full(t.shape, tone)


def build_pose(walk=0.0, amp=0.0, reach=0.0):
    """Liste des parties (capsules) en coordonnées locales + infos tête/main."""
    parts = []

    def cap(a, b, r, shader, grp, rdep=None):
        parts.append((a, b, r, r if rdep is None else rdep, shader, grp))

    # jambes, cuisse et tibia en FK dans le plan y-z
    legs = {}
    for sgn, grp in ((-1, G_LLEG), (1, G_RLEG)):
        q = walk + (0 if sgn < 0 else pi)
        a = amp * 0.5 * sin(q)
        k = amp * (0.12 + 0.95 * max(0.0, cos(q)) ** 1.5)
        hip = v3(sgn * 3.0, 0, 0)
        knee = hip + THIGH * v3(0, -cos(a), sin(a))
        ankle = knee + SHIN * v3(0, -cos(a - k), sin(a - k))
        legs[sgn] = (hip, knee, ankle, grp, q)
    # le pied le plus bas touche le sol (rebond naturel du corps)
    hip_y = 0.3 + 3.1 - min(legs[s][2][1] for s in legs)
    lift = v3(0, hip_y, 0)
    leg_info = {}
    for sgn, (hip, knee, ankle, grp, q) in legs.items():
        hip, knee, ankle = hip + lift, knee + lift, ankle + lift
        cap(hip, knee, 2.7, flat(PANTS), grp)
        cap(knee, ankle, 2.3, flat(PANTS), grp)
        heel, toe = ankle + v3(0, -1.6, -1.2), ankle + v3(0, -1.6, 3.6)
        cap(heel, toe, 1.5, flat(SHOES), grp)
        leg_info[sgn] = (toe, q)

    dy = hip_y - HIP_Y
    up = lambda x, y, z: v3(x, y + dy, z)
    cap(up(-2.8, 22.8, 0), up(2.8, 22.8, 0), 3.4, flat(PANTS), G_BODY)       # bassin
    cap(up(0, 26, 0), up(0, 33.5, 0), 7.0, jacket_shader, G_BODY, 4.2)       # torse
    cap(up(0, 38, 0), up(0, 41, 0), 2.0, flat(SKIN), G_BODY)                 # cou

    head = up(0, 47.6, 0.5)
    cap(head, head, HEAD_R, flat(SKIN), G_HEAD)
    cap(head + v3(0, -0.4, HEAD_R + 0.1), head + v3(0, -0.4, HEAD_R + 0.1), 0.9, flat(SKIN), G_HEAD)  # nez
    for s in (-1, 1):
        cap(head + v3(s * HEAD_R, -0.4, -0.3), head + v3(s * HEAD_R, -0.4, -0.3), 1.1, flat(SKIN), G_HEAD)
    hair_c = head + v3(0, 2.0, -2.6)
    cap(hair_c, hair_c, 7.6, flat(HAIR), G_HAIR)

    # bras : balancement opposé aux jambes ; bras droit (sgn=+1) tend la carte
    hand_r = None
    for sgn, grp in ((-1, G_LARM), (1, G_RARM)):
        q = walk + (0 if sgn < 0 else pi)
        pitch = -amp * 0.55 * sin(q)
        side, bend = sgn * 0.14, 0.3 + 0.4 * max(0.0, pitch)
        if sgn == 1 and reach > 0:
            r = ease(reach)
            pitch, side, bend = lerp(pitch, 1.3, r), lerp(side, -0.42, r), lerp(bend, 0.12, r)
        sh = up(sgn * 7.6, 36.2, 0)
        elbow = sh + UPPER * unit(v3(side, -cos(pitch), sin(pitch)))
        fp = pitch + bend
        wrist = elbow + FORE * unit(v3(side * 0.5, -cos(fp), sin(fp)))
        hand = wrist + 1.2 * unit(wrist - elbow)
        cap(sh, elbow, 2.1, flat(JACKET), grp)
        cap(elbow, wrist, 1.8, flat(JACKET), grp)
        cap(hand, hand, 1.9, flat(SKIN), grp)
        if sgn == 1:
            hand_r = hand
    return parts, head, hand_r, leg_info


class Figure:
    """Position et orientation à l'écran ; projette le repère local."""

    def __init__(self, cx, yaw):
        self.cx, self.yaw = cx, yaw
        self.c, self.s = cos(yaw), sin(yaw)

    def project(self, p):
        x, y, z = p
        return self.cx + x * self.c + z * self.s, FLOOR_Y - y, -x * self.s + z * self.c

    def facing(self, d):
        """Composante vers la caméra d'une direction locale."""
        return -d[0] * self.s + d[2] * self.c


def render_figure(fig, parts):
    """Rendu sur-échantillonné -> (ton, masque, groupe, profondeur) en pixels écran."""
    hs, ws = H * SS, W * SS
    tone = np.zeros((hs, ws))
    depth = np.full((hs, ws), -1e9)
    grp = np.full((hs, ws), -1)
    for a, b, rl, rd, shader, g in parts:
        ax, ay, az = fig.project(a)
        bx, by, bz = fig.project(b)
        r = np.hypot(rl * fig.c, rd * fig.s) if rl != rd else rl
        x0 = int(max((min(ax, bx) - r - 1) * SS, 0)); x1 = int(min((max(ax, bx) + r + 1) * SS, ws))
        y0 = int(max((min(ay, by) - r - 1) * SS, 0)); y1 = int(min((max(ay, by) + r + 1) * SS, hs))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        px, py = (xx + 0.5) / SS, (yy + 0.5) / SS
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) if L2 > 1e-9 else np.zeros_like(px)
        qx, qy = px - (ax + t * dx), py - (ay + t * dy)
        d2 = qx * qx + qy * qy
        inside = d2 < r * r
        h = np.sqrt(np.clip(1 - d2 / (r * r), 0, 1))
        z = az + t * (bz - az) + r * h
        win = inside & (z > depth[y0:y1, x0:x1])
        if not win.any():
            continue
        nx, ny, nz = qx / r, qy / r, h
        nl = np.stack([nx * fig.c - nz * fig.s, -ny, nx * fig.s + nz * fig.c])  # normale locale
        base = shader(nl, t)
        lit = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
        shade = base * (0.45 + 0.75 * lit)
        shade = np.select([shade < 0.12, shade < 0.32, shade < 0.62], [0.0, 0.25, 0.5], 1.0)  # aplats
        tone[y0:y1, x0:x1][win] = shade[win]
        depth[y0:y1, x0:x1][win] = z[win]
        grp[y0:y1, x0:x1][win] = g

    # réduction à la résolution écran
    cov = (grp >= 0).reshape(H, SS, W, SS).mean(axis=(1, 3))
    tsum = np.where(grp >= 0, tone, 0).reshape(H, SS, W, SS).sum(axis=(1, 3))
    cnt = (grp >= 0).reshape(H, SS, W, SS).sum(axis=(1, 3))
    tone_n = np.where(cnt > 0, tsum / np.maximum(cnt, 1), 0)
    blk_d = depth.reshape(H, SS, W, SS).transpose(0, 2, 1, 3).reshape(H, W, SS * SS)
    blk_g = grp.reshape(H, SS, W, SS).transpose(0, 2, 1, 3).reshape(H, W, SS * SS)
    idx = blk_d.argmax(axis=2)
    depth_n = np.take_along_axis(blk_d, idx[..., None], 2)[..., 0]
    grp_n = np.take_along_axis(blk_g, idx[..., None], 2)[..., 0]
    return tone_n, cov >= 0.45, grp_n, depth_n


def shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    ys = slice(max(dy, 0), H + min(dy, 0)); yd = slice(max(-dy, 0), H + min(-dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0)); xd = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def ink_figure(fig, parts, head, eyes="open", smile=False):
    """Tramage + contour lumineux + traits intérieurs + traits du visage."""
    tone, mask, grp, depth = render_figure(fig, parts)
    lit = mask & (tone > bayer(int(fig.cx)))
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        m2, g2, d2 = shift(mask, dy, dx, False), shift(grp, dy, dx, -1), shift(depth, dy, dx, -1e9)
        # trait sombre là où une partie plus proche (autre groupe) passe devant
        lit &= ~(mask & m2 & (g2 != grp) & (d2 > depth + 0.8))
    rim = mask & ~(shift(mask, 0, 1, False) & shift(mask, 0, -1, False)
                   & shift(mask, 1, 0, False) & shift(mask, -1, 0, False))
    lit |= rim

    def mark(p, v):
        x, y, _ = fig.project(p)
        x, y = int(np.floor(x)), int(np.floor(y))
        if 0 <= x < W and 0 <= y < H:
            lit[y, x] = v

    for s in (-1, 1):
        d = unit(v3(s * 0.4, 0.1, 0.91))
        if fig.facing(d) > 0.12:
            e = head + HEAD_R * d
            if eyes == "open":
                mark(e, False)
                mark(e + v3(0, 1.0, 0), False)
            else:                                       # yeux plissés ^ ^
                mark(e + v3(0, 0.9, 0), False)
                mark(e + v3(-0.9, 0, 0), False)
                mark(e + v3(0.9, 0, 0), False)
    for x in ((-0.22, 0, 0.22) if smile else (-0.1, 0.1)):
        d = unit(v3(x, -0.42 + (0.07 if smile and x else 0), 0.9))
        if fig.facing(d) > -0.05:
            mark(head + HEAD_R * d, False)
    return lit, mask


# ================================================================ décor
def floor(buf, shadow_x=None):
    """Sol en perspective : rangées de points de plus en plus serrées."""
    for y, step, off in ((58, 8, 0), (60, 5, 2), (62, 3, 1), (63, 2, 0)):
        buf[y, off::step] = True
    if shadow_x is not None:
        yy, xx = np.indices((H, W))
        sh = ((xx - shadow_x) / 12.0) ** 2 + ((yy - 61) / 2.6) ** 2 < 1
        buf[sh] = False


def compose(fig, parts, head, eyes="open", smile=False, fade=0.0):
    buf = np.zeros((H, W), dtype=bool)
    floor(buf, fig.cx)
    lit, mask = ink_figure(fig, parts, head, eyes, smile)
    if fade > 0:                       # dissolution tramée du personnage
        keep = bayer() >= fade
        lit &= keep
        mask &= keep
    buf[mask] = lit[mask]
    return buf


def sparkle(buf, x, y, size, v=True):
    """Étoile à 4 branches (+ diagonales si grande)."""
    for d in range(-size, size + 1):
        for px, py in ((x + d, y), (x, y + d)):
            if 0 <= px < W and 0 <= py < H:
                buf[py, px] = v
    if size >= 3:
        for d in (-1, 1):
            for e in (-1, 1):
                if 0 <= x + d < W and 0 <= y + e < H:
                    buf[y + e, x + d] = v


# ================================================================ carte volante
def draw_card(buf, cx, cy, w, h, angle):
    """Carte en perspective, tournant autour de l'axe vertical."""
    img = Image.fromarray(buf.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    F = 2.4 * max(w, 8)

    def P(u, v):
        X, Z = u * w / 2 * cos(angle), u * w / 2 * sin(angle)
        k = F / (F - Z)
        return (cx + X * k, cy + v * h / 2 * k)

    quad = [P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)]
    front = cos(angle) > 0
    lw = max(1, int(w / 45))
    if front:
        d.polygon(quad, fill=255, outline=0)
        d.line([P(-0.95, -0.95), P(0.95, -0.95), P(0.95, 0.95), P(-0.95, 0.95), P(-0.95, -0.95)],
               fill=0, width=lw) if w > 30 else None
        if abs(cos(angle)) > 0.3:
            d.line([P(-0.75, -0.35), P(0.05, -0.35)], fill=0, width=lw)       # nom
            d.line([P(-0.75, 0.15), P(-0.2, 0.15)], fill=0, width=lw)         # titre
            d.line([P(-0.75, 0.5), P(-0.35, 0.5)], fill=0, width=lw)
            d.rectangle([P(0.3, -0.55), P(0.75, 0.45)], outline=0, width=lw) if w > 14 else None
    else:
        d.polygon(quad, fill=0, outline=255)
    out = np.array(img) > 127
    if not front:                      # dos de carte : trame claire
        poly = Image.new("1", (W, H), 0)
        ImageDraw.Draw(poly).polygon(quad, fill=1)
        inside = np.array(poly)
        out[inside & (bayer() < 0.25)] = True
    return out


# ================================================================ QR
FONT = {
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "i": ["1", "0", "1", "1", "1", "1", "1"],
    "n": ["00000", "00000", "11110", "10001", "10001", "10001", "10001"],
}


def text(buf, x, y, s, v):
    for ch in s:
        g = FONT[ch]
        for r, row in enumerate(g):
            for c, bit in enumerate(row):
                if bit == "1" and 0 <= y + r < H and 0 <= x + c < W:
                    buf[y + r, x + c] = v
        x += len(g[0]) + 1


def qr_matrix():
    """Règle du §5.1 : version <= 3, ECC M sinon L, sinon sans www."""
    for url in (URL, URL.replace("://www.", "://")):
        for ecc in ("m", "l"):
            q = segno.make(url, error=ecc, boost_error=False, micro=False)
            if q.version <= 3:
                print(f"QR : version {q.version}, ECC {q.error}, {len(url)} car. -> {url}")
                return np.array([[bool(v) for v in row] for row in q.matrix])
    raise SystemExit("URL trop longue pour un QR v3 : raccourcir l'URL LinkedIn")


QR = qr_matrix()
QR_N = QR.shape[0]
QR_M = 2
QR_X = (W - QR_N * QR_M) // 2
QR_Y = (H - QR_N * QR_M) // 2
LOGO_X, LOGO_Y = QR_X + QR_N * QR_M + 10, 24


def module(buf, r, c, y=None):
    y = QR_Y + r * QR_M if y is None else int(round(y))
    x = QR_X + c * QR_M
    y0, y1 = max(y, 0), min(y + QR_M, H)
    if y0 < y1:
        buf[y0:y1, x:x + QR_M] = False


def logo(buf):
    bx, by = LOGO_X, LOGO_Y
    buf[by:by + 15, bx:bx + 15] = False
    buf[by, bx] = buf[by, bx + 14] = buf[by + 14, bx] = buf[by + 14, bx + 14] = True
    text(buf, bx + 3, by + 4, "in", True)


def qr_final():
    buf = np.ones((H, W), dtype=bool)  # fond allumé, modules sombres = éteints
    for r, c in zip(*np.nonzero(QR)):
        module(buf, r, c)
    text(buf, 6, 21, "SCAN", False)
    text(buf, 12, 33, "ME", False)
    logo(buf)
    return buf


def in_finder(r, c):
    n = QR_N
    return (r < 8 and c < 8) or (r < 8 and c >= n - 8) or (r >= n - 8 and c < 8)


# ================================================================ scénario
def build():
    frames = []

    def add(f, ms=FPS_MS):
        frames.append((f.copy(), ms))

    black = np.zeros((H, W), dtype=bool)
    stage = black.copy()
    floor(stage)

    # 0. allumage façon tube cathodique : un point, une ligne, puis l'image s'ouvre
    add(black, 250)
    for wdt in (2, 16, 60, 128):
        f = black.copy()
        f[32, 64 - wdt // 2:64 + wdt // 2] = True
        add(f, 40)
    for k, hh in enumerate((6, 16, 32)):
        f = black.copy()
        f[32 - hh:32 + hh, :] = stage[32 - hh:32 + hh, :]
        f[32 - hh, :] = f[32 + hh - 1, :] = True
        if k == 0:
            f[1::2, :] = False                      # lignes de balayage
        add(f, 40)
    add(stage, 250)

    # 1. marche depuis la droite (vitesse calée sur la foulée), poussière aux pas
    walk_frames, cycle = 36, 20
    x_start, x_end = W + 12, W // 2
    dust = []
    last = {}
    for i in range(walk_frames + 8):
        if i < walk_frames:
            p, amp = 2 * pi * i / cycle, 1.0
            x = lerp(x_start, x_end + 6, i / walk_frames)
        else:                                       # pas d'arrêt
            j = (i - walk_frames + 1) / 8
            p, amp = 2 * pi * (walk_frames + (i - walk_frames) * 0.5) / cycle, 1 - ease(j)
            x = lerp(x_end + 6, x_end, ease(j))
        fig = Figure(x, -pi / 2)
        parts, head, _, legs = build_pose(p, amp)
        f = compose(fig, parts, head)
        for sgn, (toe, q) in legs.items():          # contact du talon -> nuage de poussière
            ph = (q - pi / 2) % (2 * pi)
            if sgn in last and ph < last[sgn] and amp > 0.5:
                tx, _, _ = fig.project(toe)
                dust += [[tx - 3, 60.0, -0.5, -0.35, 0], [tx + 1, 60.0, 0.6, -0.45, 0], [tx - 1, 59.0, 0.1, -0.6, 0]]
            last[sgn] = ph
        for d in dust:
            d[0] += d[2]; d[1] += d[3]; d[4] += 1
            if d[4] < 7 and (d[4] < 3 or d[4] % 2 == 0) and 0 <= int(d[0]) < W:
                f[int(d[1]), int(d[0])] = True
        dust = [d for d in dust if d[4] < 7]
        add(f)

    # 2. demi-tour vers l'écran
    for k in range(1, 11):
        fig = Figure(x_end, lerp(-pi / 2, 0, ease(k / 10)))
        parts, head, _, _ = build_pose()
        add(compose(fig, parts, head))
    fig = Figure(x_end, 0.0)
    parts, head, _, _ = build_pose()
    add(compose(fig, parts, head), 350)
    add(compose(fig, parts, head, eyes="happy", smile=True), 160)
    add(compose(fig, parts, head, smile=True), 250)

    # 3. tend la carte
    card_w, card_h = 11.0, 7.0
    for k in range(1, 11):
        r = k / 10
        parts, head, hand, _ = build_pose(reach=r)
        f = compose(fig, parts, head, smile=True)
        if r > 0.55:
            s = ease((r - 0.55) / 0.45)
            hx, hy, _ = fig.project(hand)
            f = draw_card(f, hx - 3 * s, hy - 1, card_w * s, card_h * s, 0.0)
        add(f)
    parts, head, hand, _ = build_pose(reach=1.0)
    hx, hy, _ = fig.project(hand)
    ccx, ccy = hx - 3, hy - 1
    held = draw_card(compose(fig, parts, head, smile=True), ccx, ccy, card_w, card_h, 0.0)
    add(held, 200)
    # éclat sur le coin de la carte
    sx, sy = int(ccx - card_w / 2), int(ccy - card_h / 2)
    for size in (1, 2, 4, 2, 1):
        f = held.copy()
        sparkle(f, sx, sy, size)
        add(f, 50)
    add(held, 150)

    # 4. la carte s'envole en tournant et remplit l'écran ; le personnage se dissout
    n = 16
    for k in range(1, n + 1):
        t = k / n
        e = t * t
        cw = card_w * (134 / card_w) ** e
        chh = card_h * (70 / card_h) ** e
        f = compose(fig, parts, head, smile=True, fade=min(1.0, t * 1.6))
        f = draw_card(f, lerp(ccx, 64, ease(t)), lerp(ccy, 32, ease(t)), cw, chh, 2 * pi * ease(t))
        add(f)

    # 5. flash + lignes de balayage + petit décalage de lignes (glitch)
    white = np.ones((H, W), dtype=bool)
    add(white, 60)
    f = white.copy(); f[::2, :] = False
    add(f, 40)
    f = white.copy()
    for y0, dx in ((10, 6), (30, -4), (46, 3)):
        f[y0:y0 + 3, :] = np.roll(f[y0:y0 + 3, :], dx, axis=1)
        f[y0 + 1, :] = False
    add(f, 40)
    add(white, 80)

    # 6. les modules tombent en place : d'abord les trois repères, puis le reste
    rng = np.random.default_rng(3)
    mods = list(zip(*np.nonzero(QR)))
    start, fall = {}, 5
    for r, c in mods:
        start[(r, c)] = rng.integers(0, 3) if in_finder(r, c) else 4 + c * 0.45 + rng.random() * 5
    end = int(max(start.values())) + fall + 1
    for fr in range(end):
        f = white.copy()
        for (r, c), s0 in start.items():
            t = (fr - s0) / fall
            if t <= 0:
                continue
            ty = QR_Y + r * QR_M
            module(f, r, c, ty if t >= 1 else lerp(-QR_M - 2, ty, t * t))
        add(f, 35)

    # 7. balayage de lecture puis textes
    qr_only = white.copy()
    for r, c in mods:
        module(qr_only, r, c)
    for y in range(QR_Y - 2, QR_Y + QR_N * QR_M + 2, 7):
        f = qr_only.copy()
        f[y:y + 2, QR_X - 3:QR_X + QR_N * QR_M + 3] ^= True
        add(f, 30)
    seq = ["S", "SC", "SCA", "SCAN"]
    for s in seq:
        f = qr_only.copy(); text(f, 6, 21, s, False); add(f, 45)
    for s in ("M", "ME"):
        f = qr_only.copy(); text(f, 6, 21, "SCAN", False); text(f, 12, 33, s, False); add(f, 45)
    f = qr_only.copy(); text(f, 6, 21, "SCAN", False); text(f, 12, 33, "ME", False)
    g = f.copy(); sparkle(g, LOGO_X + 7, LOGO_Y + 7, 3, False); add(g, 60)

    final = qr_final()
    add(final, QR_HOLD_MS)
    return frames


# ================================================================ sorties
def to_u8g2(buf):
    """Buffer u8g2 plein SSD1306 : 8 pages x 128 colonnes, bit 0 = pixel du haut."""
    b = buf.reshape(H // 8, 8, W).astype(np.uint8)
    return bytes((b * (1 << np.arange(8))[None, :, None]).sum(axis=1).astype(np.uint8).ravel())


def oled_look(buf, scale=4):
    """Aperçu : chaque pixel = carré (scale-1)² dans une case scale², blanc légèrement froid."""
    cell = np.zeros((scale, scale), dtype=bool)
    cell[:scale - 1, :scale - 1] = True
    on = np.kron(buf, cell)
    img = np.empty(on.shape + (3,), dtype=np.uint8)
    img[:] = (6, 6, 8)
    img[on] = (235, 240, 255)
    pad = 6 * scale
    framed = np.full((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, 3), 20, dtype=np.uint8)
    framed[pad:-pad, pad:-pad] = img
    return Image.fromarray(framed)


def write_header(frames, path):
    uniq, index, seq = [], {}, []
    for f, ms in frames:
        key = to_u8g2(f)
        if key not in index:
            index[key] = len(uniq)
            uniq.append(key)
        seq.append((index[key], ms))
    lines = [
        "// Généré par tools/make_intro.py — ne pas modifier à la main.",
        f"// URL : {URL}",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define INTRO_UNIQUE_FRAMES {len(uniq)}",
        f"#define INTRO_STEPS {len(seq)}",
        f"#define QR_HOLD_MS {QR_HOLD_MS}",
        "",
        f"static const uint8_t INTRO_FRAMES[{len(uniq)}][1024] = {{",
    ]
    for data in uniq:
        lines.append("  {")
        for i in range(0, 1024, 32):
            lines.append("    " + ",".join(f"0x{b:02x}" for b in data[i:i + 32]) + ",")
        lines.append("  },")
    lines += [
        "};",
        "",
        "// Ordre de lecture : (index d'image, durée en ms). La dernière étape est le QR.",
        f"static const uint16_t INTRO_SEQ_FRAME[{len(seq)}] = {{" + ",".join(str(i) for i, _ in seq) + "};",
        f"static const uint16_t INTRO_SEQ_MS[{len(seq)}] = {{" + ",".join(str(ms) for _, ms in seq) + "};",
        "",
    ]
    path.write_text("\n".join(lines))
    return len(uniq), len(seq)


def main():
    frames = build()
    (ROOT / "preview").mkdir(exist_ok=True)
    imgs = [oled_look(f) for f, _ in frames]
    imgs[0].save(ROOT / "preview" / "intro.gif", save_all=True, append_images=imgs[1:],
                 duration=[ms for _, ms in frames], loop=0, optimize=False)

    cols = 8
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("L", (cols * (W * 2 + 4), rows * (H * 2 + 4)), 60)
    for i, (f, _) in enumerate(frames):
        im = Image.fromarray((f * 255).astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST)
        sheet.paste(im, ((i % cols) * (W * 2 + 4) + 2, (i // cols) * (H * 2 + 4) + 2))
    sheet.save(ROOT / "preview" / "intro_sheet.png")
    Image.fromarray((frames[-1][0] * 255).astype(np.uint8)).save(ROOT / "preview" / "qr_128x64.png")

    hdr = ROOT / "firmware" / "src" / "assets" / "intro_frames.h"
    hdr.parent.mkdir(parents=True, exist_ok=True)
    u, s = write_header(frames, hdr)
    anim_ms = sum(ms for _, ms in frames[:-1])
    print(f"{s} étapes, {u} images uniques ({u} Ko en flash), animation {anim_ms / 1000:.1f} s "
          f"+ QR {QR_HOLD_MS / 1000:.0f} s")


if __name__ == "__main__":
    main()
