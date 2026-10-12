#!/usr/bin/env python3
"""Génère l'animation de la carte (128x64, 1 bit, pixel natif).

Le personnage est un mannequin blanc sans visage (tête en œuf, deux yeux
ronds lumineux), construit en 3D (capsules + ellipsoïde), rendu en
sur-échantillonnage, éclairé en aplats tramés (Bayer 8x8), avec contour
lumineux et traits intérieurs.

Scénario façon anime :
  allumage tube cathodique -> traits de vitesse, il fonce depuis la droite
  (images rémanentes) -> dérapage (poussière, secousse) -> demi-tour éclair
  -> les yeux s'allument (image d'impact inversée) -> il arme son lancer
  (lignes de concentration) -> lance la carte qui tournoie vers l'écran ->
  impact (éclat, secousse) -> les modules du QR tombent en place ->
  balayage -> textes -> QR fixe QR_HOLD_MS.

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
# Mannequin blanc : tête en œuf, deux yeux ronds lumineux, aucun autre détail.
# Repère local : x = côté droit du personnage, y = vers le haut, z = vers l'avant.
# yaw = 0 : face à l'écran ; yaw = -90° : de profil, tourné vers la gauche.
LIGHT = unit(v3(-0.55, -0.6, 0.6))      # repère écran (y vers le bas), lumière haut-gauche-avant
G_BODY, G_HEAD, G_LLEG, G_RLEG, G_LARM, G_RARM = range(6)
THIGH, SHIN, UPPER, FORE = 9.5, 9.5, 8.5, 7.5
HEAD_RX, HEAD_RY, HEAD_RZ = 5.3, 6.8, 5.6
HIP_Y = 22.2                            # hauteur des hanches debout

# Poses clés (les angles sont en radians ; bras : tangage avant, écart, coude)
STAND = dict(legs={-1: (0.0, 0.05), 1: (0.0, 0.05)},
             arms={-1: (0.0, 0.12, 0.25), 1: (0.0, 0.12, 0.25)}, lean=0.0, spread=0.0)
SKID = dict(legs={-1: (0.6, 0.55), 1: (-0.45, 0.35)},
            arms={-1: (1.0, 0.5, 0.7), 1: (-0.9, 0.55, 0.3)}, lean=-0.32, spread=0.0)
READY = dict(legs={-1: (0.05, 0.35), 1: (-0.05, 0.35)},
             arms={-1: (0.25, 0.3, 0.6), 1: (0.25, 0.3, 0.6)}, lean=0.05, spread=0.2)
WINDUP = dict(legs={-1: (0.15, 0.45), 1: (-0.2, 0.4)},
              arms={-1: (1.25, 0.05, 0.15), 1: (-1.7, 0.75, 1.2)}, lean=-0.18, spread=0.22)
THROW = dict(legs={-1: (-0.1, 0.4), 1: (0.25, 0.5)},
             arms={-1: (-0.7, 0.45, 0.5), 1: (1.55, -0.3, 0.02)}, lean=0.28, spread=0.22)


def mix_pose(a, b, t):
    out = {}
    for key in a:
        if isinstance(a[key], dict):
            out[key] = {s: tuple(lerp(x, y, t) for x, y in zip(a[key][s], b[key][s])) for s in a[key]}
        else:
            out[key] = lerp(a[key], b[key], t)
    return out


def build_pose(P=None, walk=0.0, amp=0.0, run=False):
    """Liste des primitives en coordonnées locales + tête + main droite.

    P : pose explicite (dict comme STAND) ; sinon cycle de marche/course (walk, amp).
    """
    parts = []

    def cap(a, b, r, grp, rdep=None):
        parts.append(("cap", a, b, r, r if rdep is None else rdep, grp))

    P = P or {}
    spread, lean = P.get("spread", 0.0), P.get("lean", 0.0)
    if not P:
        lean = 0.32 * amp if run else 0.04 * amp

    legs = {}
    for sgn, grp in ((-1, G_LLEG), (1, G_RLEG)):
        q = walk + (0 if sgn < 0 else pi)
        if "legs" in P:
            a, k = P["legs"][sgn]
        elif run:
            a = amp * 0.75 * sin(q)
            k = amp * (0.25 + 1.5 * max(0.0, cos(q)) ** 1.2)
        else:
            a = amp * 0.5 * sin(q)
            k = amp * (0.12 + 0.95 * max(0.0, cos(q)) ** 1.5)
        hip = v3(sgn * 3.0, 0, 0)
        knee = hip + THIGH * unit(v3(sgn * spread, -cos(a), sin(a)))
        ankle = knee + SHIN * unit(v3(sgn * spread * 0.5, -cos(a - k), sin(a - k)))
        legs[sgn] = (hip, knee, ankle, grp)
    hip_y = 0.3 + 3.0 - min(l[2][1] for l in legs.values())   # le pied le plus bas touche le sol
    lift = v3(0, hip_y, 0)
    for sgn, (hip, knee, ankle, grp) in legs.items():
        hip, knee, ankle = hip + lift, knee + lift, ankle + lift
        cap(hip, knee, 2.6, grp)
        cap(knee, ankle, 2.1, grp)
        cap(ankle + v3(0, -1.6, -1.0), ankle + v3(0, -1.6, 3.4), 1.4, grp)  # pied

    pivot = v3(0, hip_y, 0)

    def up(x, y, z):
        """Point du haut du corps, incliné de `lean` autour des hanches."""
        r = v3(x, y - HIP_Y, z)
        return pivot + v3(r[0], r[1] * cos(lean) - r[2] * sin(lean), r[1] * sin(lean) + r[2] * cos(lean))

    cap(up(-2.6, 23.0, 0), up(2.6, 23.0, 0), 3.3, G_BODY)               # bassin
    cap(up(0, 24.5, 0), up(0, 28.5, 0), 4.3, G_BODY, 3.2)               # taille
    cap(up(0, 30.5, 0), up(0, 34.0, 0), 6.0, G_BODY, 3.8)               # poitrine
    cap(up(-5.2, 36.0, 0), up(5.2, 36.0, 0), 2.5, G_BODY)               # épaules
    cap(up(0, 37.5, 0), up(0, 41.0, 0), 1.7, G_BODY)                    # cou
    head = up(0, 46.6, 0.3)
    parts.append(("egg", head, None, None, None, G_HEAD))

    hand_r = None
    for sgn, grp in ((-1, G_LARM), (1, G_RARM)):
        q = walk + (0 if sgn < 0 else pi)
        if "arms" in P:
            pitch, side, bend = P["arms"][sgn]
        elif run:
            pitch, side, bend = -amp * 0.85 * sin(q), 0.12, 1.4
        else:
            pitch = -amp * 0.55 * sin(q)
            side, bend = 0.12, 0.3 + 0.4 * max(0.0, pitch)
        sh = up(sgn * 7.0, 36.0, 0)
        d1 = unit(v3(sgn * side, -cos(pitch), sin(pitch)))
        d1 = up(*(d1 + v3(0, HIP_Y, 0))) - pivot                       # suit l'inclinaison
        elbow = sh + UPPER * d1
        fp = pitch + bend
        d2 = unit(v3(sgn * side * 0.5, -cos(fp), sin(fp)))
        d2 = up(*(d2 + v3(0, HIP_Y, 0))) - pivot
        wrist = elbow + FORE * d2
        hand = wrist + 1.3 * d2
        cap(sh, elbow, 1.9, grp)
        cap(elbow, wrist, 1.6, grp)
        cap(hand, hand, 1.9, grp)
        if sgn == 1:
            hand_r = hand
    return parts, head, hand_r


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
    for kind, a, b, rl, rd, g in parts:
        ax, ay, az = fig.project(a)
        if kind == "egg":
            # ellipsoïde de révolution (axe vertical) : sa projection ne dépend pas du yaw
            rx, ry = HEAD_RX * 1.12, HEAD_RY
            x0, x1 = int(max((ax - rx - 1) * SS, 0)), int(min((ax + rx + 1) * SS, ws))
            y0, y1 = int(max((ay - ry - 1) * SS, 0)), int(min((ay + ry + 1) * SS, hs))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            v = ((yy + 0.5) / SS - ay) / HEAD_RY
            rxv = HEAD_RX * (1 - 0.12 * v)                 # plus large en haut, menton plus fin
            u = ((xx + 0.5) / SS - ax) / rxv
            d2 = u * u + v * v
            inside = d2 < 1
            h = np.sqrt(np.clip(1 - d2, 0, 1))
            z = az + HEAD_RZ * h
            n = np.stack([u / rxv, v / HEAD_RY, h / HEAD_RZ])
            n /= np.linalg.norm(n, axis=0)
            nx, ny, nz = n
        else:
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
            nx, ny, nz = qx / r, qy / r, h
        win = inside & (z > depth[y0:y1, x0:x1])
        if not win.any():
            continue
        lit = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
        shade = 0.38 + 0.85 * lit
        shade = np.select([shade < 0.42, shade < 0.62], [0.25, 0.5], 1.0)   # aplats (cel-shading)
        tone[y0:y1, x0:x1][win] = shade[win]
        depth[y0:y1, x0:x1][win] = z[win]
        grp[y0:y1, x0:x1][win] = g

    # réduction à la résolution écran
    cnt = (grp >= 0).reshape(H, SS, W, SS).sum(axis=(1, 3))
    tsum = np.where(grp >= 0, tone, 0).reshape(H, SS, W, SS).sum(axis=(1, 3))
    tone_n = np.where(cnt > 0, tsum / np.maximum(cnt, 1), 0)
    blk_d = depth.reshape(H, SS, W, SS).transpose(0, 2, 1, 3).reshape(H, W, SS * SS)
    blk_g = grp.reshape(H, SS, W, SS).transpose(0, 2, 1, 3).reshape(H, W, SS * SS)
    idx = blk_d.argmax(axis=2)
    depth_n = np.take_along_axis(blk_d, idx[..., None], 2)[..., 0]
    grp_n = np.take_along_axis(blk_g, idx[..., None], 2)[..., 0]
    return tone_n, cnt >= SS * SS * 0.45, grp_n, depth_n


def shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    ys = slice(max(dy, 0), H + min(dy, 0)); yd = slice(max(-dy, 0), H + min(-dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0)); xd = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def ink_figure(fig, parts, head, eyes="glow"):
    """Tramage + contour lumineux + traits intérieurs + yeux."""
    tone, mask, grp, depth = render_figure(fig, parts)
    lit = mask & (tone > bayer(int(fig.cx)))
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        m2, g2, d2 = shift(mask, dy, dx, False), shift(grp, dy, dx, -1), shift(depth, dy, dx, -1e9)
        lit &= ~(mask & m2 & (g2 != grp) & (d2 > depth + 0.8))   # une partie plus proche passe devant
    rim = mask & ~(shift(mask, 0, 1, False) & shift(mask, 0, -1, False)
                   & shift(mask, 1, 0, False) & shift(mask, -1, 0, False))
    lit |= rim
    for x, y, f in eye_points(fig, head):
        if f > 0.55:                    # œil de face : rond sombre, pupille lumineuse
            for ex, ey in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
                           (0, -2), (0, 2), (-2, 0), (2, 0)):
                if 0 <= x + ex < W and 0 <= y + ey < H:
                    lit[y + ey, x + ex] = False
            if eyes == "glow":
                lit[y, x] = True
        elif f > 0.12:                  # œil de profil : fente
            for ey in (-1, 0, 1):
                if 0 <= y + ey < H and 0 <= x < W:
                    lit[y + ey, x] = ey == 0 and eyes == "glow"
    return lit, mask


def eye_points(fig, head):
    out = []
    for s in (-1, 1):
        d = unit(v3(s * 0.42, -0.05, 0.9))
        f = fig.facing(d)
        if f > 0.12:
            p = head + v3(d[0] * HEAD_RX, d[1] * HEAD_RY, d[2] * HEAD_RZ)
            x, y, _ = fig.project(p)
            out.append((int(round(x)), int(round(y)), f))
    return out


# ================================================================ décor et effets
def floor(buf, shadow_x=None):
    """Sol en perspective : rangées de points de plus en plus serrées."""
    for y, step, off in ((58, 8, 0), (60, 5, 2), (62, 3, 1), (63, 2, 0)):
        buf[y, off::step] = True
    if shadow_x is not None:
        yy, xx = np.indices((H, W))
        sh = ((xx - shadow_x) / 12.0) ** 2 + ((yy - 61) / 2.6) ** 2 < 1
        buf[sh] = False


def speed_lines(buf, rng, n=14):
    """Traits horizontaux de vitesse (le personnage file vers la gauche)."""
    img = Image.fromarray(buf.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        y = int(rng.integers(2, 57))
        x = int(rng.integers(-20, W))
        d.line([(x, y), (x + int(rng.integers(12, 46)), y)], fill=255)
    return np.array(img) > 127


def focus_lines(buf, cx, cy, r_in, rng, n=26):
    """Lignes de concentration manga : triangles fins venant des bords vers le centre."""
    img = Image.fromarray(buf.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        a = rng.random() * 2 * pi
        r0 = r_in + rng.random() * 14
        w = 0.008 + rng.random() * 0.028
        tip = (cx + r0 * cos(a), cy + r0 * sin(a) * 0.62)
        b1 = (cx + 120 * cos(a - w), cy + 120 * sin(a - w) * 0.62)
        b2 = (cx + 120 * cos(a + w), cy + 120 * sin(a + w) * 0.62)
        d.polygon([tip, b1, b2], fill=255)
    return np.array(img) > 127


def burst(buf, cx, cy, rng, n=22, v=True):
    """Éclat d'impact : rayons depuis un point."""
    img = Image.fromarray(buf.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        a = rng.random() * 2 * pi
        r0, r1 = 6 + rng.random() * 10, 50 + rng.random() * 80
        d.line([(cx + r0 * cos(a), cy + r0 * sin(a)), (cx + r1 * cos(a), cy + r1 * sin(a))],
               fill=255 if v else 0, width=int(1 + rng.integers(0, 2)))
    return np.array(img) > 127


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


def put_figure(buf, fig, parts, head, eyes="glow", level=1.0):
    """Pose le personnage sur buf ; level < 1 = image fantôme tramée."""
    lit, mask = ink_figure(fig, parts, head, eyes)
    if level < 1:
        keep = bayer() < level
        buf |= lit & keep
        return buf
    buf[mask] = lit[mask]
    return buf


# ================================================================ carte
def draw_card(buf, cx, cy, w, h, rot=0.0, flip=0.0):
    """Carte tournant dans le plan (rot) et autour de son axe vertical (flip)."""
    img = Image.fromarray(buf.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    cf = cos(flip)

    def P(u, v):
        x, y = u * w / 2 * cf, v * h / 2
        return (cx + x * cos(rot) - y * sin(rot), cy + x * sin(rot) + y * cos(rot))

    quad = [P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)]
    lw = max(1, int(w / 45))
    if cf > 0:
        d.polygon(quad, fill=255, outline=0)
        if w * cf > 9:
            d.line([P(-0.75, -0.35), P(0.05, -0.35)], fill=0, width=lw)
            d.line([P(-0.75, 0.15), P(-0.2, 0.15)], fill=0, width=lw)
            d.line([P(-0.75, 0.5), P(-0.35, 0.5)], fill=0, width=lw)
            d.polygon([P(0.3, -0.55), P(0.75, -0.55), P(0.75, 0.45), P(0.3, 0.45)], outline=0)
        if w > 30:
            d.line([P(-0.95, -0.92), P(0.95, -0.92), P(0.95, 0.92), P(-0.95, 0.92), P(-0.95, -0.92)],
                   fill=0, width=lw)
        return np.array(img) > 127
    d.polygon(quad, fill=0, outline=255)                 # dos de carte : noir, double liseré
    if w * -cf > 6:
        d.polygon([P(-0.7, -0.6), P(0.7, -0.6), P(0.7, 0.6), P(-0.7, 0.6)], outline=255)
    return np.array(img) > 127


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
    rng = np.random.default_rng(11)

    def add(f, ms=FPS_MS, shake=(0, 0)):
        if shake != (0, 0):
            f = shift(f, shake[1], shake[0], False)
        frames.append((f.copy(), ms))

    black = np.zeros((H, W), dtype=bool)
    stage = black.copy()
    floor(stage)

    # 0. allumage façon tube cathodique
    add(black, 200)
    for wdt in (2, 20, 70, 128):
        f = black.copy()
        f[32, 64 - wdt // 2:64 + wdt // 2] = True
        add(f, 35)
    for k, hh in enumerate((8, 20, 32)):
        f = black.copy()
        f[32 - hh:32 + hh, :] = stage[32 - hh:32 + hh, :]
        f[32 - hh, :] = f[32 + hh - 1, :] = True
        if k == 0:
            f[1::2, :] = False
        add(f, 35)
    add(stage, 220)

    # 1. traits de vitesse annoncent l'arrivée, puis le personnage fonce depuis la droite
    for _ in range(3):
        add(speed_lines(stage, rng, 8))
    x, ghosts, p = W + 16.0, [], 0.0
    stop_x = 62.0
    while x > stop_x + 18:
        f = speed_lines(stage, rng)
        fig = Figure(x, -pi / 2)
        parts, head, _ = build_pose(walk=p, amp=1.0, run=True)
        for lvl, g in zip((0.45, 0.25, 0.12), ghosts[::-1]):   # images rémanentes
            put_figure(f, *g, level=lvl)
        floor(f, x)
        put_figure(f, fig, parts, head)
        add(f)
        ghosts = (ghosts + [(fig, parts, head)])[-3:]
        x -= 9.0
        p += 2 * pi / 7

    # 2. dérapage : penché en arrière, poussière et étincelles, secousse
    dust = []
    n = 9
    x0 = x
    for k in range(1, n + 1):
        t = k / n
        x = lerp(x0, stop_x, 1 - (1 - t) ** 3)
        pose = mix_pose(SKID, READY, ease((t - 0.6) / 0.4)) if t > 0.6 else SKID
        fig = Figure(x, -pi / 2)
        parts, head, _ = build_pose(pose)
        f = stage.copy()
        floor(f, x)
        if k <= 4:
            for g, lvl in zip(ghosts[::-1], (0.3, 0.15)):
                put_figure(f, *g, level=lvl)
            ghosts = ghosts[1:]
        if k < 7:
            for _ in range(4):
                dust.append([x + 4 + rng.random() * 6, 60.0, 0.8 + rng.random() * 1.8, -0.6 - rng.random() * 1.2, 0])
        for d in dust:
            d[0] += d[2]; d[1] += d[3]; d[3] += 0.25; d[4] += 1
            if d[4] < 8 and 0 <= int(d[0]) < W and 0 <= int(d[1]) < H:
                f[int(d[1]), int(d[0])] = True
                if d[4] < 3 and int(d[0]) + 1 < W:
                    f[int(d[1]), int(d[0]) + 1] = True
        dust = [d for d in dust if d[4] < 8]
        put_figure(f, fig, parts, head)
        add(f, 45, shake=((1, -1, 1, 0)[k % 4] if k < 5 else 0, 0))

    # 3. demi-tour éclair vers l'écran
    for k in range(1, 5):
        t = ease(k / 4)
        fig = Figure(stop_x, lerp(-pi / 2, 0, t))
        parts, head, _ = build_pose(READY)
        f = stage.copy(); floor(f, stop_x)
        add(put_figure(f, fig, parts, head), 35)
    fig = Figure(stop_x, 0.0)
    parts, head, _ = build_pose(READY)
    ready = stage.copy()
    floor(ready, stop_x)
    put_figure(ready, fig, parts, head)
    add(ready, 260)

    # 4. les yeux s'allument : éclat + image d'impact inversée
    eyes = eye_points(fig, head)
    for size in (2, 4):
        f = ready.copy()
        for ex, ey, _ in eyes:
            sparkle(f, ex, ey, size)
        add(f, 50)
    f = ready.copy()
    for ex, ey, _ in eyes:
        sparkle(f, ex, ey, 6)
    add(~f, 70)
    add(ready, 160)

    # 5. armé du lancer, lignes de concentration
    cy_focus = 34
    for k in range(1, 8):
        t = ease(k / 7)
        pose = mix_pose(READY, WINDUP, t)
        figk = Figure(stop_x, lerp(0, -0.35, t))
        parts, head, hand = build_pose(pose)
        f = focus_lines(black, stop_x, cy_focus, 38, rng)
        floor(f, stop_x)
        put_figure(f, figk, parts, head)
        if k >= 3:                                        # la carte apparaît entre les doigts
            hx, hy, _ = figk.project(hand)
            f = draw_card(f, hx, hy - 2, 7, 4.5, rot=-0.6)
        add(f)
    for _ in range(3):                                    # tension
        f = focus_lines(black, stop_x, cy_focus, 34, rng, 36)
        floor(f, stop_x)
        put_figure(f, figk, parts, head)
        f = draw_card(f, hx, hy - 2, 7, 4.5, rot=-0.6)
        add(f, 60)
    wind_hand = (hx, hy - 2)

    # 6. LANCER : bras en traînée, la carte part en tournoyant vers l'écran
    for k, t in enumerate((0.45, 1.0)):
        pose = mix_pose(WINDUP, THROW, t)
        figk = Figure(stop_x, lerp(-0.35, 0.35, t))
        parts, head, hand = build_pose(pose)
        f = focus_lines(black, stop_x, cy_focus, 38, rng)
        floor(f, stop_x)
        hx, hy, _ = figk.project(hand)
        img = Image.fromarray(f.astype(np.uint8) * 255)
        dr = ImageDraw.Draw(img)
        for o in (-3, 0, 3):                              # traînée du geste
            dr.line([(wind_hand[0] + o, wind_hand[1]), ((wind_hand[0] + hx) / 2 + o, min(wind_hand[1], hy) - 6),
                     (hx + o, hy)], fill=255, width=1)
        f = np.array(img) > 127
        put_figure(f, figk, parts, head)
        add(f, 40)
    throw_fig, throw_parts, throw_head = figk, parts, head
    start = (hx, hy)

    n = 12
    trail = []
    for k in range(1, n + 1):
        t = k / n
        e = t ** 1.7
        cw, chh = 7 * (150 / 7) ** e, 4.5 * (78 / 4.5) ** e
        cx = lerp(start[0], 64, ease(t))
        cy = lerp(start[1], 32, ease(t)) - 16 * sin(pi * min(t * 1.4, 1))   # trajectoire en cloche
        rot = 4 * pi * (1 - t) ** 2
        flip = 4 * pi * ease(t)
        f = focus_lines(black, cx, cy, 20 + 30 * t, rng, 24)
        floor(f, stop_x)
        put_figure(f, throw_fig, throw_parts, throw_head)
        for (tx, ty, tw, th, tr, tf), lvl in zip(trail[-2:], (0.25, 0.5)):
            ghost = draw_card(black.copy(), tx, ty, tw, th, tr, tf)
            f |= ghost & (bayer() < lvl)
        f = draw_card(f, cx, cy, cw, chh, rot, flip)
        trail.append((cx, cy, cw, chh, rot, flip))
        add(f, 40 if t < 0.7 else 30)

    # 7. IMPACT : écran blanc, éclat noir, secousse
    white = np.ones((H, W), dtype=bool)
    add(white, 40)
    add(burst(black, 64, 32, rng, 26), 50)
    add(~burst(black, 64, 32, rng, 18), 50, shake=(2, 1))
    add(white, 40, shake=(-2, 0))
    f = white.copy(); f[::2, :] = False
    add(f, 35)
    add(white, 60)

    # 8. les modules tombent en place : d'abord les trois repères, puis le reste
    mods = list(zip(*np.nonzero(QR)))
    start_at, fall = {}, 4
    for r, c in mods:
        start_at[(r, c)] = rng.integers(0, 3) if in_finder(r, c) else 3 + c * 0.35 + rng.random() * 4
    end = int(max(start_at.values())) + fall + 1
    for fr in range(end):
        f = white.copy()
        landed_finders = fr == int(2 + fall)
        for (r, c), s0 in start_at.items():
            t = (fr - s0) / fall
            if t <= 0:
                continue
            ty = QR_Y + r * QR_M
            module(f, r, c, ty if t >= 1 else lerp(-QR_M - 2, ty, t * t))
        add(f, 30, shake=(1, 0) if landed_finders else (0, 0))

    # 9. balayage de lecture puis textes
    qr_only = white.copy()
    for r, c in mods:
        module(qr_only, r, c)
    for y in range(QR_Y - 2, QR_Y + QR_N * QR_M + 2, 8):
        f = qr_only.copy()
        f[y:y + 2, QR_X - 3:QR_X + QR_N * QR_M + 3] ^= True
        add(f, 28)
    for s in ("S", "SC", "SCA", "SCAN"):
        f = qr_only.copy(); text(f, 6, 21, s, False); add(f, 40)
    for s in ("M", "ME"):
        f = qr_only.copy(); text(f, 6, 21, "SCAN", False); text(f, 12, 33, s, False); add(f, 40)
    f = qr_only.copy(); text(f, 6, 21, "SCAN", False); text(f, 12, 33, "ME", False)
    sparkle(f, LOGO_X + 7, LOGO_Y + 7, 3, False)
    add(f, 60)

    add(qr_final(), QR_HOLD_MS)
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
