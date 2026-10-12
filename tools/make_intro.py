#!/usr/bin/env python3
"""Génère l'animation de la carte (128x64, 1 bit, pixel natif).

Le personnage est un mannequin blanc sans visage (tête en œuf, deux yeux
ronds lumineux), construit en 3D (capsules + ellipsoïde), rendu en
sur-échantillonnage, éclairé en aplats tramés (Bayer 8x8), avec contour
lumineux et traits intérieurs.

Scénario façon anime (~11 s, chaque plan nommé ; voir tools/audit_intro.py) :
  démarrage    : point, ligne, neige, invite "> BOOT CARD_OS", barre de progression
  1 profil     : traits de vitesse, il fonce (images rémanentes), dérape, s'immobilise
  2 gros plan  : il tourne la tête, orbites vides, les yeux s'allument (un seul
                 flash inversé), la visée se resserre : LOCK
  3 dessus     : la caméra descend, il arme, lance au ralenti, la carte file en tournoyant
  4 face       : la carte arrive de loin droit sur l'écran
  5 impact     : la carte plaque l'écran, un faisceau la numérise, ses traits se
                 décomposent en pixels qui deviennent le QR
  6 QR         : balayage, "SCAN ME", puis QR fixe QR_HOLD_MS
Au plus 2 flashes plein écran par seconde (WCAG 2.3.1 : 3 max).

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
FLOOR_Y = 63            # pieds sur la dernière ligne : tout l'écran sert au personnage

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
WINDUP = dict(legs={-1: (0.15, 0.25), 1: (-0.2, 0.2)},
              arms={-1: (1.25, 0.05, 0.15), 1: (-1.7, 0.75, 1.2)}, lean=-0.18, spread=0.12)
THROW = dict(legs={-1: (-0.1, 0.2), 1: (0.3, 0.3)},
             arms={-1: (-0.7, 0.45, 0.5), 1: (1.55, -0.3, 0.02)}, lean=0.28, spread=0.12)


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
    """Caméra orthographique : position du personnage (cx, cy = pieds à l'écran),
    orientation (yaw), plongée de la caméra (pitch, 0 = de face, pi/2 = vue de dessus)
    et zoom (scale, pixels écran par unité)."""

    def __init__(self, cx, yaw, pitch=0.0, scale=1.0, cy=FLOOR_Y):
        self.cx, self.cy, self.yaw, self.pitch, self.scale = cx, cy, yaw, pitch, scale
        self.c, self.s = cos(yaw), sin(yaw)
        self.cp, self.sp = cos(pitch), sin(pitch)

    def project(self, p):
        x, y, z = p
        X, Z = x * self.c + z * self.s, -x * self.s + z * self.c
        up, depth = y * self.cp - Z * self.sp, y * self.sp + Z * self.cp
        return self.cx + X * self.scale, self.cy - up * self.scale, depth * self.scale

    def facing(self, d):
        """Composante vers la caméra d'une direction locale."""
        Z = -d[0] * self.s + d[2] * self.c
        return d[1] * self.sp + Z * self.cp

    def centered_on(self, p, x=W / 2, y=H / 2):
        """Même caméra, recadrée pour que le point local p tombe en (x, y)."""
        px, py, _ = self.project(p)
        return Figure(self.cx + x - px, self.yaw, self.pitch, self.scale, self.cy + y - py)


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
            sc = fig.scale
            ryp = np.hypot(HEAD_RY * fig.cp, HEAD_RZ * fig.sp) * sc   # hauteur apparente selon la plongée
            rx, ry = HEAD_RX * 1.12 * sc, ryp
            x0, x1 = int(max((ax - rx - 1) * SS, 0)), int(min((ax + rx + 1) * SS, ws))
            y0, y1 = int(max((ay - ry - 1) * SS, 0)), int(min((ay + ry + 1) * SS, hs))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            v = ((yy + 0.5) / SS - ay) / ryp
            rxv = HEAD_RX * sc * (1 - 0.12 * v * fig.cp)   # plus large en haut, menton plus fin
            u = ((xx + 0.5) / SS - ax) / rxv
            d2 = u * u + v * v
            inside = d2 < 1
            h = np.sqrt(np.clip(1 - d2, 0, 1))
            z = az + HEAD_RZ * sc * h
            n = np.stack([u / rxv, v / ryp, h / (HEAD_RZ * sc)])
            n /= np.linalg.norm(n, axis=0)
            nx, ny, nz = n
        else:
            bx, by, bz = fig.project(b)
            r = (np.hypot(rl * fig.c, rd * fig.s) if rl != rd else rl) * fig.scale
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
        lit &= ~(mask & m2 & (g2 != grp) & (d2 > depth + 0.8 * fig.scale))   # une partie plus proche passe devant
    rim = mask & ~(shift(mask, 0, 1, False) & shift(mask, 0, -1, False)
                   & shift(mask, 1, 0, False) & shift(mask, -1, 0, False))
    lit |= rim
    yy, xx = np.indices((H, W))
    for x, y, f in eye_points(fig, head):
        # rond sombre, écrasé quand l'œil est vu de biais ; pupille lumineuse au centre
        r = max(1.7, 1.05 * fig.scale)
        rx, ry = max(r * f, 0.8), max(r * (0.35 + 0.65 * fig.cp), 0.8)
        d = ((xx + 0.5 - x) / rx) ** 2 + ((yy + 0.5 - y) / ry) ** 2
        ring = 1 + 1.4 / min(rx, ry)
        lit[(d > 1) & (d <= ring * ring) & mask] = True   # liseré clair : l'orbite se détache de l'ombre
        lit[d <= 1] = False
        if eyes == "glow":
            g = 0.42 if fig.scale > 1.8 else 0.3
            lit[d <= g * g] = True
            if not (d <= g * g).any() and 0 <= int(x) < W and 0 <= int(y) < H:
                lit[int(y), int(x)] = True
    return lit, mask


def eye_points(fig, head):
    out = []
    for s in (-1, 1):
        d = unit(v3(s * 0.42, -0.05, 0.9))
        f = fig.facing(d)
        if f > 0.12:
            p = head + v3(d[0] * HEAD_RX, d[1] * HEAD_RY, d[2] * HEAD_RZ)
            x, y, _ = fig.project(p)
            out.append((x, y, f))
    return out


# ================================================================ décor et effets
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
def _card(buf, P, size, front):
    """Dessine une carte dont P(u, v) (u, v dans [-1, 1]) donne les coins à l'écran."""
    img = Image.fromarray(buf.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    quad = [P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)]
    lw = max(1, int(size / 45))
    if front:
        d.polygon(quad, fill=255, outline=0)
        if size > 9:
            d.line([P(-0.75, -0.35), P(0.05, -0.35)], fill=0, width=lw)
            d.line([P(-0.75, 0.15), P(-0.2, 0.15)], fill=0, width=lw)
            d.line([P(-0.75, 0.5), P(-0.35, 0.5)], fill=0, width=lw)
            d.polygon([P(0.3, -0.55), P(0.75, -0.55), P(0.75, 0.45), P(0.3, 0.45)], outline=0)
        if size > 30:
            d.line([P(-0.95, -0.92), P(0.95, -0.92), P(0.95, 0.92), P(-0.95, 0.92), P(-0.95, -0.92)],
                   fill=0, width=lw)
    else:
        d.polygon(quad, fill=0, outline=255)             # dos de carte : noir, double liseré
        if size > 6:
            d.polygon([P(-0.7, -0.6), P(0.7, -0.6), P(0.7, 0.6), P(-0.7, 0.6)], outline=255)
    return np.array(img) > 127


def draw_card(buf, cx, cy, w, h, rot=0.0, flip=0.0):
    """Carte face caméra, tournant dans le plan (rot) et autour de son axe vertical (flip)."""
    cf = cos(flip)

    def P(u, v):
        x, y = u * w / 2 * cf, v * h / 2
        return (cx + x * cos(rot) - y * sin(rot), cy + x * sin(rot) + y * cos(rot))

    return _card(buf, P, w * abs(cf), cf > 0)


def draw_card3d(buf, fig, center, w, h, spin):
    """Carte posée à plat (horizontale) dans la scène 3D, tournant comme un frisbee."""
    def P(u, v):
        a, b = u * w / 2, v * h / 2
        x, y, _ = fig.project(center + v3(a * cos(spin) - b * sin(spin), 0, a * sin(spin) + b * cos(spin)))
        return (x, y)

    img = Image.fromarray(buf.astype(np.uint8) * 255)          # halo noir : la carte se détache
    ImageDraw.Draw(img).polygon([P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)], fill=0, outline=0, width=3)
    return _card(np.array(img) > 127, P, w * fig.scale, fig.facing(v3(0, 1, 0)) > 0)


def spotlight(buf, fig, radius=24.0):
    """Vue de dessus : flaque de lumière tramée au sol, ombre sous le personnage."""
    gx, gy, _ = fig.project(v3(0, 0, 0))
    rx, ry = radius * fig.scale, radius * fig.scale * fig.sp
    yy, xx = np.indices((H, W))
    d = ((xx - gx) / rx) ** 2 + ((yy - gy) / ry) ** 2
    pool = (d < 1) & (bayer() < 0.32 * (1 - d) + 0.04)
    buf |= pool
    return buf


# ================================================================ QR
# Police 5x7 (majuscules, chiffres, quelques signes) pour l'interface façon terminal
_GLYPHS = """
A 01110 10001 10001 11111 10001 10001 10001
B 11110 10001 10001 11110 10001 10001 11110
C 01111 10000 10000 10000 10000 10000 01111
D 11110 10001 10001 10001 10001 10001 11110
E 11111 10000 10000 11110 10000 10000 11111
F 11111 10000 10000 11110 10000 10000 10000
G 01111 10000 10000 10011 10001 10001 01111
H 10001 10001 10001 11111 10001 10001 10001
I 11111 00100 00100 00100 00100 00100 11111
J 00111 00010 00010 00010 00010 10010 01100
K 10001 10010 10100 11000 10100 10010 10001
L 10000 10000 10000 10000 10000 10000 11111
M 10001 11011 10101 10101 10001 10001 10001
N 10001 11001 10101 10011 10001 10001 10001
O 01110 10001 10001 10001 10001 10001 01110
P 11110 10001 10001 11110 10000 10000 10000
Q 01110 10001 10001 10001 10101 10010 01101
R 11110 10001 10001 11110 10100 10010 10001
S 01111 10000 10000 01110 00001 00001 11110
T 11111 00100 00100 00100 00100 00100 00100
U 10001 10001 10001 10001 10001 10001 01110
V 10001 10001 10001 10001 10001 01010 00100
W 10001 10001 10001 10101 10101 10101 01010
X 10001 10001 01010 00100 01010 10001 10001
Y 10001 10001 01010 00100 00100 00100 00100
Z 11111 00001 00010 00100 01000 10000 11111
0 01110 10011 10101 10101 11001 10001 01110
1 00100 01100 00100 00100 00100 00100 01110
2 01110 10001 00001 00010 00100 01000 11111
3 11110 00001 00001 01110 00001 00001 11110
4 00010 00110 01010 10010 11111 00010 00010
5 11111 10000 11110 00001 00001 10001 01110
6 00110 01000 10000 11110 10001 10001 01110
7 11111 00001 00010 00100 01000 01000 01000
8 01110 10001 10001 01110 10001 10001 01110
9 01110 10001 10001 01111 00001 00010 01100
> 01000 00100 00010 00001 00010 00100 01000
. 00000 00000 00000 00000 00000 01100 01100
: 00000 01100 01100 00000 01100 01100 00000
_ 00000 00000 00000 00000 00000 00000 11111
- 00000 00000 00000 11111 00000 00000 00000
/ 00001 00010 00010 00100 01000 01000 10000
% 11001 11010 00010 00100 01000 01011 10011
"""
FONT = {line[0]: line[2:].split() for line in _GLYPHS.strip().splitlines()}
FONT[" "] = ["000"] * 7
FONT["i"] = ["1", "0", "1", "1", "1", "1", "1"]
FONT["n"] = ["00000", "00000", "11110", "10001", "10001", "10001", "10001"]


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
def reticle(buf, cx, cy, half_w, half_h, arm=4, v=True):
    """Coins de visée (interface "verrouillage de cible")."""
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = int(cx + sx * half_w), int(cy + sy * half_h)
            for i in range(arm):
                for px, py in ((x - sx * i, y), (x, y - sy * i)):
                    if 0 <= px < W and 0 <= py < H:
                        buf[py, px] = v
    return buf


def build():
    """Liste de (image, durée ms, nom du plan)."""
    frames = []
    rng = np.random.default_rng(11)
    shot = ["boot"]

    def add(f, ms=FPS_MS, shake=(0, 0)):
        if shake != (0, 0):
            f = shift(f, shake[1], shake[0], False)
        frames.append((f.copy(), ms, shot[0]))

    black = np.zeros((H, W), dtype=bool)

    # ---- 0. DÉMARRAGE : point, ligne, neige légère, invite de commande
    add(black, 250)
    for wdt in (2, 20, 70, 128):
        f = black.copy()
        f[32, 64 - wdt // 2:64 + wdt // 2] = True
        add(f, 35)
    for hh in (6, 32):
        f = black.copy()
        f[32 - hh:32 + hh, :] = rng.random((2 * hh, W)) < 0.18       # neige (peu dense : pas de flash)
        add(f, 40)
    add(black, 120)
    lines = ["> BOOT CARD_OS", "> SCAN PROFILE", "> TARGET LOCKED"]
    for li, line in enumerate(lines):
        for n in range(2, len(line) + 1, 2):
            f = black.copy()
            for lj in range(li):
                text(f, 4, 6 + lj * 11, lines[lj], True)
            text(f, 4, 6 + li * 11, line[:n] + "_", True)
            add(f, 30)
        if li == 1:                                                   # barre de progression
            for p in range(0, 101, 20):
                g = f.copy()
                g[48:52, 4:124] = False
                g[48, 4:124] = g[51, 4:124] = True
                g[48:52, 4] = g[48:52, 123] = True
                g[49:51, 5:5 + int(118 * p / 100)] = True
                text(g, 4, 55, f"{p}%" if p < 100 else "OK", True)
                add(g, 60)
    f = black.copy()
    for lj, line in enumerate(lines):
        text(f, 4, 6 + lj * 11, line, True)
    add(f, 380)
    for k in range(4):                                                # l'interface s'efface en trame
        add(f & (bayer() >= (k + 1) / 4), 40)

    # ---- PLAN 1 (large, de profil) : il fonce depuis la droite et dérape
    shot[0] = "1 course + dérapage"
    S1 = 1.08
    for _ in range(4):
        add(speed_lines(black, rng, 8))
    x, ghosts, p = W + 20.0, [], 0.0
    stop_x = 64.0
    while x > stop_x + 20:
        f = speed_lines(black, rng)
        fig = Figure(x, -pi / 2, scale=S1)
        parts, head, _ = build_pose(walk=p, amp=1.0, run=True)
        for lvl, g in zip((0.45, 0.25, 0.12), ghosts[::-1]):
            put_figure(f, *g, level=lvl)
        put_figure(f, fig, parts, head)
        add(f)
        ghosts = (ghosts + [(fig, parts, head)])[-3:]
        x -= 8.0
        p += 2 * pi / 8

    dust = []
    n, x0 = 12, x
    for k in range(1, n + 1):
        t = k / n
        x = lerp(x0, stop_x, 1 - (1 - t) ** 3)
        pose = mix_pose(SKID, READY, ease((t - 0.65) / 0.35)) if t > 0.65 else SKID
        fig = Figure(x, -pi / 2, scale=S1)
        parts, head, _ = build_pose(pose)
        f = black.copy()
        if k <= 4:
            for g, lvl in zip(ghosts[::-1], (0.3, 0.15)):
                put_figure(f, *g, level=lvl)
            ghosts = ghosts[1:]
        if k < 8:
            for _ in range(4):
                dust.append([x + 4 + rng.random() * 6, 62.0, 0.8 + rng.random() * 1.8, -0.7 - rng.random() * 1.3, 0])
        for d in dust:
            d[0] += d[2]; d[1] += d[3]; d[3] += 0.25; d[4] += 1
            if d[4] < 9 and 0 <= int(d[0]) < W - 1 and 0 <= int(d[1]) < H:
                f[int(d[1]), int(d[0])] = True
                if d[4] < 3:
                    f[int(d[1]), int(d[0]) + 1] = True
        dust = [d for d in dust if d[4] < 9]
        put_figure(f, fig, parts, head)
        add(f, 45 if k < 4 else 60, shake=((1, -1, 1, 0)[k % 4] if k < 5 else 0, 0))
    add(f, 450)                                                       # respiration : il s'immobilise

    # ---- PLAN 2 (gros plan) : il tourne la tête, les yeux s'allument, la visée le verrouille
    shot[0] = "2 gros plan, yeux"
    parts, head, _ = build_pose(READY)
    for k in range(10):
        t = k / 9
        fig = Figure(64, lerp(-1.3, 0, ease(t)), scale=lerp(2.4, 2.8, t)).centered_on(head, 64, 31)
        add(put_figure(black.copy(), fig, parts, head, eyes="off"), 55)
    fig = Figure(64, 0.0, scale=2.85).centered_on(head, 64, 31)
    dark = put_figure(black.copy(), fig, parts, head, eyes="off")
    add(dark, 550)                                                    # suspense : orbites vides
    lit = put_figure(black.copy(), fig, parts, head)
    eyes = eye_points(fig, head)
    add(lit, 120)
    for size in (3, 6, 9):
        f = lit.copy()
        for ex, ey, _ in eyes:
            sparkle(f, int(ex), int(ey), size)
        add(f, 50)
    f = focus_lines(black, 64, 31, 42, rng, 18)
    f[lit] = False
    f |= lit
    for ex, ey, _ in eyes:
        sparkle(f, int(ex), int(ey), 12)
    add(~f, 70)                                                       # un seul flash inversé
    hx, hy, _ = fig.project(head)
    for k in range(7):                                                # la visée se resserre sur la tête
        t = ease(k / 6)
        fig2 = Figure(64, 0.0, scale=lerp(2.85, 3.05, k / 6)).centered_on(head, 64, 31)
        f = focus_lines(black, 64, 31, 44, rng, 16)
        put_figure(f, fig2, parts, head)
        reticle(f, hx, hy - 1, lerp(60, 21, t), lerp(31, 24, t), 5)
        add(f, 60)
    text(f, 4, 2, "LOCK", True)
    add(f, 500)

    # ---- PLAN 3 (vue de dessus, la caméra descend) : il arme et lance la carte
    shot[0] = "3 vue de dessus, lancer"

    def overhead(t):
        return Figure(64, 0.0, pitch=lerp(1.2, 0.8, t), scale=1.55).centered_on(v3(0, 26, 2), 64, 31)

    for k in range(1, 15):
        t = ease(k / 14)
        pose = mix_pose(READY, WINDUP, t)
        figk = overhead(k / 18)
        figk = Figure(figk.cx, lerp(0, -0.4, t), figk.pitch, figk.scale, figk.cy)
        parts, head, hand = build_pose(pose)
        f = spotlight(black.copy(), figk)
        put_figure(f, figk, parts, head)
        if k >= 6:
            f = draw_card3d(f, figk, hand + v3(0, 1.5, 0), 8, 5.5, 0.5)
        add(f, 55)
    add(f, 380)                                                       # tension avant le lancer
    wx, wy, _ = figk.project(hand)

    for k, (t, ms) in enumerate(((0.35, 110), (0.7, 110), (1.0, 60))):    # LANCER au ralenti puis net
        pose = mix_pose(WINDUP, THROW, t)
        base = overhead((15 + k) / 18)
        figk = Figure(base.cx, lerp(-0.4, 0.4, t), base.pitch, base.scale, base.cy)
        parts, head, hand = build_pose(pose)
        f = spotlight(black.copy(), figk)
        hx, hy, _ = figk.project(hand)
        img = Image.fromarray(f.astype(np.uint8) * 255)
        dr = ImageDraw.Draw(img)
        for o in (-4, -1, 2):
            dr.line([(wx, wy + o), ((wx + hx) / 2 + 6, (wy + hy) / 2 + o), (hx, hy + o)], fill=255)
        f = np.array(img) > 127
        put_figure(f, figk, parts, head)
        add(f, ms)
    throw = (figk, parts, head)
    release = hand + v3(0, 1.5, 0)
    trail = []
    for k in range(1, 12):
        t = k / 11
        c = release + v3(6 * t, 10 * t, 42 * t * t)
        spin = 0.5 + 10 * t
        f = spotlight(black.copy(), throw[0])
        put_figure(f, *throw)
        for tc, ts in trail[-2:]:
            f |= draw_card3d(black.copy(), throw[0], tc, 9 + 6 * t, 6 + 4 * t, ts) & (bayer() < 0.35)
        f = draw_card3d(f, throw[0], c, 9 + 6 * t, 6 + 4 * t, spin)
        if k == 1:
            f = burst(f, *throw[0].project(release)[:2], rng, 10)
        trail.append((c, spin))
        add(f, 45)

    # ---- PLAN 4 (de face) : la carte arrive de loin, droit sur l'écran
    shot[0] = "4 la carte arrive"
    trail = []
    n = 18
    for k in range(1, n + 1):
        t = k / n
        e = t ** 2.2                                                  # lente au loin, puis fonce
        cw, chh = 5 * (150 / 5) ** e, 3.2 * (78 / 3.2) ** e
        cx, cy = lerp(72, 64, t) + 3 * sin(3 * t), lerp(24, 32, t)
        rot, flip = 5 * pi * (1 - t) ** 2, 6 * pi * ease(t)
        f = focus_lines(black, cx, cy, 14 + 30 * t, rng, 14 + int(8 * t))
        for (tx, ty, tw, th, tr, tf), lvl in zip(trail[-2:], (0.25, 0.5)):
            f |= draw_card(black.copy(), tx, ty, tw, th, tr, tf) & (bayer() < lvl)
        f = draw_card(f, cx, cy, cw, chh, rot, flip)
        trail.append((cx, cy, cw, chh, rot, flip))
        add(f, 50 if t < 0.6 else 35)

    # ---- IMPACT : la carte plaque l'écran (un seul passage au blanc), éclat et secousse
    shot[0] = "5 impact + numérisation"
    card = draw_card(black.copy(), 64, 32, 150, 78, 0, 0)
    for dx, dy in ((3, 2), (-2, -1), (1, 1)):
        add(card & ~burst(black, 64, 32, rng, 14), 45, shake=(dx, dy))
    add(card, 300)

    # ---- la carte se numérise : ses traits se décomposent en pixels qui deviennent le QR
    src = np.argwhere(~card[2:-2, 2:-2]) + 2
    mods = list(zip(*np.nonzero(QR)))
    order = rng.permutation(len(mods))
    beam_frames = 8
    for k in range(beam_frames):                                      # faisceau de numérisation
        f = card.copy()
        y = int((k + 1) / beam_frames * H)
        f[max(y - 2, 0):y, :] = False
        f[:max(y - 2, 0), :] &= (bayer()[:max(y - 2, 0), :] < 0.95)   # zone numérisée : grain
        add(f, 35)
    parts_px = []
    for i, (r, c) in enumerate(mods):
        sy, sx = src[order[i] % len(src)]
        parts_px.append((sx, sy, QR_X + c * QR_M, QR_Y + r * QR_M, 2 + sy / H * 6 + rng.random() * 4))
    steps = 22
    for k in range(steps + 1):
        f = np.ones((H, W), dtype=bool)
        for sx, sy, tx, ty, d0 in parts_px:
            t = ease((k - d0) / 9)
            x = int(round(lerp(sx, tx, t)))
            y = int(round(lerp(sy, ty, t) - 10 * sin(pi * t)))       # petit saut en arc
            f[max(y, 0):max(y + QR_M, 0), max(x, 0):max(x + QR_M, 0)] = False
        add(f, 35)

    # ---- QR : balayage de lecture puis textes
    shot[0] = "6 QR + textes"
    white = np.ones((H, W), dtype=bool)
    qr_only = white.copy()
    for r, c in mods:
        module(qr_only, r, c)
    add(qr_only, 200)
    for y in range(QR_Y - 2, QR_Y + QR_N * QR_M + 2, 6):
        f = qr_only.copy()
        f[y:y + 2, QR_X - 3:QR_X + QR_N * QR_M + 3] ^= True
        add(f, 30)
    for s in ("S", "SC", "SCA", "SCAN"):
        f = qr_only.copy(); text(f, 6, 21, s, False); add(f, 50)
    for s in ("M", "ME"):
        f = qr_only.copy(); text(f, 6, 21, "SCAN", False); text(f, 12, 33, s, False); add(f, 50)
    f = qr_only.copy(); text(f, 6, 21, "SCAN", False); text(f, 12, 33, "ME", False)
    sparkle(f, LOGO_X + 7, LOGO_Y + 7, 3, False)
    add(f, 80)

    frames.append((qr_final(), QR_HOLD_MS, "QR fixe"))
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
    frames = [(f, ms) for f, ms, _ in build()]
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
