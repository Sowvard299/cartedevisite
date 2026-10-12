#!/usr/bin/env python3
"""Génère l'animation de la carte (128x64, 1 bit) :
le personnage arrive en marchant depuis la droite, se tourne face à l'écran,
tend sa carte, la carte grossit et se transforme en QR code (LinkedIn),
puis le QR reste affiché QR_HOLD_MS.

Sorties :
  preview/intro.gif                      aperçu x4, rendu "pixels d'OLED", vraies durées
  preview/intro_sheet.png                planche de toutes les images
  firmware/src/assets/intro_frames.h     images au format buffer u8g2 (SSD1306)

Usage : python tools/make_intro.py
"""
from pathlib import Path

import numpy as np
import segno
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.linkedin.com/in/nathan-le-fol-ab7b3b230"
QR_HOLD_MS = 10_000
W, H = 128, 64
FRAME_MS = 90          # cadence de base (~11 fps)
ART = 2                # 1 pixel "art" = 2x2 pixels écran (gros pixels)
SPR_W, SPR_H = 20, 28  # sprite en pixels art -> 40x56 à l'écran
FLOOR_Y = 62


# ---------------------------------------------------------------- dessin 1 bit
class Canvas:
    def __init__(self, w, h, fill=False):
        self.a = np.full((h, w), fill, dtype=bool)

    def px(self, x, y, v=True):
        x, y = int(round(x)), int(round(y))
        if 0 <= y < self.a.shape[0] and 0 <= x < self.a.shape[1]:
            self.a[y, x] = v

    def rect(self, x0, y0, x1, y1, v=True):
        """Rectangle plein, bornes incluses."""
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.px(x, y, v)

    def line(self, x0, y0, x1, y1, v=True, thick=1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(n + 1):
            t = i / n
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for dx in range(thick):
                for dy in range(thick):
                    self.px(x + dx, y + dy, v)

    def disc(self, cx, cy, r, v=True):
        for y in range(self.a.shape[0]):
            for x in range(self.a.shape[1]):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.a[y, x] = v


def dither(mask, phase=0):
    """Damier 50 % : donne un "gris" sur un écran 1 bit."""
    yy, xx = np.indices(mask.shape)
    return mask & ((xx + yy + phase) % 2 == 0)


def stamp(dst, layer, outline=True):
    """Pose un calque allumé sur dst, avec un contour éteint d'1 px pour le détacher."""
    if outline:
        grown = layer.copy()
        grown[1:, :] |= layer[:-1, :]
        grown[:-1, :] |= layer[1:, :]
        grown[:, 1:] |= layer[:, :-1]
        grown[:, :-1] |= layer[:, 1:]
        dst &= ~grown
    dst |= layer


def new_layer():
    return Canvas(SPR_W, SPR_H)


# ---------------------------------------------------------------- personnage
def legs_and_arms(phase, side):
    """Membres pour la vue de profil. phase en radians. side=+1 avant, -1 arrière."""
    s = np.sin(phase) * side
    lift = 1 if np.cos(phase) * side > 0.5 else 0
    leg = new_layer()
    foot_x = 10 - 3.5 * s
    leg.line(9, 19, foot_x, 25 - lift, thick=2)
    leg.rect(int(round(foot_x)) - 2, 26 - lift, int(round(foot_x)) + 1, 27 - lift)  # pied vers la gauche
    arm = new_layer()
    hand_x = 10 + 3 * s
    arm.line(9, 12, hand_x, 18, thick=2)
    return leg.a, arm.a


def sprite_side(phase):
    """Profil tourné vers la gauche, en train de marcher."""
    out = np.zeros((SPR_H, SPR_W), dtype=bool)
    bob = 1 if abs(np.cos(phase)) > 0.7 else 0  # le corps monte au passage des jambes
    back_leg, back_arm = legs_and_arms(phase, -1)
    front_leg, front_arm = legs_and_arms(phase, +1)

    out |= dither(back_leg, 1)
    out |= dither(back_arm, 1)

    body = new_layer()
    body.disc(10, 5, 4.6)                 # tête
    body.rect(9, 10, 11, 10)              # cou
    body.rect(7, 11, 13, 19)              # veste
    yy, xx = np.indices((SPR_H, SPR_W))
    hair = body.a & (yy <= 9) & ((yy <= 2) | ((xx >= 11) & (yy <= 7)))
    b = body.a & ~hair
    b |= dither(hair)
    b[5, 7] = False                       # œil
    b[4, 6:9] = False                     # sourcil
    b[8, 7] = False                       # bouche
    b[6, 5] = True                        # nez
    b = np.roll(b, -bob, axis=0)
    stamp(out, b)
    stamp(out, np.roll(front_leg, -0, axis=0))
    stamp(out, np.roll(front_arm, -bob, axis=0))
    return out


def front_body(eyes="open", look=0, torso=(6, 14)):
    c = new_layer()
    c.disc(10, 5, 4.6)
    c.rect(9, 10, 11, 10)
    c.rect(torso[0], 11, torso[1], 19)
    b = c.a.copy()
    yy, xx = np.indices((SPR_H, SPR_W))
    hair = b & (yy <= 9) & ((yy <= 3) | ((yy <= 5) & ((xx <= 6 + look) | (xx >= 14 + look))))
    b &= ~hair
    b |= dither(hair)
    ex = (8 + look, 12 + look)
    if eyes == "open":
        for x in ex:
            b[5, x] = False
    else:                                 # yeux fermés "^ ^" (sourire)
        for x in ex:
            b[5, x - 1] = False
            b[5, x + 1] = False
            b[4, x] = False
    b[7, 9 + look:12 + look] = False      # sourire
    b[11:17:2, 10] = False                # cravate en pointillé
    return b


def legs_front(out, gap=10):
    l = new_layer()
    l.rect(7, 20, 9, 26)
    l.rect(11, 20, 13, 26)
    l.rect(6, 27, 9, 27)
    l.rect(11, 27, 14, 27)
    stamp(out, l.a)


def sprite_turn():
    """Trois-quarts : entre le profil et la face."""
    out = np.zeros((SPR_H, SPR_W), dtype=bool)
    legs_front(out)
    stamp(out, front_body(look=-2, torso=(7, 13)))
    arm = new_layer()
    arm.line(7, 12, 7, 18, thick=2)
    stamp(out, arm.a)
    return out


def sprite_front(eyes="open", arm=0.0):
    """Face à l'écran. arm : 0 = bras le long du corps, 1 = carte tendue."""
    out = np.zeros((SPR_H, SPR_W), dtype=bool)
    legs_front(out)
    stamp(out, front_body(eyes))
    left = new_layer()
    left.line(4, 12, 4, 18, thick=2)
    stamp(out, left.a)
    right = new_layer()
    hx = 15 + (10 - 15) * arm
    hy = 18 + (15 - 18) * arm
    right.line(15, 12, hx, hy, thick=2)
    stamp(out, right.a)
    return out


def card_rect_in_hand(sx, sy, arm):
    """Rectangle (écran) de la petite carte tenue en main, ou None."""
    if arm < 1:
        return None
    hx, hy = 10, 15                       # main en coordonnées art
    x0 = sx + (hx - 4) * ART
    y0 = sy + (hy - 3) * ART
    return (x0, y0, 12, 8)


# ---------------------------------------------------------------- texte 5x7
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


# ---------------------------------------------------------------- QR
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


def bayer_order(n):
    """Ordre de révélation "pixel" des modules : matrice de Bayer 4x4 + un peu de hasard."""
    b4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
    rng = np.random.default_rng(7)
    yy, xx = np.indices((n, n))
    return b4[yy % 4, xx % 4] + rng.random((n, n)) * 0.9


def qr_frame(progress=1.0, labels=True):
    buf = np.ones((H, W), dtype=bool)     # fond allumé, modules sombres = éteints
    order = bayer_order(QR_N)
    shown = QR & (order < progress * 16)
    for r in range(QR_N):
        for c in range(QR_N):
            if shown[r, c]:
                buf[QR_Y + r * QR_M:QR_Y + (r + 1) * QR_M, QR_X + c * QR_M:QR_X + (c + 1) * QR_M] = False
    if labels:
        text(buf, 6, 21, "SCAN", False)
        text(buf, 12, 33, "ME", False)
        # logo "in" : carré sombre, lettres allumées, hors zone calme
        bx, by = QR_X + QR_N * QR_M + 10, 24
        buf[by:by + 15, bx:bx + 15] = False
        buf[by, bx] = buf[by, bx + 14] = buf[by + 14, bx] = buf[by + 14, bx + 14] = True
        text(buf, bx + 3, by + 4, "in", True)
    return buf


# ---------------------------------------------------------------- scène
def floor(buf):
    buf[FLOOR_Y, ::3] = True


def scene(sprite, sx):
    buf = np.zeros((H, W), dtype=bool)
    floor(buf)
    big = np.kron(sprite, np.ones((ART, ART), dtype=bool))
    sy = FLOOR_Y - SPR_H * ART
    for y in range(big.shape[0]):
        for x in range(big.shape[1]):
            X, Y = sx + x, sy + y
            if big[y, x] and 0 <= X < W and 0 <= Y < H:
                buf[Y, X] = True
    return buf, sy


def draw_card(buf, x, y, w, h):
    """Carte : contour éteint, intérieur allumé, deux "lignes de texte"."""
    x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + w + 1, W), min(y + h + 1, H)
    buf[y0:y1, x0:x1] = False
    buf[y:y + h, x:x + w] = True
    if w >= 10 and h >= 6:
        lw = max(w // 2, 3)
        buf[y + h // 3, x + 2:x + 2 + lw] = False
        buf[y + 2 * h // 3, x + 2:x + 2 + lw * 2 // 3] = False


def ease(t):
    return t * t * (3 - 2 * t)


def build():
    frames = []                           # liste de (tableau 64x128, durée ms)
    add = lambda f, ms=FRAME_MS: frames.append((f, ms))

    # 0. écran vide, juste le sol
    empty = np.zeros((H, W), dtype=bool)
    floor(empty)
    add(empty, 400)

    # 1. marche depuis la droite
    cx = (W - SPR_W * ART) // 2
    x, step, i = W, 4, 0
    while x > cx:
        x = max(x - step, cx)
        f, _ = scene(sprite_side(i * np.pi / 3), x)
        add(f)
        i += 1
    f, _ = scene(sprite_side(np.pi / 2), cx)  # arrêt, pieds joints
    add(f, 180)

    # 2. se tourne face à l'écran
    add(scene(sprite_turn(), cx)[0], 140)
    add(scene(sprite_front(), cx)[0], 450)
    add(scene(sprite_front("happy"), cx)[0], 350)
    add(scene(sprite_front(), cx)[0], 200)

    # 3. tend la carte
    for a in (0.35, 0.7, 1.0):
        add(scene(sprite_front("happy", a), cx)[0], 100)
    f, sy = scene(sprite_front("happy", 1.0), cx)
    card = card_rect_in_hand(cx, sy, 1.0)
    draw_card(f, *card)
    add(f, 600)

    # 4. la carte grossit jusqu'à remplir l'écran (pas de 2 px = gros pixels)
    base, _ = scene(sprite_front("happy", 1.0), cx)
    x0, y0, w0, h0 = card
    n = 9
    for k in range(1, n + 1):
        t = ease(k / n)
        x = int(x0 * (1 - t)) // 2 * 2
        y = int(y0 * (1 - t)) // 2 * 2
        w = int(w0 + (W - w0) * t + 1) // 2 * 2
        h = int(h0 + (H - h0) * t + 1) // 2 * 2
        f = base.copy()
        draw_card(f, x, y, w, h)
        add(f, 70)
    add(np.ones((H, W), dtype=bool), 150)   # écran entièrement allumé

    # 5. le QR apparaît module par module (dissolution "pixel")
    for k in range(1, 9):
        add(qr_frame(k / 8, labels=False), 80)
    add(qr_frame(1.0, labels=True), QR_HOLD_MS)
    return frames


# ---------------------------------------------------------------- sorties
def to_u8g2(buf):
    """Buffer u8g2 plein SSD1306 : 8 pages x 128 colonnes, bit 0 = pixel du haut."""
    out = bytearray(W * H // 8)
    for page in range(H // 8):
        for x in range(W):
            v = 0
            for b in range(8):
                if buf[page * 8 + b, x]:
                    v |= 1 << b
            out[page * W + x] = v
    return bytes(out)


def oled_look(buf, scale=4):
    """Aperçu : chaque pixel = carré 3x3 dans une case 4x4, blanc légèrement froid."""
    h, w = buf.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    img[:] = (6, 6, 8)
    on = np.array([235, 240, 255], dtype=np.uint8)
    for y, x in zip(*np.nonzero(buf)):
        img[y * scale:y * scale + scale - 1, x * scale:x * scale + scale - 1] = on
    pad = 6 * scale
    framed = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, 3), dtype=np.uint8)
    framed[:] = (20, 20, 22)
    framed[pad:-pad, pad:-pad] = img
    return Image.fromarray(framed)


def write_header(frames, path):
    uniq, index = [], {}
    seq = []
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
        f"static const uint8_t INTRO_SEQ_FRAME[{len(seq)}] = {{" + ",".join(str(i) for i, _ in seq) + "};",
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

    cols = 6
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
