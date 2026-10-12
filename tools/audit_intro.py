#!/usr/bin/env python3
"""Audit chiffré de l'animation générée par make_intro.py.

Mesure, image par image : rythme (durées), mouvement (pixels qui changent),
remplissage de l'écran, et flashes plein écran. Le seuil des flashes suit la
règle WCAG 2.3.1 (au plus 3 flashes par seconde) : une démo publique ne doit
pas risquer de déclencher une crise photosensible.

Usage : python tools/audit_intro.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_intro as m  # noqa: E402


def main():
    frames = m.build()
    anim = frames[:-1]
    labels, imgs, ms = [], [], []
    for fr in anim:
        imgs.append(fr[0]); ms.append(fr[1]); labels.append(fr[2] if len(fr) > 2 else "?")
    t = np.cumsum([0] + ms[:-1])
    lit = np.array([f.mean() for f in imgs])
    diff = np.array([0.0] + [(a ^ b).mean() for a, b in zip(imgs[:-1], imgs[1:])])

    print(f"Durée de l'animation : {sum(ms) / 1000:.1f} s  ({len(anim)} étapes)  + QR {frames[-1][1] / 1000:.0f} s")
    print(f"Durée d'image : min {min(ms)} ms, médiane {int(np.median(ms))} ms, max {max(ms)} ms")

    # plans
    print("\nPlans :")
    start = 0
    for i in range(1, len(anim) + 1):
        if i == len(anim) or labels[i] != labels[start]:
            dur = sum(ms[start:i])
            print(f"  {t[start] / 1000:5.2f} s  {labels[start]:<28} {dur / 1000:4.2f} s  "
                  f"mouvement moyen {diff[start + 1:i].mean() * 100 if i - start > 1 else 0:4.1f} %")
            start = i

    # flashes : grande variation de luminosité moyenne d'une image à l'autre
    flashes = [t[i] for i in range(1, len(anim)) if abs(lit[i] - lit[i - 1]) > 0.3]
    worst = max((sum(1 for f in flashes if s <= f < s + 1000) for s in flashes), default=0)
    print(f"\nFlashes plein écran : {len(flashes)} ; pire fenêtre d'1 s : {worst} "
          f"({'OK' if worst <= 3 else 'TROP (WCAG 2.3.1 : 3 max/s)'})")

    # pauses : moments où rien ne bouge pendant plus de 400 ms
    still, acc = [], 0
    for i in range(len(anim)):
        acc += ms[i]                                   # temps affiché sans changement
        if i + 1 < len(anim) and diff[i + 1] < 0.002:
            continue
        if acc >= 400:
            still.append(f"{labels[i]} ({acc} ms)")
        acc = 0
    print(f"Respirations (≥ 400 ms sans mouvement) : {', '.join(still) if still else 'aucune'}")
    print(f"Images très chargées (45–70 % de pixels allumés, fond noir) : "
          f"{int(((lit > 0.45) & (lit < 0.7)).sum())}")


if __name__ == "__main__":
    main()
