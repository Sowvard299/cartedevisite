# cartedevisite

Carte de visite électronique : écran OLED 0,96" 128×64 (SSD1306) + ESP32-C3.
À l'appui, un mannequin pixel-art arrive, lance sa carte (nom, accroche, école),
qui se numérise en QR code LinkedIn.

## Personnaliser une carte

1. Ajouter une entrée dans `content/config.yaml` (nom, accroche, école visée, URL LinkedIn).
2. `pip install -r tools/requirements.txt`
3. `python tools/make_intro.py --unit N` → aperçus dans `preview/unit<N>_*`
   et images du firmware dans `firmware/src/assets/intro_frames.h`.
4. `python tools/audit_intro.py N` → durée par plan, flashes (WCAG), respirations.
5. `cd firmware && pio run -t upload` (interrupteur batterie sur OFF avant de brancher l'USB).

Le générateur refuse une carte dont le texte ne tient pas à l'écran ou dont l'URL
ne tient pas dans un QR version 3.
