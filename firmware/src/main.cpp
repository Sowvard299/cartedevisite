// Effet écran : le personnage arrive, tend sa carte, elle devient un QR (LinkedIn).
// Les images sont générées par tools/make_intro.py -> assets/intro_frames.h.
// Bouton sur GPIO 3 (vers GND) : pendant l'anim -> QR direct ; pendant le QR -> rejoue.
// Après QR_HOLD_MS sans appui : écran éteint, deep sleep, réveil par le bouton.
#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>
#include "esp_sleep.h"
#include "driver/gpio.h"
#include "assets/intro_frames.h"

#ifdef DISPLAY_SH1106   // option 1.3"
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#else
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#endif

static const int PIN_BUTTON = 3;
static const int PIN_SDA = 6;
static const int PIN_SCL = 7;

static bool buttonPressed() {
  static bool last = false;
  static uint32_t lastChange = 0;
  bool now = digitalRead(PIN_BUTTON) == LOW;
  if (now != last && millis() - lastChange > 30) {
    last = now;
    lastChange = millis();
    return now;  // front descendant uniquement
  }
  return false;
}

static void showFrame(uint16_t idx) {
  memcpy(u8g2.getBufferPtr(), INTRO_FRAMES[idx], 1024);
  u8g2.sendBuffer();
}

// Affiche une image et attend sa durée ; renvoie true si le bouton a été pressé.
static bool showStep(uint16_t step) {
  uint32_t t0 = millis();  // la durée inclut l'envoi I2C (~23 ms à 400 kHz)
  showFrame(INTRO_SEQ_FRAME[step]);
  while (millis() - t0 < INTRO_SEQ_MS[step]) {
    if (buttonPressed()) return true;
    delay(2);
  }
  return false;
}

// Joue l'animation ; un appui saute directement au QR.
static void playIntro() {
  for (uint16_t s = 0; s < INTRO_STEPS - 1; s++) {
    if (showStep(s)) break;
  }
}

static void goToSleep() {
  while (digitalRead(PIN_BUTTON) == LOW) delay(10);  // sinon réveil immédiat
  u8g2.setPowerSave(1);
  gpio_pullup_en((gpio_num_t)PIN_BUTTON);
  gpio_hold_en((gpio_num_t)PIN_BUTTON);
  gpio_deep_sleep_hold_en();
  esp_deep_sleep_enable_gpio_wakeup(1ULL << PIN_BUTTON, ESP_GPIO_WAKEUP_GPIO_LOW);
  esp_deep_sleep_start();
}

void setup() {
  gpio_hold_dis((gpio_num_t)PIN_BUTTON);
  setCpuFrequencyMhz(80);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  Wire.begin(PIN_SDA, PIN_SCL);
  u8g2.setBusClock(400000);
  u8g2.begin();
  u8g2.setContrast(255);
}

void loop() {
  playIntro();
  // QR : dernière étape, tenue QR_HOLD_MS ; un appui rejoue l'animation.
  if (!showStep(INTRO_STEPS - 1)) goToSleep();
}
