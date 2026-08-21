# 🌘 Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio

Profesionální desktopová aplikace v Pythonu (PyQt6 + OpenCV) pro skládání expozičních řad fotografií do jednoho HDR snímku s vysokým dynamickým rozsahem. Navržena speciálně pro náročné astrofotografické scény jako je **úplné zatmění Slunce**, korona, protuberance i pro klasické pozemské HDR fotografie.

---

## ✨ Klíčové funkce

- **🎯 Bleskový výřez ROI (300x300 px / Fast Edit Mode)**:
  - Přepínání jedním kliknutím mezi celým snímkem a **výřezem kolem Slunce** (300x300, 450x450, 600x600 px).
  - V režimu výřezu probíhá složení a veškeré úpravy v reálném čase (**odezva pod 30 ms**).
  - Tlačítko **`☀️ Najít Slunce`** nebo kliknutí myší na obrazovku okamžitě vycentruje výřez na zatmění.
  - Při exportu se automaticky spočítá a uloží **100% plné rozlišení v 16-bitové hloubce**.
- **Automatické rozpoznání EV a expozičních časů**:
  - Automatické vyčtení expozičních časů, ISO a clony z **EXIF** metadat.
  - Inteligentní analýza jasu scény: Pokud EXIF chybí, aplikace analyzuje histogram a seřadí snímky od nejkratší expozice (-EV) po nejdelší (+EV).
  - Možnost definovat krok (např. **9 snímků po 1.0 EV**).
- **Zarovnání a detekce černého disku Měsíce**:
  - 🌑 **Detekce černého disku Měsíce v záři korony**: Hledá kruhový černý disk Měsíce obklopený světlem korony s filtrem na geometrickou cirkularitu.
  - 🛠️ **Interaktivní ruční dozarovnání fotku po fotce**:
    - Režim **Rozdíl hran (Difference)**, **50% průhledné překrytí (Blend)** i **Blikání (Flicker)**.
    - Možnost posouvat každý snímek po 0.5px / 1px / 5px šipkami na klávesnici i tlačítky na obrazovce.
- **Skládací algoritmy (HDR & Fusion Engines)**:
  - **Mertens Exposure Fusion** (*Doporučeno pro zatmění Slunce*): Laplaceova pyramidová bezešvá fúze s potlačením šumu.
  - **Debevec 32-bit HDR** s kalibrací křivky odezvy snímače (CRF) a tonemappingem (Reinhard, Drago, Mantiuk).
  - **Robertson 32-bit HDR**.
- **Potlačení šumu & Filtr sluneční korony**:
  - 🛡️ **Redukce šumu (Grain filter)**: Adaptivní bilaterální vyhlazení šumu senzoru bez rozmazání jemných struktur korony.
  - ☀️ **Eclipse Coronal Detail Filter**: Zvýraznění jemných magnetických siločar ve vnější i vnitřní koroně se striktní ochranou tmavého nebe.
- **Export**:
  - **16-bit TIFF** (ideální pro další postprocessing v PixInsight, Photoshopu či Lightroomu).
  - **PNG**, **100% JPG** a **32-bit Radiance HDR**.

---

## 🚀 Instalace a spuštění

### 1. Požadavky
- Python 3.10+ (např. Python 3.13)
- Nainstalované knihovny:

```bash
pip install -r requirements.txt
```

### 2. Spuštění aplikace
```bash
python main.py
```
nebo
```bash
py main.py
```

---

## 📸 Jak používat aplikaci

1. **Načtení snímků**:
   - Přetáhněte (Drag & Drop) soubory (např. 9 JPG snímků zatmění) přímo do okna aplikace nebo klikněte na **`+ Přidat fotky`**.
2. **Blesková práce ve výřezu (Fast ROI)**:
   - Klikněte nahoře na **`🎯 Rychlý výřez (ROI 300px)`** a **`☀️ Najít Slunce`** (nebo klikněte myší kamkoliv na Slunce).
   - Složení i veškeré posuvníky reagují **okamžitě bez jakéhokoliv čekání**.
3. **Přepnutí na celý snímek / Export**:
   - Vypnutím tlačítka výřezu se vrátíte na celý snímek.
   - Klikněte na **`💾 Exportovat v plné kvalitě`** a uložte výsledný 16-bit TIFF nebo JPG.
