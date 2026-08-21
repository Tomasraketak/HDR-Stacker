# 🌘 Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio

Profesionální desktopová aplikace v Pythonu (PyQt6 + OpenCV) pro skládání expozičních řad fotografií do jednoho HDR snímku s vysokým dynamickým rozsahem. Navržena speciálně pro náročné astrofotografické scény jako je **úplné zatmění Slunce**, korona, protuberance i pro klasické pozemské HDR fotografie.

---

## ✨ Klíčové funkce

- **Automatické rozpoznání EV a expozičních časů**:
  - Automatické vyčtení expozičních časů, ISO a clony z **EXIF** metadat.
  - Inteligentní analýza jasu scény: Pokud EXIF chybí, aplikace analyzuje histogram a seřadí snímky od nejkratší expozice (-EV) po nejdelší (+EV).
  - Možnost definovat krok (např. **9 snímků po 1.0 EV**).
- **Pokročilé metody zarovnání (Multi-Algorithm Alignment)**:
  - 🌑 **Astronomické zarovnání disku zatmění (Subpixel Eclipse Disc)**: Detekuje subpixelový střed siluety Měsíce/Slunce. Ideální pro zatmění Slunce, kde se jas jednotlivých expozic liší o několik řádů.
  - ⚡ **ECC (Enhanced Correlation Coefficient)**: Subpixelová afinní/euklidovská optimalizace posunu a rotace.
  - 🔍 **ORB & RANSAC**: Detekce klíčových bodů s robustním filtrováním.
  - 📐 **MTB (Median Threshold Bitmap)**: Klasické prahové zarovnání.
  - 🚫 **Bez zarovnání**: Pro snímky pořízené na přesné paralaktické montáži.
- **Skládací algoritmy (HDR & Fusion Engines)**:
  - **Mertens Exposure Fusion** (*Doporučeno pro zatmění Slunce*): Laplaceova pyramidová bezešvá fúze s potlačením šumu v podexponovaných oblastech.
  - **Debevec 32-bit HDR** s kalibrací křivky odezvy snímače (CRF) a tonemappingem (Reinhard, Drago, Mantiuk).
  - **Robertson 32-bit HDR**.
- **Potlačení šumu & Filtr sluneční korony**:
  - 🛡️ **Redukce šumu (Grain filter)**: Adaptivní bilaterální vyhlazení šumu senzoru bez rozmazání jemných struktur korony.
  - ☀️ **Eclipse Coronal Detail Filter**: Zvýraznění jemných magnetických siločar ve vnější i vnitřní koroně se striktní ochranou tmavého nebe (šum na pozadí není nikdy doostřován).
- **Bleskový živý náhled (Zero-Latency Live Preview)**:
  - Vektorizované a 1D LUT zpracování barev, jasu, kontrastu, saturace a gamma křivek. Posuvníky reagují okamžitě (60 FPS) i na fotografiích ve vysokém rozlišení.
- **Interaktivní GUI (PyQt6)**:
  - Moderní tmavý astro motiv.
  - Plynulý zoom (kolečko myši), posun myší (pan), zobrazení 100% (1:1 pixel) a režim rozděleného srovnání (Split Before/After).
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
2. **Kontrola / nastavení EV a zarovnání**:
   - Aplikace automaticky seřadí snímky a přiřadí jim EV.
   - Vyberte metodu zarovnání (výchozí: **Zatmění - subpixel střed**).
3. **Složení expozic**:
   - Klikněte na **`⚡ Složit snímky (HDR Merge)`**.
4. **Okamžité doladění detailů a barev**:
   - Upravte sytost, kontrast, jas, redukci šumu a zvýraznění korony posuvníky vpravo s okamžitou odezvou.
5. **Export**:
   - Klikněte na **`💾 Exportovat výsledek`** a uložte výsledný snímek jako 16-bit TIFF nebo JPG.

---

## 📂 Struktura projektu

```
solar_hdr_stacker/
├── core/
│   ├── exif_and_analysis.py  # EXIF čtení a analýza jasu
│   ├── aligner.py            # Eclipse disc, ECC, ORB, MTB zarovnání
│   ├── merger.py             # Mertens, Debevec, Robertson fúze
│   └── postprocess.py        # LUT postprocess, denoise & coronal enhancer
├── gui/
│   ├── main_window.py        # Hlavní okno a realtime rendering
│   ├── image_viewer.py       # Zoomovatelný prohlížeč & srovnávač
│   ├── exposure_list_widget.py # Seznam a správa expozic
│   ├── controls_panel.py     # Ovládací panel posuvníků
│   └── styles.py             # Tmavý astronomický motiv
├── main.py                   # Spouštěcí bod aplikace
├── requirements.txt
├── .gitignore
└── README.md
```
