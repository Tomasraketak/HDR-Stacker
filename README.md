# 🌘 Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio

Profesionální desktopová aplikace v Pythonu (PyQt6 + OpenCV) pro skládání expozičních řad fotografií do jednoho HDR snímku s vysokým dynamickým rozsahem. Navržena speciálně pro náročné astrofotografické scény jako je **úplné zatmění Slunce**, korona, protuberance i pro klasické pozemské HDR fotografie.

---

## ✨ Klíčové funkce

- **Automatické rozpoznání EV a expozičních časů**:
  - Automatické vyčtení expozičních časů, ISO a clony z **EXIF** metadat.
  - Inteligentní analýza jasu scény: Pokud EXIF chybí, aplikace analyzuje histogram a seřadí snímky od nejkratší expozice (-EV) po nejdelší (+EV).
  - Možnost definovat krok (např. **9 snímků po 1.0 EV**).
- **Zarovnání snímků (Alignment)**:
  - **MTB (Median Threshold Bitmap)**: Algoritmus navržený pro zarovnávání snímků s různým jasem bez ovlivnění expozice.
- **Skládací algoritmy (HDR & Fusion Engines)**:
  - **Mertens Exposure Fusion** (*Doporučeno pro zatmění Slunce*): Laplaceova pyramidová bezešvá fúze. Nevyžaduje kalibraci snímače a nevytváří nepřirozené HDR mapovací haló efekty.
  - **Debevec 32-bit HDR** s kalibrací křivky odezvy snímače (CRF) a tonemappingem (Reinhard, Drago, Mantiuk).
  - **Robertson 32-bit HDR**.
- **Eclipse Coronal Detail Filter (Filtr sluneční korony)**:
  - Víceúrovňový filtr detailů a vysokých frekvencí s ochranou přepalů a šumu na pozadí, který vytáhne jemné magnetické siločáry a paprsky ve vnější i vnitřní koroně.
- **Interaktivní GUI (PyQt6)**:
  - Moderní tmavý astro motiv.
  - Plynulý zoom (kolečko myši), posun myší (pan), zobrazení 100% (1:1 pixel) a režim rozděleného srovnání (Split Before/After).
  - Živé posuvníky pro úpravu jasu, kontrastu, gamma, saturace, stínů a světel.
  - Asynchronní zpracování na pozadí (GUI nikdy nezamrzá).
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
2. **Kontrola / nastavení EV**:
   - Aplikace automaticky seřadí snímky a přiřadí jim EV.
   - V případě potřeby můžete upravit krok v poli *Krok expozice* (např. 1.0 EV) a kliknout na **`🔄 Seřadit a spočítat EV`**.
3. **Složení expozic**:
   - Klikněte na **`⚡ Složit snímky (HDR Merge)`**.
4. **Doladění detailů korony a barev**:
   - Upravte posuvník **`Zvýraznění detailů`** (sluneční korona) a poloměr filtru.
   - Dolaďte jas, kontrast, gamma a stíny v pravém panelu s okamžitým náhledem.
5. **Export**:
   - Klikněte na **`💾 Exportovat výsledek`** a uložte výsledný snímek jako 16-bit TIFF nebo JPG.

---

## 📂 Struktura projektu

```
solar_hdr_stacker/
├── core/
│   ├── exif_and_analysis.py  # EXIF čtení a analýza jasu
│   ├── aligner.py            # MTB zarovnání snímků
│   ├── merger.py             # Mertens, Debevec, Robertson fúze
│   └── postprocess.py        # Coronal detail enhancer & export
├── gui/
│   ├── main_window.py        # Hlavní okno a threading
│   ├── image_viewer.py       # Zoomovatelný prohlížeč & srovnávač
│   ├── exposure_list_widget.py # Seznam a správa expozic
│   ├── controls_panel.py     # Ovládací panel posuvníků
│   └── styles.py             # Tmavý astronomický motiv
├── main.py                   # Spouštěcí bod aplikace
├── requirements.txt
├── .gitignore
└── README.md
```
