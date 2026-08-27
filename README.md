# 🌞 Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio

A desktop application in Python (PyQt6 + OpenCV) for stacking exposure brackets into a single
High Dynamic Range image. Built for total solar eclipses — the corona, prominences and the
diamond ring — as well as ordinary landscape HDR.

---

## 🚀 Rychlý start (Windows 10 / 11)

**Nejrychlejší cesta — stačí dvakrát kliknout:**

1. Nainstalujte **Python 3.10 – 3.12** z [python.org](https://www.python.org/downloads/windows/).
   Při instalaci **zaškrtněte „Add Python to PATH“**.
2. Ve složce s programem poklepejte na **`run.bat`**.

`run.bat` při prvním spuštění sám vytvoří virtuální prostředí, doinstaluje knihovny
a spustí aplikaci. Při dalších spuštěních už jen spustí aplikaci (během pár sekund).

**Ruční cesta (PowerShell nebo Příkazový řádek):**

```powershell
cd C:\cesta\k\HDR-Stacker
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**macOS / Linux:**

```bash
cd /cesta/k/HDR-Stacker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

> **Poznámka k Pythonu 3.13+:** PyQt6 a OpenCV pro něj nemusí mít připravené instalační
> balíčky. Pokud instalace skončí chybou o „building wheel“, použijte Python 3.12.

---

## 📖 Jak program používat

### 1. Načtěte expoziční řadu
Přetáhněte fotky (např. 9 JPEG snímků zatmění) přímo do okna aplikace, nebo klikněte na
**`+ Přidat fotky`** (`Ctrl+O`). Program sám přečte časy závěrky z EXIF, seřadí snímky od
nejtmavšího po nejsvětlejší a spočítá EV. Když EXIF chybí, seřadí je podle jasu scény
a EV odhadne z kroku, který nastavíte v poli **Krok expozice**.

Snímek můžete kdykoliv vyřadit odškrtnutím políčka v prvním sloupci seznamu.

### 2. Zarovnejte snímky
- **`💡 Podle statických světel`** (výchozí) je nejpřesnější volba, když máte v záběru
  krajinu. Najde vzor pouličních lamp a vzdálených světel v dolní části snímku. Ta se
  nehýbou, takže měří **přímo otřesy fotoaparátu**. Na testovacích datech dosahuje
  přesnosti kolem **0,08 px** proti 1,5 px u zarovnání podle disku.
  Lampy se párují jako souhvězdí, ne podle jasu pixelů, takže to funguje napříč celou
  expoziční řadou. Snímky, které se nepodaří spolehlivě zarovnat, program **nechá být
  a nahlásí je** ve stavovém řádku — ty pak doladíte ručně.
- **`🌑 Detekce černého disku Měsíce`** najde kruhový disk Měsíce v záři korony.
  Použijte, když je v záběru jen obloha, nebo když se Slunce mezi snímky posunulo.
- **`🚫 Bez zarovnání`** použijte, pokud jste fotili z pevného stativu.

> **Pozor na rozdíl:** zarovnání podle lamp srovná **krajinu**, zarovnání podle disku
> srovná **korónu**. Během delší série se Slunce po obloze posune, takže obojí naráz
> nejde — vyberte si, co má ve výsledku sedět.
- **`🛠️ Ruční dozarovnání`** (`Ctrl+M`) otevře okno pro doladění snímek po snímku:
  - Režim **Rozdíl hran** — nesedící hrany svítí barevně, sedící zmizí. Nejpřesnější.
  - Režim **Blend** a **Blikání** pro rychlou kontrolu.
  - Posun šipkami na klávesnici; **Shift** = hrubý krok 5 px, **Ctrl** = jemný krok 0,2 px.
  - Tlačítko **`🌑 Najít černý disk Měsíce`** předvyplní posuny automaticky.
  - **Zrušit** vrátí všechny posuny do stavu před otevřením okna.

### 3. Pracujte v režimu výřezu (nejrychlejší způsob editace)
Klikněte na **`🎯 ROI`** v horní liště a pak na **`☀️ Najít`** — nebo prostě klikněte
myší do fotky tam, kde je Slunce. Skládá se jen výřez kolem koróny (300×300 až 1200×1200 px),
takže každá změna posuvníku je vidět prakticky okamžitě. Výřez můžete kdykoliv přetáhnout myší.

### 4. Vylaďte výsledek
V panelu vpravo:
- **Metoda HDR** — pro zatmění nechte **Mertens Exposure Fusion**. Nepotřebuje znát
  expoziční časy a dává nejčistší korónu. **Debevec** a **Robertson** počítají skutečnou
  32bitovou mapu jasu, ale vyžadují správné časy závěrky z EXIF.
- **Předvolby** (Vnitřní korona, Vnější korona, Diamantový prsten, Krajina se zatměním)
  nastaví všechny posuvníky najednou jako rozumný výchozí bod.
- **Detaily korony** zvýrazní jemné struktury magnetického pole. Tmavá obloha je přitom
  chráněná, takže se nezvýrazňuje šum.
- Dvojklikem na jakýkoliv posuvník ho vrátíte na výchozí hodnotu.

### 5. Ořízněte (volitelné)
V panelu vpravo zaškrtněte **`Oříznout výsledek`**. Pak buď klikněte na
**`🖱️ Vybrat oblast myší`** a táhněte přes snímek, nebo zadejte X, Y, šířku a výšku
číselně. Během výběru se zobrazí celý snímek s oranžovým rámečkem a vodítky třetin;
jakmile výběr dokončíte, uvidíte rovnou oříznutý výsledek.

Ořez se použije **stejně na všechny expozice** — v náhledu i při exportu. Zarovnání
proběhne ještě před ořezem, takže se u okrajů neztrácí obrazová data.

### 6. Exportujte
**`💾 Exportovat`** (`Ctrl+S`) spočítá výsledek znovu z originálů v plném rozlišení
a uloží ho jako **16bitový TIFF**, **JPEG**, **16bitový PNG** nebo **32bitový Radiance HDR**.
Náhled je záměrně zmenšený kvůli rychlosti — na kvalitu exportu to nemá vliv.

### Klávesové zkratky

| Zkratka | Akce |
|---|---|
| `Ctrl+O` | Přidat fotky |
| `Ctrl+R` | Složit snímky |
| `Ctrl+S` | Exportovat v plné kvalitě |
| `Ctrl+M` | Ruční dozarovnání |
| `Ctrl+0` / `Ctrl+1` | Přizpůsobit oknu / zobrazit 1:1 |
| Kolečko myši | Zoom · dvojklik = přizpůsobit |

---

## 🛠 Řešení potíží

| Problém | Řešení |
|---|---|
| Aplikace je pomalá nebo se zasekává | Nastavte **Pracovní rychlost** na `🚀 1/8 rozlišení` a používejte režim **🎯 ROI**. |
| Hlásí nedostatek paměti při exportu | Program sám nabídne export ve zmenšeném rozlišení — potvrďte **Ano**. Pomůže i zavření prohlížeče. |
| „Všechny snímky mají prakticky stejný expoziční čas“ | Snímky nemají použitelné EXIF časy. Přepněte **Metodu HDR** na **Mertens**, která časy nepotřebuje. |
| Disk Měsíce se nenajde | Použijte **🛠️ Ruční dozarovnání** a posuňte snímky ručně podle rozdílového náhledu. |
| Export se nepodaří zapsat | Zkontrolujte, že soubor není otevřený v jiném programu a že do složky lze zapisovat. |
| Chyba při instalaci PyQt6 | Použijte Python 3.12 místo 3.13+. |
| Okno se nevejde na obrazovku | Okna se sama zmenší podle plochy monitoru. Ovládací panely lze rolovat, takže tlačítka dole zůstanou vždy dosažitelná. |

Pokud dojde k neočekávané chybě, aplikace ji zobrazí v dialogu (včetně technického
výpisu pod tlačítkem *Show Details*) a **běží dál** — rozpracovaná práce se neztratí.

---

## ✨ Key features

- **Fast ROI crop mode** — toggle between the full frame and a 300×300 … 1200×1200 px crop
  around the Sun for near-instant editing. Click anywhere in the image to recentre it.
  Exports always run at full resolution regardless of the preview crop.
- **Automatic EV and exposure detection** — shutter speed, ISO and aperture from EXIF, with
  a histogram-based fallback that sorts frames from the shortest (-EV) to the longest (+EV)
  exposure. Manual shifts and exclusions are preserved across re-sorts.
- **Lunar disc detection and alignment** — finds the circular black moon disc inside the
  coronal glow using a circularity filter plus a bright-halo check, refined to subpixel
  accuracy at full resolution. Detection runs on a bounded proxy, so it is fast on 45 MP frames.
- **Interactive frame-by-frame manual alignment** — edge difference, 50 % blend and flicker
  modes; frames load on a background thread; Cancel restores the original shifts.
- **Fusion engines** — Mertens exposure fusion (recommended for eclipses), Debevec and
  Robertson 32-bit HDR with CRF calibration and Reinhard / Drago / Mantiuk tonemapping.
- **Noise reduction and coronal detail filter** — edge-preserving bilateral denoising and
  multi-scale coronal enhancement that is gated off in the dark sky, so grain is never sharpened.
- **Export** — 16-bit TIFF, 16-bit PNG, quality JPEG and 32-bit Radiance HDR, with
  Unicode-safe file writing.

## 🧠 Stability and performance notes

The application is built to stay alive on a mid-range laptop:

- **Memory-bounded fusion.** Full-resolution stacks above 12 Mpx are fused in overlapping
  horizontal bands. On a 9 × 24 Mpx bracket this cuts peak RSS from **6.0 GB to 2.5 GB**
  at the same speed, and the result matches single-pass fusion to within 3 × 10⁻⁷.
- **Memory-aware export.** Before a full-resolution export the app estimates the requirement
  against actual free RAM and offers a safe reduced-scale export rather than being killed.
- **Debounced re-stacking.** Dragging the ROI used to spawn one background thread per mouse
  move, each decoding the whole bracket from disk. Requests are now coalesced.
- **Shared decoded-image cache.** Bounded to a fraction of free RAM, so a frame is decoded
  at most once per working scale.
- **Safe thread lifecycle.** Background workers are never terminated mid-allocation and
  never dropped while running — both are reliable ways to corrupt the heap and crash Qt.
- **Non-finite values are scrubbed** at every stage, so a zero or duplicated exposure time
  produces a clear message instead of a NaN image.
- **A global exception hook** turns any unforeseen error into a dialog instead of the
  silent process abort that PyQt6 performs by default.

## 🧪 Tests

```bash
python tests/test_stacker.py
```

Covers the numerical core (disc detection, alignment, all three fusion engines, tonemapping,
post-processing, every export format including Unicode paths) plus GUI stability scenarios:
rapid ROI dragging, repeated worker cancellation, missing files, dialog cancel semantics,
an end-to-end full-resolution export, and closing the window with work in flight.

## 📋 Requirements

- Python 3.10 – 3.12
- PyQt6, OpenCV, NumPy, Pillow (see `requirements.txt`)
