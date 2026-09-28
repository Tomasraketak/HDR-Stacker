# 🌞 Astro HDR Stacker — Solar Eclipse & Exposure Fusion Studio

A desktop application in Python (PyQt6 + OpenCV) for stacking exposure brackets into a single
High Dynamic Range image. Built for total solar eclipses — the corona, prominences and the
diamond ring — as well as ordinary landscape HDR.

---

## 🚀 Instalace, aktualizace a spuštění

Potřebujete **Python 3.10 – 3.12** z [python.org](https://www.python.org/downloads/)
(na Windows při instalaci **zaškrtněte „Add Python to PATH“**) a **Git**
z [git-scm.com](https://git-scm.com/downloads).

### Windows (Příkazový řádek i PowerShell)

**Instalace** — jen poprvé:

```bat
git clone https://github.com/Tomasraketak/HDR-Stacker.git
cd HDR-Stacker
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Aktualizace** na nejnovější verzi:

```bat
cd HDR-Stacker
git pull
.venv\Scripts\python -m pip install --upgrade -r requirements.txt
```

**Spuštění:**

```bat
cd HDR-Stacker
.venv\Scripts\python main.py
```

Místo spouštěcího příkazu stačí ve složce programu poklepat na **`run.bat`**. Když ještě
není nic nainstalované, sám vytvoří virtuální prostředí, doinstaluje knihovny a spustí
aplikaci. Při dalších spuštěních už jen spustí aplikaci.

### macOS / Linux

**Instalace** — jen poprvé:

```bash
git clone https://github.com/Tomasraketak/HDR-Stacker.git
cd HDR-Stacker
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Aktualizace:**

```bash
cd HDR-Stacker
git pull
.venv/bin/python -m pip install --upgrade -r requirements.txt
```

**Spuštění:**

```bash
cd HDR-Stacker
.venv/bin/python main.py
```

> **`cd HDR-Stacker`** platí ze složky, kde jste spustili `git clone`. Jinak napište celou
> cestu, např. `cd C:\Users\Jmeno\HDR-Stacker`.
>
> **Bez Gitu:** na GitHubu klikněte na **Code → Download ZIP**, rozbalte a pokračujte od
> příkazu `py -m venv .venv` (resp. `python3 -m venv .venv`). Aktualizace je pak nový ZIP
> rozbalený přes starou složku a znovu příkaz `pip install` z aktualizace.

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

### 6. Retušujte (volitelné) — stébla trávy, ptáci, prach
Když se do záběru připletla stébla trávy, pták nebo smítko na senzoru, klikněte nad
náhledem na **`🩹 Retuš`** a přetřete je štětcem. Přetřené místo se doplní okolní oblohou,
takže to vypadá, jako by tam nikdy nic nebylo:
- **Světlo oblohy** se dopočítá jako „membrána“ napjatá přes okolí. Plynule tak navazuje
  na každý přechod, i na zář nad obzorem, a nevznikne šev.
- **Zrno** (šum) se zkopíruje z čistého místa kousek vedle, aby výplň nebyla podezřele
  hladká.
- Štětec má **měkký okraj**: plný kruh u kurzoru je plný účinek, čárkovaný kruh je místo,
  kde účinek doznívá. Rozostřené stéblo má měkký lem a ten zmizí také.

Ovládání: **levé tlačítko** maluje, **pravé** posouvá pohled, **kolečko** zoomuje,
klávesy **`[`** a **`]`** zmenšují a zvětšují štětec (velikost jde nastavit i v poli
vedle tlačítka). **`Ctrl+Z`** vrátí poslední tah, **`🗑`** smaže celou retuš.

Retuš se ukládá do projektu jako tahy štětcem, ne jako obrázek. Při exportu se proto
provede znovu v plném rozlišení a sedí i po změně ořezu. Funguje nejlépe na obloze
a jiných hladkých plochách. Na krajině nebo městě by výplň rozmazala detaily.

### 7. Uložte si projekt
Přes menu **`Projekt → Uložit projekt`** (`Ctrl+S`) si uložte celé rozpracování do
souboru `.ahdrproj`. Uloží se **všechno**: které fotky jsou načtené, ruční i automatické
zarovnání každého snímku, které snímky jsou vyřazené, ořez, metoda HDR, zarovnání
a všechny posuvníky.

Příště stačí **`Projekt → Otevřít projekt`** (`Ctrl+Shift+O`) a jste přesně tam,
kde jste skončili — žádné znovunastavování.

Program navíc při zavření **automaticky ukládá poslední relaci** a při dalším spuštění
ji sám obnoví. Vypnout to jde v menu **`Projekt → Obnovovat poslední relaci při startu`**,
ručně vyvolat přes **`Obnovit poslední relaci`**.

> **Co projekt obsahuje:** jen cesty k fotkám a vaše nastavení, ne samotné fotografie —
> soubor má pár kilobajtů. Když fotky přesunete, projekt je nenajde; když ale přesunete
> **celou složku i s projektem**, funguje dál, protože se ukládají i relativní cesty.
> Chybějící fotky se jen nahlásí a zbytek projektu se načte.

### 8. Exportujte
**`💾 Exportovat`** (`Ctrl+E`) spočítá výsledek znovu z originálů v plném rozlišení
a uloží ho jako **16bitový TIFF**, **JPEG**, **16bitový PNG** nebo **32bitový Radiance HDR**.
Náhled je záměrně zmenšený kvůli rychlosti — na kvalitu exportu to nemá vliv.

### 9. Časosběrný kompozit zatmění (volitelné)
Klasický snímek, kde nad krajinou visí řada Sluncí a Měsíc je postupně „ukusuje“, až nastane
úplné zatmění, a pak zase odchází. Otevřete ho přes **`🌗 Časosběr zatmění`** v pravém panelu
nebo **`Nástroje → Časosběrný kompozit zatmění`** (`Ctrl+T`).

1. **Pozadí** — načtěte už složenou (stacknutou) fotku úplné fáze, třeba 16bitový TIFF
   exportovaný tímto programem. Program sám najde disk Měsíce v koróně (střed i průměr).
   Čas snímku se čte z EXIF; stack z tohoto programu EXIF nemá, takže klikněte na
   **`🕑 Převzít čas z EXIF jiné fotky`** a vyberte jednu z původních expozic úplné fáze.
   Nastavte **časové pásmo, ve kterém má fotoaparát nastavené hodiny** (letní čas = UTC+2;
   fotoaparát nastavený na zimní čas = UTC+1) a místo — z GPS v EXIF, z nabídky měst na pásu
   totality 12. 8. 2026, nebo souřadnicemi.
   - **Hodiny fotoaparátu šly napřed nebo pozadu?** Klikněte na **`⏱ Seřídit…`**, vyberte
     snímek, u kterého znáte přesný čas, a zadejte ho. Typicky je to fotka začátku úplné fáze,
     jejíž čas (2. kontakt) najdete v tabulce místních okolností zatmění. Například fotka má
     v EXIF 19:33:08, ale úplná fáze tam začala v 19:28:25, obojí v UTC+1. Program spočítá
     **Korekci hodin −283 s** („hodiny šly 4 min 43 s napřed“) a použije ji pro pozadí i
     všechny srpky. Korekci lze zadat i přímo v sekundách.
   - Pokud srpky fotil **jiný přístroj** než pozadí, nastavte rozdíl jeho hodin v poli
     **Srpky navíc** (část 3). Přičte se jen k srpkům.
2. **Kalibrace** — klikněte na **`〰 Vyznačit horizont`** a táhněte myší podél vzdáleného
   obzoru (co nejdelší úsečka). Slunce je předvyplněné; když ne, **`☀ Vyznačit Slunce`**:
   klik do středu disku a tah k okraji. Z toho program spočítá model fotoaparátu — měřítko
   (px na stupeň), náklon i směr — a ukáže, jak dobře průměr Slunce a horizont souhlasí.
   Pokud jste fotili z kopce, nastavte **Výšku horizontu** do minusu (100 m ≈ −0,3°).
3. **Částečné fáze** — **`+ Přidat snímky s filtrem`**. U každé fotky se najde sluneční
   disk, vyřízne se a podle **času z EXIF** se umístí přesně tam, kde v tu chvíli Slunce
   na obloze bylo. Fotky částečných fází nemusí mít stejný záběr jako pozadí. Černé snímky
   bez Slunce se samy vypnou.
   - Najde se i **malé Slunce na širokém záběru a tenký srpek** těsně před úplnou fází
     nebo po ní (poloměr ~10 px, tloušťka 1–2 px). Kružnice se proloží vnějším okrajem Slunce.
   - Snímky ze **stejného objektivu a zoomu** dostanou **společný poloměr Slunce**, změřený
     na snímcích, kde je vidět aspoň půlka okraje. U tenkého srpku se pak dopočítá jen
     střed. Díky tomu mají všechna Slunce v kompozitu stejnou velikost. Snímky s jiným
     zoomem (podle ohniska v EXIF nebo podle velikosti Slunce) se nemíchají.
   - U přeexponovaného Slunce se do okraje nezapočítává záře kolem něj.
4. **Vzhled** — jas povrchu Slunce se u každé fotky **automaticky vyrovná** na společnou
   hodnotu. Dál lze volit barvu (původní / sjednocená / neutrální / zlatavá), prolnutí
   (Měsíc průhledný nebo černý), velikost Sluncí a zda zapadající Slunce schovat za obzor.
5. **Barvy a tóny** — **jas** (EV), **kontrast**, **střední tóny**, **saturace**,
   **teplota** a **odstín** se nastavují na třech záložkách:
   - **Všechna Slunce** (master) — jedním pohybem pro všechny srpky najednou;
   - **Vybraný snímek** — jen pro jedno Slunce, **přičítá se** k masteru (master teplota
     +30 a u snímku −10 dá u toho snímku +20); tady se dá i vypnout automatické vyrovnání jasu;
   - **Pozadí** — snímek úplné fáze (korona, obloha, krajina).

   Kontrast u Slunce pracuje kolem jasu jeho povrchu: přidáním se prohloubí okrajové
   ztemnění a skvrny, ubráním se disk zploští. Černá zůstává černá, takže Měsíc
   „ukusující“ Slunce nikdy nezešedne. Dvojklik na posuvník ho vrátí na 0,
   **`↺ Vynulovat úpravy`** vynuluje celou záložku.
6. **Retuš pozadí** — stébla trávy nebo ptáky na obloze pozadí přetřete štětcem
   (**`🩹 Retušovat štětcem`**), stejně jako v hlavním okně (viz bod 6 výše). Slunce se
   kreslí až na retušované pozadí, takže štětec žádné nesmaže.
7. **Kontrola a doladění** — přes fotku se kreslí **denní dráha Slunce** s časovými
   značkami, volitelně i **ekliptika**, vyznačený a vypočtený horizont a značky snímků.
   Každé Slunce jde **přetáhnout myší** nebo posunout šipkami (Shift = 5 px, Ctrl = 0,2 px);
   **`↺ Vrátit na vypočtenou polohu`** ruční posun zruší.
8. **`💾 Exportovat kompozit`** vykreslí výsledek v plném rozlišení (TIFF 16 bit, PNG, JPEG).

Celé nastavení kompozitu se ukládá do projektu `.ahdrproj` spolu s HDR skládáním.

> **Přesnost:** poloha se počítá z efemeridy Slunce (algoritmus Meeus/NOAA, chyba ~0,01°)
> včetně atmosférické refrakce a ukotvuje se na Slunce v pozadí. Na přesném čase ale záleží
> i u snímků ze stejného fotoaparátu: výška Slunce nad vyznačeným horizontem se počítá
> z absolutního času. Hodiny o 4 min 43 s napřed posunuly na testovací scéně Slunce až
> o 36 px, proto je seřiďte (**`⏱ Seřídit…`**). Slunce se za 2 minuty posune o svůj průměr.

### Klávesové zkratky

| Zkratka | Akce |
|---|---|
| `Ctrl+N` | Nový projekt |
| `Ctrl+S` / `Ctrl+Shift+S` | Uložit projekt / uložit jako |
| `Ctrl+Shift+O` | Otevřít projekt |
| `Ctrl+O` | Přidat fotky |
| `Ctrl+R` | Složit snímky |
| `Ctrl+E` | Exportovat v plné kvalitě |
| `Ctrl+M` | Ruční dozarovnání |
| `Ctrl+T` | Časosběrný kompozit zatmění |
| `Ctrl+0` / `Ctrl+1` | Přizpůsobit oknu / zobrazit 1:1 |
| `Ctrl+Z` | Vrátit poslední tah retuše |
| `[` / `]` | Zmenšit / zvětšit štětec retuše |
| `Ctrl+Q` | Konec |
| Kolečko myši | Zoom · dvojklik = přizpůsobit |

---

## 🛠 Řešení potíží

| Problém | Řešení |
|---|---|
| Aplikace je pomalá nebo se zasekává | Nastavte **Pracovní rychlost** na `🚀 1/8 rozlišení` a používejte režim **🎯 ROI**. |
| Hlásí nedostatek paměti při exportu | Program sám nabídne export ve zmenšeném rozlišení — potvrďte **Ano**. Pomůže i zavření prohlížeče. |
| „Všechny snímky mají prakticky stejný expoziční čas“ | Snímky nemají použitelné EXIF časy. Přepněte **Metodu HDR** na **Mertens**, která časy nepotřebuje. |
| Disk Měsíce se nenajde | Použijte **🛠️ Ruční dozarovnání** a posuňte snímky ručně podle rozdílového náhledu. |
| Projekt nenajde fotky | Fotky byly přesunuty. Přesouvejte vždy celou složku i s projektem — pak fungují relativní cesty. |
| Export se nepodaří zapsat | Zkontrolujte, že soubor není otevřený v jiném programu a že do složky lze zapisovat. |
| Chyba při instalaci PyQt6 | Použijte Python 3.12 místo 3.13+. |
| Kompozit hlásí „Slunce je pod horizontem“ | Zkontrolujte čas pozadí, časové pásmo (letní čas = UTC+2) a polohu. |
| Průměr Slunce a horizont si odporují | Často jde o špatný čas: zkontrolujte časové pásmo fotoaparátu (UTC+1 / UTC+2) a **Korekci hodin**. Dál výšku horizontu (z kopce je obzor níž); v nouzi zvolte **Měřítko z: Jen průměr Slunce**. |
| Srpek sedí vedle dráhy | Nejspíš nesedí čas snímku — seřiďte hodiny (**`⏱ Seřídit…`** u pozadí), případně upravte čas u vybraného snímku nebo pole **Srpky navíc**. |
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
- **Eclipse sequence composite** — drop filtered partial-phase frames into a stacked
  totality background. The background is calibrated as a pinhole camera from the marked
  horizon, Sun and solar diameter (weighted least squares), and every Sun is placed from its
  EXIF timestamp through a built-in solar ephemeris (Meeus, with refraction) — on synthetic
  data within 0.05 px over a 70-minute sequence. Crescents are fitted on the solar limb only
  (sub-pixel tracing for Suns ~10 px in radius; thin crescents take the sequence's common
  solar radius, clustered per lens/zoom), surface brightness is equalised automatically, and exposure, contrast, mid-tones,
  saturation, temperature and tint can be graded for all Suns at once, per frame on top of
  that, and for the background. The tone curves pin black, so the Moon never turns grey.
  The Sun's daily path, time ticks and the ecliptic can be overlaid for checking.
- **Retouch brush** — paint away grass blades, birds or dust from the sky, in the HDR
  result and in the composite's background. The painted area is refilled with a harmonic
  (membrane) fill that continues every sky gradient without a seam, plus grain copied from
  a clean patch nearby; the brush has a soft edge that also removes an out-of-focus
  object's halo. Strokes are stored as vector paths, so the export re-heals them at full
  resolution; the preview heals only the new stroke, and undo restores the saved pixels.
- **Project files** — save the whole session (frames, per-frame alignment, exclusions,
  crop, every setting) to a small JSON `.ahdrproj` and reopen it exactly as it was.
  Paths are stored both absolutely and relative to the project, so moving a folder with
  its project intact keeps working; missing photos are reported and skipped rather than
  aborting the load. The session is also snapshotted automatically on exit.

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
The eclipse composite is checked against the Meeus reference examples and against a
synthetic sky photographed by a known camera: disc detection on fat, thin and red
crescents (also tiny ones in a 24 Mpx frame, and overexposed ones with a glow), one solar
radius across a sequence, camera recovery, placement accuracy, brightness equalisation, the master /
per-frame / background colour grades, horizon clipping, project round-trips and the editor
itself. The retouch brush is checked on a noisy sky behind a blurred grass blade (no ghost
left, matching grain, untouched pixels beyond the brush, preview equal to export, exact
undo) and end to end in both windows, including the full-resolution export.

## 📋 Requirements

- Python 3.10 – 3.12
- PyQt6, OpenCV, NumPy, Pillow (see `requirements.txt`)
