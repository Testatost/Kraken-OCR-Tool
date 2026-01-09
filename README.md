# Kraken OCR GUI

Eine minimalistische **GUI für Kraken OCR** auf Basis von **PySide6**.  
Das Tool unterstützt Einzelbilder und Batch-Verarbeitung, Overlay-Visualisierung,
Baseline- oder Legacy-Segmentierung sowie verschiedene Exportformate.

---

## Features

- GUI (PySide6) für Kraken OCR
- Einzelbild- und Batch-OCR (Ordner)
- Legacy-Segmentierung (`nlbin + pageseg`)
- Baseline-Segmentierung (`blla`, mit externem Segmentierungsmodell)
- Overlay mit Bounding Boxes und Zeilennummern
- Sprach-Postprocessing (auto / de / en / fr / la)
- Tabellen-/Register-Erkennung (heuristisch)
- Exportformate:
  - TXT
  - CSV
  - JSON
  - ALTO XML
  - hOCR

---

## Voraussetzungen

### System

- Python **3.9 – 3.12**
- Linux oder Windows
- Optional:
  - CUDA-fähige NVIDIA-GPU
  - Apple Silicon (MPS)

---

## Installation der Abhängigkeiten

### Linux (bash)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install kraken Pillow PySide6 torch
```

---

### Windows (PowerShell oder CMD)

```powershell
python -m pip install --upgrade pip
python -m pip install kraken Pillow PySide6 torch
```

---

## Repository klonen

```bash
git clone https://github.com/Testatost/Kraken-OCR-Tool.git
cd Kraken-OCR-Tool
```

---

## Starten

### Linux

```bash
python3 main.py
```

### Windows

```powershell
python main.py
```

---

## Bedienung (Kurzfassung)

1. Bild oder Ordner auswählen
2. Kraken **Recognition-Modell** auswählen
3. Segmentierungsmodus wählen:
   - **Legacy** (kein zusätzliches Modell nötig)
   - **Baseline** (blla, Segmentierungsmodell nötig)
4. Optional:
   - Sprache setzen (bei multilingualen Modellen)
   - Tabellenmodus aktivieren
5. OCR starten
6. Ergebnis exportieren

---

# Kraken OCR-Tool

**Fundquelle für Modelle:**  
https://zenodo.org/communities/ocr_models/records?q&l=list&p=1&s=10&sort=newest

---

## (persönliche) Empfehlungen:

## Baseline-Segmentierungs-Modelle

#### General segmentation model for print and handwriting
Webseite: https://zenodo.org/records/14602569  
Download: https://zenodo.org/records/14602569/files/blla.mlmodel?download=1

#### Kraken segmentation model for two-column prints
Webseite: https://zenodo.org/records/10783346  
Download: https://zenodo.org/records/10783346/files/seg_news_1.0.mlmodel?download=1

---

## Kraken-Erkennungs-Modelle

#### Fraktur model trained from enhanced Austrian Newspapers dataset
Webseite: https://zenodo.org/records/7933402  
Download: https://zenodo.org/records/7933402/files/austriannewspapers.mlmodel?download=1

#### CCATMuS-Print [Large]
Webseite: https://zenodo.org/records/10592716  
Download: https://zenodo.org/records/10592716/files/catmus-print-fondue-large.mlmodel?download=1

#### OCR model for German prints trained from several datasets
Webseite: https://zenodo.org/records/10519596  
Download: https://zenodo.org/records/10519596/files/german_print.mlmodel?download=1

#### HTR model for German manuscripts trained from several datasets
Webseite: https://zenodo.org/records/7933463  
Download: https://zenodo.org/records/7933463/files/german_handwriting.mlmodel?download=1

#### FoNDUE-GD
Webseite: https://zenodo.org/records/14399779  
Download: https://zenodo.org/records/14399779/files/FoNDUE-GD_v2_de.mlmodel?download=1

---

## Hinweise

- Die Sprachwahl wirkt **nur als Post-Processing** (Zeichenfilter),
  nicht als echtes Sprachmodell.
- Tabellen-/Register-Erkennung ist **heuristisch** und nicht perfekt.
- Für große Batch-Jobs wird eine GPU empfohlen.
