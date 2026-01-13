# KrakenOCR-Tool

KrakenOCR-Tool ist ein grafisches OCR-Tool auf Basis von **Kraken** und **PySide6** mit interaktiver Zeilenanzeige, Bounding-Box-Overlays und umfangreichen Exportfunktionen. Es richtet sich insbesondere an Digital-Humanities-Projekte, Archivarbeit und alle Anwendungsfälle, in denen strukturierte und nachvollziehbare OCR-Ergebnisse benötigt werden.

## Features

- OCR mit Kraken (Legacy- und Baseline-Segmentierung)
- Interaktive Anzeige erkannter Zeilen
- Klickbare Bounding-Boxen mit Zeilennummern
- Mehrsprachige Post-Processing-Filter (heuristisch)
- Heuristische Tabellen- und Register-Erkennung
- Batch-Verarbeitung ganzer Ordner
- Unterstützte Exportformate:
  - Plain Text (.txt)
  - CSV / JSON (optional tabellenbasiert)
  - ALTO XML
  - hOCR
  - Bilder (.png, .jpg, .bmp) inkl. Overlays
  - Durchsuchbares PDF (Bild + unsichtbarer Textlayer)
- CPU-, CUDA- und experimentelle MPS-Unterstützung

## Voraussetzungen

### Betriebssystem

- Windows 10 / 11
- Linux (getestet mit Fedora)
- macOS (eingeschränkt, ohne CUDA)

### Python

- Empfohlen: Python 3.10 oder 3.11
- Python 3.13 kann funktionieren, ist aber nicht garantiert

    python --version

## Installation

### Repository klonen

    git clone https://github.com/<dein-user>/KrakenOCR-Tool.git
    cd KrakenOCR-Tool

### Virtuelle Umgebung erstellen

    python -m venv .venv

Linux / macOS:

    source .venv/bin/activate

Windows (PowerShell):

    .venv\Scripts\Activate.ps1

### Abhängigkeiten installieren

    pip install --upgrade pip
    pip install pillow pyside6 reportlab torch kraken

Hinweis: Für GPU-Beschleunigung muss `torch` passend zur installierten CUDA-Version installiert werden. Siehe https://pytorch.org/get-started/locally/

## Starten

    python main.py

Nach dem Start öffnet sich die grafische Oberfläche.

## Kraken-Modelle

Das Tool benötigt mindestens ein Kraken-Recognition-Modell (z. B. `.mlmodel`, `.pt`, `.pth`). Optional kann ein Baseline-Segmentierungsmodell für `blla` verwendet werden. Die Modelle werden nicht mitgeliefert und müssen im Programm manuell ausgewählt werden.

## Exportformate

- Text: .txt
- Tabellen: .csv, .json
- OCR-Formate: ALTO XML, hOCR
- Bilder: .png, .jpg, .bmp
- PDF: durchsuchbares PDF mit unsichtbarem Textlayer

## Build als Windows-EXE (optional)

### Voraussetzungen

- Windows
- Python (gleiche Version wie beim Entwickeln)
- PyInstaller

    pip install pyinstaller

### Build mit Spec-Datei

    pyinstaller KrakenOCR-Tool.spec --clean

Die erzeugte EXE befindet sich anschließend unter:

    dist/KrakenOCR-Tool/
