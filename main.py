import os
import sys
import math
import statistics
import json
import csv
import warnings
import re
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple, Dict, Callable

# GUI Framework
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QUrl, QTimer, QSize, QPointF, QPoint, QDateTime, QLocale
from PySide6.QtGui import (
    QPixmap, QPen, QBrush, QColor, QFont, QDragEnterEvent, QDropEvent, QAction,
    QKeySequence, QActionGroup, QIcon, QPalette
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QLabel, QPushButton, QProgressBar, QVBoxLayout,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsSimpleTextItem, QSplitter, QStatusBar,
    QMenu, QTableWidget, QTableWidgetItem, QHeaderView, QToolBar,
    QAbstractItemView, QInputDialog, QDialog, QDialogButtonBox, QRadioButton,
    QListWidget as QListWidget2, QSpinBox, QFormLayout, QPlainTextEdit, QToolButton,
)

# PySide object validity helper
from shiboken6 import isValid

# Image & PDF
from PIL import Image
from PIL.ImageQt import ImageQt
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

# Kraken & ML
warnings.filterwarnings("ignore", message="Using legacy polygon extractor*", category=UserWarning)
from kraken import blla, rpred, serialization, pageseg
from kraken.lib import models, vgsl
import torch


# -----------------------------
# CONSTANTS
# -----------------------------
READING_MODES = {
    "TB_LR": 0,
    "TB_RL": 1,
    "BT_LR": 2,
    "BT_RL": 3,
}

STATUS_WAITING = 0
STATUS_PROCESSING = 1
STATUS_DONE = 2
STATUS_ERROR = 3

STATUS_ICONS = {
    STATUS_WAITING: "⏳",
    STATUS_PROCESSING: "⚙️",
    STATUS_DONE: "✅",
    STATUS_ERROR: "❌"
}

THEMES = {
    "bright": {
        "bg": "#f0f0f0",
        "fg": "#000000",
        "canvas_bg": "#ffffff",
        "table_base": QColor(240, 240, 240),
        "toolbar_text": "#000000",
        "toolbar_border": "#000000",
    },
    "dark": {
        "bg": "#2b2b2b",
        "fg": "#ffffff",
        "canvas_bg": "#1e1e1e",
        "table_base": QColor(43, 43, 43),
        "toolbar_text": "#ffffff",
        "toolbar_border": "#ffffff",
    }
}

ZENODO_URL = "https://zenodo.org/communities/ocr_models/records?q=&l=list&p=1&s=10&sort=mostdownloaded"

# -----------------------------
# TRANSLATIONS
# -----------------------------
TRANSLATIONS = {
    "de": {
        "app_title": "Kraken OCR Professional",
        "toolbar_main": "Werkzeugleiste",
        "menu_file": "&Datei",
        "menu_edit": "&Bearbeiten",
        "menu_export": "Exportieren als...",
        "menu_exit": "Beenden",
        "menu_models": "&Modelle",
        "menu_options": "&Optionen",
        "menu_languages": "Sprachen",
        "menu_hw": "CPU/GPU",
        "menu_reading": "Leserichtung",
        "menu_appearance": "Erscheinungsbild",

        "log_toggle_show": "Log",
        "log_toggle_hide": "Log",
        "menu_export_log": "Log als .txt exportieren...",
        "dlg_save_log": "Log speichern",
        "dlg_filter_txt": "Text (*.txt)",
        "log_started": "Programm gestartet.",
        "log_queue_cleared": "Queue geleert.",

        "lang_de": "Deutsch",
        "lang_en": "English",
        "lang_fr": "Français",

        "hw_cpu": "CPU",
        "hw_cuda": "GPU – CUDA (NVIDIA)",
        "hw_rocm": "GPU – ROCm (AMD)",
        "hw_mps": "GPU – MPS (Apple)",

        "act_undo": "Rückgängig",
        "act_redo": "Wiederholen",

        "msg_hw_not_available": "Diese Hardware ist auf diesem System nicht verfügbar. Wechsle zu CPU.",
        "msg_using_device": "Verwende Gerät: {}",
        "msg_detected_gpu": "Erkannt: {}",
        "msg_device_cpu": "CPU",
        "msg_device_cuda": "CUDA",
        "msg_device_rocm": "ROCm",
        "msg_device_mps": "MPS",

        "act_add_files": "Dateien laden...",
        "act_download_model": "Modell herunterladen (Zenodo)",
        "act_delete": "Löschen",
        "act_rename": "Umbenennen...",
        "act_clear_queue": "Queue leeren",
        "act_start_ocr": "Start",
        "act_stop_ocr": "Stopp",
        "act_re_ocr": "Wiederholen",
        "act_re_ocr_tip": "Ausgewählte Datei(en) erneut verarbeiten",
        "act_overlay_show": "Overlay-Boxen anzeigen",

        "status_ready": "Bereit.",
        "status_waiting": "Wartet",
        "status_processing": "Verarbeite...",
        "status_done": "Fertig",
        "status_error": "Fehler",

        "lbl_queue": "Wartebereich:",
        "lbl_lines": "Erkannte Zeilen:",
        "col_file": "Datei",
        "col_status": "Status",

        "drop_hint": "Datei(en) hierher ziehen und ablegen",
        "queue_drop_hint": "Datei(en) hierher ziehen und ablegen",

        "info_title": "Information",
        "warn_title": "Warnung",
        "err_title": "Fehler",

        "theme_bright": "Hell",
        "theme_dark": "Dunkel",

        "warn_queue_empty": "Wartebereich ist leer oder alle Elemente wurden verarbeitet.",
        "warn_select_done": "Keine Datei(en) für erneutes OCRn geladen.",
        "warn_need_rec": "Bitte wählen Sie zuerst ein Format-Modell (Recognition) aus.",
        "warn_need_seg": "Bitte wählen Sie zuerst ein Baseline-Modell aus.",

        "msg_stopping": "Breche ab...",
        "msg_finished": "Batch abgeschlossen.",
        "msg_device": "Gerät gesetzt auf: {}",
        "msg_exported": "Exportiert: {}",
        "msg_loaded_rec": "Format-Modell: {}",
        "msg_loaded_seg": "Baseline-Modell: {}",

        "err_load": "Bild kann nicht geladen werden: {}",

        "dlg_title_rename": "Umbenennen",
        "dlg_label_name": "Neuer Dateiname:",
        "dlg_save": "Speichern",
        "dlg_load_img": "Bilder wählen",
        "dlg_filter_img": "Bilder (*.png *.jpg *.jpeg *.tif *.bmp *.webp)",
        "dlg_choose_rec": "Recognition-Modell: ",
        "dlg_choose_seg": "Baseline-Modell: ",
        "dlg_filter_model": "Modelle (*.mlmodel *.pt)",

        "reading_tb_lr": "Oben → Unten + Links → Rechts",
        "reading_tb_rl": "Oben → Unten + Rechts → Links",
        "reading_bt_lr": "Unten → Oben + Links → Rechts",
        "reading_bt_rl": "Unten → Oben + Rechts → Links",

        "line_menu_move_up": "Zeile nach oben",
        "line_menu_move_down": "Zeile nach unten",
        "line_menu_delete": "Zeile löschen",
        "line_menu_add_above": "Zeile darüber hinzufügen",
        "line_menu_add_below": "Zeile darunter hinzufügen",
        "line_menu_draw_box": "Overlay-Box zeichnen",
        "line_menu_edit_box": "Overlay-Box bearbeiten (ziehen/skalieren)",
        "line_menu_move_to": "Zeile verschieben zu…",

        "dlg_new_line_title": "Neue Zeile",
        "dlg_new_line_label": "Text der neuen Zeile:",

        "dlg_move_to_title": "Zeile verschieben",
        "dlg_move_to_label": "Ziel-Zeilennummer (1…):",

        "canvas_menu_add_box_draw": "Overlay-Box hinzufügen (zeichnen)",
        "canvas_menu_delete_box": "Overlay-Box löschen",
        "canvas_menu_edit_box": "Overlay-Box bearbeiten…",
        "canvas_menu_select_line": "Zeile auswählen",

        "dlg_box_title": "Overlay-Box",
        "dlg_box_left": "links",
        "dlg_box_top": "oben",
        "dlg_box_right": "rechts",
        "dlg_box_bottom": "unten",
        "dlg_box_apply": "Anwenden",

        "export_choose_mode_title": "Export",
        "export_mode_all": "Alle Dateien exportieren",
        "export_mode_selected": "Ausgewählte Dateien exportieren",
        "export_select_files_title": "Dateien auswählen",
        "export_select_files_hint": "Wählen Sie die Dateien für den Export:",
        "export_choose_folder": "Zielordner wählen",
        "export_need_done": "Mindestens eine ausgewählte Datei ist nicht fertig verarbeitet.",
        "export_none_selected": "Keine Dateien ausgewählt.",

        "undo_nothing": "Nichts zum Rückgängig machen.",
        "redo_nothing": "Nichts zum Wiederholen.",
        "overlay_only_after_ocr": "Overlay-Bearbeitung ist erst nach abgeschlossener OCR möglich.",

        "new_line_from_box_title": "Neue Zeile",
        "new_line_from_box_label": "Text für die neue Zeile (optional):",

        "log_added_files": "{} Datei(en) zur Queue hinzugefügt.",
        "log_ocr_started": "OCR gestartet: {} Datei(en), Device={}, Reading={}",
        "log_stop_requested": "OCR-Abbruch angefordert.",
        "log_file_started": "Starte Datei: {}",
        "log_file_done": "Fertig: {} ({} Zeilen)",
        "log_file_error": "Fehler: {} -> {}",
        "log_export_done": "Export abgeschlossen: {} Datei(en) als {} nach {}",
        "log_export_single": "Export: {} -> {}",
        "log_export_log_done": "Log exportiert: {}",

    },

    "en": {
        "app_title": "Kraken OCR Professional",
        "toolbar_main": "Toolbar",
        "menu_file": "&File",
        "menu_edit": "&Edit",
        "menu_export": "Export as...",
        "menu_exit": "Exit",
        "menu_models": "&Models",
        "menu_options": "&Options",
        "menu_languages": "Languages",
        "menu_hw": "CPU/GPU",
        "menu_reading": "Reading Direction",
        "menu_appearance": "Appearance",

        "log_toggle_show": "Log",
        "log_toggle_hide": "Log",
        "menu_export_log": "Export log as .txt...",
        "dlg_save_log": "Save log",
        "dlg_filter_txt": "Text (*.txt)",
        "log_started": "Program started.",
        "log_queue_cleared": "Queue cleared.",

        "lang_de": "German",
        "lang_en": "English",
        "lang_fr": "French",

        "hw_cpu": "CPU",
        "hw_cuda": "GPU – CUDA (NVIDIA)",
        "hw_rocm": "GPU – ROCm (AMD)",
        "hw_mps": "GPU – MPS (Apple)",

        "act_undo": "Undo",
        "act_redo": "Redo",

        "msg_hw_not_available": "This hardware is not available on this system. Switching to CPU.",
        "msg_using_device": "Using device: {}",
        "msg_detected_gpu": "Detected: {}",
        "msg_device_cpu": "CPU",
        "msg_device_cuda": "CUDA",
        "msg_device_rocm": "ROCm",
        "msg_device_mps": "MPS",

        "act_add_files": "Load files...",
        "act_download_model": "Download model (Zenodo)",
        "act_delete": "Delete",
        "act_rename": "Rename...",
        "act_clear_queue": "Clear queue",
        "act_start_ocr": "Start",
        "act_stop_ocr": "Stop",
        "act_re_ocr": "Reprocess",
        "act_re_ocr_tip": "Reprocess selected file(s)",
        "act_overlay_show": "Show overlay boxes",

        "status_ready": "Ready.",
        "status_waiting": "Waiting",
        "status_processing": "Processing...",
        "status_done": "Done",
        "status_error": "Error",

        "lbl_queue": "Queue:",
        "lbl_lines": "Recognized lines:",
        "col_file": "File",
        "col_status": "Status",

        "drop_hint": "Drag & drop files here",
        "queue_drop_hint": "Drag & drop files here",

        "info_title": "Information",
        "warn_title": "Warning",
        "err_title": "Error",

        "theme_bright": "Bright",
        "theme_dark": "Dark",

        "warn_queue_empty": "Queue is empty or all items are processed.",
        "warn_select_done": "No file(s) loaded for re-OCR.",
        "warn_need_rec": "Please select a format model (recognition) first.",
        "warn_need_seg": "Please select a baseline model first.",

        "msg_stopping": "Stopping...",
        "msg_finished": "Batch finished.",
        "msg_device": "Device set to: {}",
        "msg_exported": "Exported: {}",
        "msg_loaded_rec": "Format model: {}",
        "msg_loaded_seg": "Baseline model: {}",

        "err_load": "Cannot load image: {}",

        "dlg_title_rename": "Rename",
        "dlg_label_name": "New filename:",
        "dlg_save": "Save",
        "dlg_load_img": "Choose images",
        "dlg_filter_img": "Images (*.png *.jpg *.jpeg *.tif *.bmp *.webp)",
        "dlg_choose_rec": "recognition model: ",
        "dlg_choose_seg": "baseline model: ",
        "dlg_filter_model": "Models (*.mlmodel *.pt)",

        "reading_tb_lr": "Top → Bottom + Left → Right",
        "reading_tb_rl": "Top → Bottom + Right → Left",
        "reading_bt_lr": "Bottom → Top + Left → Right",
        "reading_bt_rl": "Bottom → Top + Right → Left",

        "line_menu_move_up": "Move line up",
        "line_menu_move_down": "Move line down",
        "line_menu_delete": "Delete line",
        "line_menu_add_above": "Add line above",
        "line_menu_add_below": "Add line below",
        "line_menu_draw_box": "Draw overlay box",
        "line_menu_edit_box": "Edit overlay box (move/resize)",
        "line_menu_move_to": "Move line to…",

        "dlg_new_line_title": "New line",
        "dlg_new_line_label": "Text of the new line:",

        "dlg_move_to_title": "Move line",
        "dlg_move_to_label": "Target line number (1…):",

        "canvas_menu_add_box_draw": "Add overlay box (draw)",
        "canvas_menu_delete_box": "Delete overlay box",
        "canvas_menu_edit_box": "Edit overlay box…",
        "canvas_menu_select_line": "Select line",

        "dlg_box_title": "Overlay box",
        "dlg_box_left": "left",
        "dlg_box_top": "top",
        "dlg_box_right": "right",
        "dlg_box_bottom": "bottom",
        "dlg_box_apply": "Apply",

        "export_choose_mode_title": "Export",
        "export_mode_all": "Export all files",
        "export_mode_selected": "Export selected files",
        "export_select_files_title": "Select files",
        "export_select_files_hint": "Choose files to export:",
        "export_choose_folder": "Choose destination folder",
        "export_need_done": "At least one selected file is not finished.",
        "export_none_selected": "No files selected.",

        "undo_nothing": "Nothing to undo.",
        "redo_nothing": "Nothing to redo.",
        "overlay_only_after_ocr": "Overlay editing is only available after OCR is finished.",

        "new_line_from_box_title": "New line",
        "new_line_from_box_label": "Text for the new line (optional):",

        "log_added_files": "{} file(s) added to the queue.",
        "log_ocr_started": "OCR started: {} file(s), Device={}, Reading={}",
        "log_stop_requested": "OCR stop requested.",
        "log_file_started": "Starting file: {}",
        "log_file_done": "Done: {} ({} lines)",
        "log_file_error": "Error: {} -> {}",
        "log_export_done": "Export finished: {} file(s) as {} to {}",
        "log_export_single": "Export: {} -> {}",
        "log_export_log_done": "Log exported: {}",

    },

    "fr": {
        "app_title": "Professionnel Kraken OCR",
        "toolbar_main": "Barre d’outils",
        "menu_file": "&Fichier",
        "menu_edit": "&Édition",
        "menu_export": "Exporter en tant que...",
        "menu_exit": "Quitter",
        "menu_models": "&Modèles",
        "menu_options": "&Options",
        "menu_languages": "Langues",
        "menu_hw": "CPU/GPU",
        "menu_reading": "Direction de lecture",
        "menu_appearance": "Apparence",

        "log_toggle_show": "Log",
        "log_toggle_hide": "Log",
        "menu_export_log": "Exporter le log en .txt...",
        "dlg_save_log": "Enregistrer le log",
        "dlg_filter_txt": "Texte (*.txt)",
        "log_started": "Programme démarré.",
        "log_queue_cleared": "File d’attente vidée.",

        "lang_de": "Allemand",
        "lang_en": "Anglais",
        "lang_fr": "Français",

        "hw_cpu": "CPU",
        "hw_cuda": "GPU – CUDA (NVIDIA)",
        "hw_rocm": "GPU – ROCm (AMD)",
        "hw_mps": "GPU – MPS (Apple)",

        "act_undo": "Annuler",
        "act_redo": "Rétablir",

        "msg_hw_not_available": "Ce matériel n’est pas disponible sur ce système. Retour au CPU.",
        "msg_using_device": "Appareil utilisé : {}",
        "msg_detected_gpu": "Détecté : {}",
        "msg_device_cpu": "CPU",
        "msg_device_cuda": "CUDA",
        "msg_device_rocm": "ROCm",
        "msg_device_mps": "MPS",

        "act_add_files": "Charger des fichiers…",
        "act_download_model": "Télécharger le modèle (Zenodo)",
        "act_delete": "Supprimer",
        "act_rename": "Renommer...",
        "act_clear_queue": "Vider la file d’attente",
        "act_start_ocr": "Démarrer",
        "act_stop_ocr": "Arrêter",
        "act_re_ocr": "Relancer",
        "act_re_ocr_tip": "Relancer le traitement du/des fichier(s) sélectionné(s)",
        "act_overlay_show": "Afficher les boîtes de superposition",

        "status_ready": "Prêt.",
        "status_waiting": "En attente",
        "status_processing": "Traitement...",
        "status_done": "Terminé",
        "status_error": "Erreur",

        "lbl_queue": "File d’attente:",
        "lbl_lines": "Lignes reconnues:",
        "col_file": "Fichier",
        "col_status": "Statut",

        "drop_hint": "Glissez-déposez des fichiers ici",
        "queue_drop_hint": "Glissez-déposez des fichiers ici",

        "info_title": "Information",
        "warn_title": "Avertissement",
        "err_title": "Erreur",

        "theme_bright": "Clair",
        "theme_dark": "Sombre",

        "warn_queue_empty": "La file d’attente est vide ou tous les éléments ont été traités.",
        "warn_select_done": "Aucun fichier chargé pour relancer l’OCR.",
        "warn_need_rec": "Veuillez d’abord sélectionner un modèle de format (reconnaissance).",
        "warn_need_seg": "Veuillez d’abord sélectionner un modèle de baseline.",

        "msg_stopping": "Arrêt...",
        "msg_finished": "Traitement terminé.",
        "msg_device": "Appareil réglé sur: {}",
        "msg_exported": "Exporté: {}",
        "msg_loaded_rec": "Modèle de format: {}",
        "msg_loaded_seg": "Modèle de baseline: {}",

        "err_load": "Impossible de charger l’image: {}",

        "dlg_title_rename": "Renommer",
        "dlg_label_name": "Nouveau nom de fichier:",
        "dlg_save": "Enregistrer",
        "dlg_load_img": "Choisir des images",
        "dlg_filter_img": "Images (*.png *.jpg *.jpeg *.tif *.bmp *.webp)",
        "dlg_choose_rec": "le modèle de reconnaissance: ",
        "dlg_choose_seg": "le modèle de baseline: ",
        "dlg_filter_model": "Modèles (*.mlmodel *.pt)",

        "reading_tb_lr": "Haut → Bas + Gauche → Droite",
        "reading_tb_rl": "Haut → Bas + Droite → Gauche",
        "reading_bt_lr": "Bas → Haut + Gauche → Droite",
        "reading_bt_rl": "Bas → Haut + Droite → Gauche",

        "line_menu_move_up": "Monter la ligne",
        "line_menu_move_down": "Descendre la ligne",
        "line_menu_delete": "Supprimer la ligne",
        "line_menu_add_above": "Ajouter une ligne au-dessus",
        "line_menu_add_below": "Ajouter une ligne en dessous",
        "line_menu_draw_box": "Dessiner la boîte",
        "line_menu_edit_box": "Modifier la boîte (déplacer/redimensionner)",
        "line_menu_move_to": "Déplacer la ligne vers…",

        "dlg_new_line_title": "Nouvelle ligne",
        "dlg_new_line_label": "Texte de la nouvelle ligne:",

        "dlg_move_to_title": "Déplacer la ligne",
        "dlg_move_to_label": "Numéro de ligne cible (1…):",

        "canvas_menu_add_box_draw": "Ajouter une boîte (dessiner)",
        "canvas_menu_delete_box": "Supprimer la boîte",
        "canvas_menu_edit_box": "Modifier la boîte…",
        "canvas_menu_select_line": "Sélectionner la ligne",

        "dlg_box_title": "Boîte de superposition",
        "dlg_box_left": "gauche",
        "dlg_box_top": "haut",
        "dlg_box_right": "droite",
        "dlg_box_bottom": "bas",
        "dlg_box_apply": "Appliquer",

        "export_choose_mode_title": "Export",
        "export_mode_all": "Exporter tous les fichiers",
        "export_mode_selected": "Exporter les fichiers sélectionnés",
        "export_select_files_title": "Sélectionner des fichiers",
        "export_select_files_hint": "Choisissez les fichiers à exporter :",
        "export_choose_folder": "Choisir le dossier de destination",
        "export_need_done": "Au moins un fichier sélectionné n’est pas terminé.",
        "export_none_selected": "Aucun fichier sélectionné.",

        "undo_nothing": "Rien à annuler.",
        "redo_nothing": "Rien à rétablir.",
        "overlay_only_after_ocr": "L’édition des overlays n’est disponible qu’après l’OCR.",

        "new_line_from_box_title": "Nouvelle ligne",
        "new_line_from_box_label": "Texte pour la nouvelle ligne (optionnel):",

        "log_added_files": "{} fichier(s) ajouté(s) à la file d’attente.",
        "log_ocr_started": "OCR démarré : {} fichier(s), Appareil={}, Lecture={}",
        "log_stop_requested": "Arrêt de l’OCR demandé.",
        "log_file_started": "Traitement du fichier : {}",
        "log_file_done": "Terminé : {} ({} lignes)",
        "log_file_error": "Erreur : {} -> {}",
        "log_export_done": "Export terminé : {} fichier(s) en {} vers {}",
        "log_export_single": "Export : {} -> {}",
        "log_export_log_done": "Log exporté : {}",

    }
}

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]

# -----------------------------
# DATA CLASSES
# -----------------------------
@dataclass
class RecordView:
    idx: int
    text: str
    bbox: Optional[BBox]

UndoSnapshot = Tuple[List[Tuple[str, Optional[BBox]]], int]

@dataclass
class TaskItem:
    path: str
    display_name: str
    status: int = STATUS_WAITING
    results: Optional[Tuple[str, list, Image.Image, List[RecordView]]] = None
    edited: bool = False
    undo_stack: List[UndoSnapshot] = field(default_factory=list)
    redo_stack: List[UndoSnapshot] = field(default_factory=list)

@dataclass
class OCRJob:
    input_paths: List[str]
    recognition_model_path: str
    segmentation_model_path: Optional[str]
    device: str
    reading_direction: int
    export_format: str
    export_dir: Optional[str]
    segmenter_mode: str = "blla"

# -----------------------------
# GEOMETRY & SORTING
# -----------------------------
Point = Tuple[float, float]


def _coerce_points(obj: Any) -> List[Point]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        first = obj[0]
        if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
            try:
                return [(float(x), float(y)) for x, y in obj]
            except Exception:
                return []
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
            pts: List[Point] = []
            for contour in obj:
                pts.extend(_coerce_points(contour))
            return pts
    return []


def _bbox_from_points(points: List[Point], pad: int = 0) -> Optional[Tuple[int, int, int, int]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = int(min(xs)) - pad
    y0 = int(min(ys)) - pad
    x1 = int(max(xs)) + pad
    y1 = int(max(ys)) + pad
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def record_bbox(r: Any) -> Optional[Tuple[int, int, int, int]]:
    bbox = getattr(r, "bbox", None)
    if bbox:
        try:
            x0, y0, x1, y1 = bbox
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
        except Exception:
            pass

    for attr in ("boundary", "polygon"):
        boundary = getattr(r, attr, None)
        if boundary:
            pts = _coerce_points(boundary)
            bb = _bbox_from_points(pts, pad=2)
            if bb:
                return bb

    baseline = getattr(r, "baseline", None)
    if baseline:
        pts = _coerce_points(baseline)
        bb = _bbox_from_points(pts, pad=2)
        if bb:
            x0, y0, x1, y1 = bb
            vpad = 14
            return x0, y0 - vpad, x1, y1 + vpad
    return None

def baseline_length(bl) -> float:
    pts = _coerce_points(bl)
    if len(pts) < 2:
        return 0.0
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    return math.hypot(x2 - x1, y2 - y1)

# Vertikale Separator-Records (Spaltentrenner)
VSEP_RE = re.compile(r'^[\|\u2502\u2503]+$')  # | │ ┃

# Horizontale Separator-Records (Zeilentrenner)
HSEP_RE = re.compile(r'^[_\-\u2500\u2501\u2504\u2505]{3,}$')  # ___ --- ─ ━ etc. (mind. 3)

def sort_records_reading_order(records, image_width: int, image_height: int, reading_mode: int = READING_MODES["TB_LR"]):

    # ---------- helpers ----------
    def cx(bb): return (bb[0] + bb[2]) / 2.0
    def cy(bb): return (bb[1] + bb[3]) / 2.0
    def bw(bb): return bb[2] - bb[0]
    def bh(bb): return bb[3] - bb[1]

    def quant(vals, p):
        if not vals:
            return None
        vs = sorted(vals)
        k = (len(vs) - 1) * p
        f = int(k)
        c = min(f + 1, len(vs) - 1)
        if f == c:
            return vs[f]
        return vs[f] + (vs[c] - vs[f]) * (k - f)

    # Direction flags
    rev_y = (reading_mode in (READING_MODES["BT_LR"], READING_MODES["BT_RL"]))     # bottom->top
    rev_cols = (reading_mode in (READING_MODES["TB_RL"], READING_MODES["BT_RL"])) # columns right->left

    W = max(1, int(image_width))

    # ---------- collect bboxes ----------
    raw = []
    for r in records:
        bb = record_bbox(r)
        if bb:
            raw.append((r, bb))
    if not raw:
        return list(records)

    # ---------- estimate skew angle from baselines (best) ----------
    angles = []
    for r, _ in raw:
        bl = getattr(r, "baseline", None)
        pts = _coerce_points(bl)
        if len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[-1]
            dx = (x2 - x1)
            dy = (y2 - y1)
            if abs(dx) > 1.0:
                a = math.atan2(dy, dx)
                # reject crazy angles
                if abs(a) < math.radians(20):
                    angles.append(a)

    skew = statistics.median(angles) if angles else 0.0
    # rotate coordinates by -skew (deskew)
    cs = math.cos(-skew)
    sn = math.sin(-skew)

    Wc = max(1.0, float(image_width)) / 2.0
    Hc = max(1.0, float(image_height)) / 2.0

    def rot(x, y):
        # translate -> rotate -> translate back
        x -= Wc
        y -= Hc
        xr = x * cs - y * sn
        yr = x * sn + y * cs
        return (xr + Wc, yr + Hc)

    def deskew_bb(bb):
        x0, y0, x1, y1 = bb
        pts = [rot(x0, y0), rot(x1, y0), rot(x1, y1), rot(x0, y1)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    items = []
    for r, bb in raw:
        dbb = deskew_bb(bb)
        items.append((r, bb, dbb))

    # ---------- typical line height (deskewed) ----------
    hs = [ (dbb[3]-dbb[1]) for _,_,dbb in items if (dbb[3]-dbb[1]) > 0 ]
    med_h = sorted(hs)[len(hs)//2] if hs else 14.0
    MIN_H = max(10.0, 0.6 * med_h)

    def is_fullwidth(dbb):
        return (dbb[2]-dbb[0]) >= 0.82 * W

    # body candidates
    body = [(r, bb, dbb) for (r, bb, dbb) in items if (dbb[3]-dbb[1]) >= MIN_H and not is_fullwidth(dbb)]
    if len(body) < 8:
        # fallback: deskewed y then x
        ordered = sorted(items, key=lambda x: (cy(x[2]), cx(x[2])), reverse=rev_y)
        return [r for r,_,_ in ordered]

    # ---------- header/footer by y quantiles (deskewed) ----------
    ys_top = [dbb[1] for _,_,dbb in body]
    ys_bot = [dbb[3] for _,_,dbb in body]
    body_top = quant(ys_top, 0.08)
    body_bot = quant(ys_bot, 0.92)
    if body_top is None or body_bot is None:
        ordered = sorted(items, key=lambda x: (cy(x[2]), cx(x[2])), reverse=rev_y)
        return [r for r,_,_ in ordered]

    MARGIN_Y = max(10.0, 0.8 * med_h)

    header, footer, midband = [], [], []
    for r, bb, dbb in items:
        if dbb[3] < (body_top - MARGIN_Y):
            header.append((r, bb, dbb))
        elif dbb[1] > (body_bot + MARGIN_Y):
            footer.append((r, bb, dbb))
        else:
            midband.append((r, bb, dbb))

    def sort_y_then_x(lst):
        return sorted(lst, key=lambda x: (cy(x[2]), cx(x[2])), reverse=rev_y)

    header_sorted = sort_y_then_x(header)
    footer_sorted = sort_y_then_x(footer)

    # ---------- detect vertical separators ('|', '│', '┃') as gutters ----------
    # We treat records that are basically only "|" and are tall+thin as separators.
    sep_x = []
    for r, bb, dbb in midband:
        pred = getattr(r, "prediction", "")
        t = str(pred).strip()
        if not t:
            continue
        if VSEP_RE.match(t):
            w_sep = (dbb[2]-dbb[0])
            h_sep = (dbb[3]-dbb[1])
            # etwas weniger streng -> Separator wird eher erkannt
            if w_sep <= 0.05 * W and h_sep >= 1.8 * med_h:
                sep_x.append(cx(dbb))


    sep_x.sort()
    # keep only separators that are reasonably distinct
    filtered = []
    for x in sep_x:
        if not filtered or abs(x - filtered[-1]) > max(10.0, 0.02 * W):
            filtered.append(x)
    sep_x = filtered

    # ---------- build columns ----------
    mid_text = [(r, bb, dbb) for (r, bb, dbb) in midband if (dbb[3]-dbb[1]) >= MIN_H and not is_fullwidth(dbb)]

    # If we have explicit separators, use them as boundaries:
    # columns = count(separators) + 1
    if len(sep_x) >= 1:
        bounds = sep_x[:]  # each is a boundary x
        ncols = len(bounds) + 1

        GUTTER = max(18.0, 0.01 * W)  # Schutzbereich um die Trennlinie

        def col_index_for(dbb):
            # Fullwidth immer in "erste Spalte" (Header/Spanner behandeln wir später separat)
            if is_fullwidth(dbb):
                return 0
            # Wenn eine Box komplett links von der Trennlinie liegt -> links
            # (wichtig für rechtsbündige kurze Zeilen wie "w. Koehler.")
            for i, b in enumerate(bounds):
                if dbb[2] <= b - GUTTER:   # right edge klar links
                    return i

            # Wenn komplett rechts -> rechts
            for i, b in enumerate(bounds):
                if dbb[0] >= b + GUTTER:   # left edge klar rechts
                    continue
                # überlappt GUTTER -> entscheide über Center
                break

            x_center = (dbb[0] + dbb[2]) / 2.0
            i = 0
            for b in bounds:
                if x_center < b:
                    return i
                i += 1
            return ncols - 1

        cols = [[] for _ in range(ncols)]
        for r, bb, dbb in midband:
            cols[col_index_for(dbb)].append((r, bb, dbb))

    else:
        # No explicit separators: cluster by left edge x0 (deskewed), but robust against skew/indent
        x_threshold = max(55.0, 0.07 * W)      # a bit larger -> skew tolerant
        indent_dx   = max(30.0, 0.05 * W)      # merge indents more aggressively
        min_items_for_real_col = max(10, int(0.12 * len(mid_text)))  # require stronger evidence

        clusters = []  # {"x": mean_x0, "items":[...]}
        for r, bb, dbb in mid_text:
            x0 = dbb[0]
            placed = False
            for c in clusters:
                if abs(c["x"] - x0) <= x_threshold:
                    c["items"].append((r, bb, dbb))
                    c["x"] = (c["x"] * 0.85) + (x0 * 0.15)
                    placed = True
                    break
            if not placed:
                clusters.append({"x": float(x0), "items": [(r, bb, dbb)]})

        clusters.sort(key=lambda c: c["x"])
        # --- NEW: merge "indent/center" clusters by strong horizontal overlap ---
        def q(vals, p):
            if not vals:
                return None
            vs = sorted(vals)
            k = (len(vs) - 1) * p
            f = int(k)
            c = min(f + 1, len(vs) - 1)
            if f == c:
                return vs[f]
            return vs[f] + (vs[c] - vs[f]) * (k - f)

        def span(c):
            lefts = [it[2][0] for it in c["items"]]   # dbb x0
            rights = [it[2][2] for it in c["items"]]  # dbb x1
            l = q(lefts, 0.20) if lefts else c["x"]
            r = q(rights, 0.80) if rights else c["x"]
            if l is None or r is None:
                return (c["x"], c["x"])
            return (float(l), float(r))

        def should_merge(c1, c2):
            l1, r1 = span(c1)
            l2, r2 = span(c2)
            w1 = max(1.0, r1 - l1)
            w2 = max(1.0, r2 - l2)

            # overlap ratio (indent/center => high, real columns => near 0)
            overlap = max(0.0, min(r1, r2) - max(l1, l2))
            overlap_ratio = overlap / max(1.0, min(w1, w2))

            dx = abs(c2["x"] - c1["x"])

            # If close-ish in x OR one sits inside the other, AND they overlap strongly -> merge
            close = dx <= max(80.0, 0.12 * W)
            inside = (l2 >= l1 - 0.03 * W and r2 <= r1 + 0.03 * W) or (l1 >= l2 - 0.03 * W and r1 <= r2 + 0.03 * W)

            return (overlap_ratio >= 0.55) and (close or inside)

        merged_pass = True
        while merged_pass and len(clusters) > 1:
            merged_pass = False
            new_list = []
            i = 0
            while i < len(clusters):
                if i < len(clusters) - 1 and should_merge(clusters[i], clusters[i + 1]):
                    a = clusters[i]
                    b = clusters[i + 1]
                    a["items"].extend(b["items"])
                    # update mean x0
                    a["x"] = float(sum(it[2][0] for it in a["items"])) / max(1, len(a["items"]))
                    new_list.append(a)
                    i += 2
                    merged_pass = True
                else:
                    new_list.append(clusters[i])
                    i += 1
            clusters = new_list

        clusters.sort(key=lambda c: c["x"])

        # merge small "indent" clusters into nearest real cluster
        def is_real(c):
            return len(c["items"]) >= min_items_for_real_col

        merged = clusters[:]
        for c in list(merged):
            if is_real(c):
                continue
            # nearest real cluster
            best = None
            best_d = None
            for t in merged:
                if t is c or not is_real(t):
                    continue
                d = abs(t["x"] - c["x"])
                if best_d is None or d < best_d:
                    best, best_d = t, d
            if best is not None and best_d is not None and best_d <= indent_dx:
                best["items"].extend(c["items"])
                merged = [z for z in merged if z is not c]

        merged.sort(key=lambda c: c["x"])

        # If still looks like "one column with indents" -> treat as single column.
        if len(merged) >= 2:
            sizes = sorted([len(c["items"]) for c in merged], reverse=True)
            biggest = sizes[0]
            ratio = biggest / max(1, sum(sizes))
            # if one cluster dominates strongly -> single column
            if ratio >= 0.70:
                merged = [max(merged, key=lambda c: len(c["items"]))]

        if len(merged) <= 1:
            # single column -> just y then x
            core = sort_y_then_x(midband)
            return [r for r,_,_ in header_sorted] + [r for r,_,_ in core] + [r for r,_,_ in footer_sorted]

        # build bounds between cluster starts
        col_starts = [c["x"] for c in merged]
        bounds = [(col_starts[i] + col_starts[i+1]) / 2.0 for i in range(len(col_starts) - 1)]

        def col_index_for(dbb):
            x = dbb[0]
            if is_fullwidth(dbb):
                return 0
            for i, b in enumerate(bounds):
                if x < b:
                    return i
            return len(col_starts) - 1

        cols = [[] for _ in range(len(col_starts))]
        for r, bb, dbb in midband:
            cols[col_index_for(dbb)].append((r, bb, dbb))

        # ---------- SECOND PASS: centered headings above columns -> header ----------
        def body_like(dbb):
            h = (dbb[3] - dbb[1])
            w = (dbb[2] - dbb[0])
            if h < MIN_H:
                return False
            if is_fullwidth(dbb):
                return False
            return w >= 0.10 * W

        # oberste "echte" Textzeile in den Spalten finden
        col_tops = []
        for col in cols:
            ys = [it[2][1] for it in col if body_like(it[2])]
            if ys:
                col_tops.append(min(ys))
        first_body_y = min(col_tops) if col_tops else body_top

        def is_centered_heading(dbb):
            w = (dbb[2] - dbb[0])
            if w > 0.85 * W:
                # sehr breit -> eher normaler Absatz; den lassen wir hier in Ruhe
                return False
            x_center = (dbb[0] + dbb[2]) / 2.0
            return abs(x_center - (W / 2.0)) <= 0.18 * W  # "zentriert genug"

        promote = []
        keep_mid = []
        Y_PAD = max(10.0, 0.9 * med_h)

        for r, bb, dbb in midband:
            # deutlich oberhalb der ersten Spaltenzeile UND zentriert -> header
            if (dbb[3] < (first_body_y - Y_PAD)) and is_centered_heading(dbb):
                promote.append((r, bb, dbb))
            else:
                keep_mid.append((r, bb, dbb))

        if promote:
            # in header aufnehmen
            header_sorted = sort_y_then_x(header + promote)
            midband = keep_mid

            # cols NEU aus midband aufbauen
            cols = [[] for _ in range(len(col_starts))]
            for r, bb, dbb in midband:
                cols[col_index_for(dbb)].append((r, bb, dbb))

    # ---------- sort within each column ----------
    def sort_col(col):
        return sorted(col, key=lambda x: (cy(x[2]), cx(x[2])), reverse=rev_y)

    cols = [sort_col(c) for c in cols]

    col_order = list(range(len(cols)))
    if rev_cols:
        col_order = list(reversed(col_order))

    core = []
    for ci in col_order:
        core.extend(cols[ci])

    return [r for r,_,_ in header_sorted] + [r for r,_,_ in core] + [r for r,_,_ in footer_sorted]


def clamp_bbox(bb: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bb
    return (max(0, min(w - 1, x0)), max(0, min(h - 1, y0)),
            max(0, min(w, x1)), max(0, min(h, y1)))


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


# -----------------------------
# TABLE EXPORT HELPERS
# -----------------------------
def cluster_columns(records: List[RecordView], x_threshold: int = 45):
    cols = []
    for r in records:
        if not r.bbox:
            continue
        x0 = r.bbox[0]
        placed = False
        for c in cols:
            if abs(c["x"] - x0) <= x_threshold:
                c["items"].append(r)
                c["x"] = int((c["x"] * 0.8) + (x0 * 0.2))
                placed = True
                break
        if not placed:
            cols.append({"x": x0, "items": [r]})
    cols.sort(key=lambda c: c["x"])
    return [c["items"] for c in cols]


def is_same_visual_row(a: RecordView, b: RecordView, page_width: int) -> bool:
    if not a.bbox or not b.bbox:
        return False

    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox

    # y-Ähnlichkeit
    if abs(ay0 - by0) > 12:
        return False

    w = max(1, int(page_width))
    mid = w // 2

    aw = ax1 - ax0
    bw = bx1 - bx0

    # Wenn beide Boxen "Textzeilen-breit" sind und in unterschiedlichen Spalten liegen,
    # dann sind das KEINE Tabellenzellen derselben Zeile.
    textish_a = aw >= int(0.30 * w)
    textish_b = bw >= int(0.30 * w)

    a_left = (ax0 < mid and ax1 <= mid + int(0.05*w))
    b_right = (bx1 > mid and bx0 >= mid - int(0.05*w))
    b_left = (bx0 < mid and bx1 <= mid + int(0.05*w))
    a_right = (ax1 > mid and ax0 >= mid - int(0.05*w))

    if textish_a and textish_b and ((a_left and b_right) or (b_left and a_right)):
        return False

    return True

def group_rows_by_y(records: List[RecordView], page_width: int):
    recs = [r for r in records if r.bbox]
    if not recs:
        return []

    w = max(1, int(page_width))

    # robuste Zeilenhöhe
    hs = sorted([(rv.bbox[3] - rv.bbox[1]) for rv in recs if (rv.bbox[3] - rv.bbox[1]) > 0])
    med_h = hs[len(hs)//2] if hs else 14

    # enger = "Abstand geringer" (striktere Gruppierung)
    y_tol = max(10, int(0.45 * med_h))

    # NEW: horizontale Separatoren (_____ / ---- / ───) erkennen
    sep_y: List[float] = []
    filtered_recs: List[RecordView] = []
    for rv in recs:
        txt = (rv.text or "").strip()
        x0, y0, x1, y1 = rv.bbox
        bw = (x1 - x0)
        bh = (y1 - y0)

        is_hsep = bool(HSEP_RE.match(txt)) and (bw >= 0.55 * w) and (bh <= 0.7 * med_h)
        if is_hsep:
            sep_y.append((y0 + y1) / 2.0)
        else:
            filtered_recs.append(rv)

    sep_y.sort()
    recs = filtered_recs

    def center_y(rv):
        x0, y0, x1, y1 = rv.bbox
        return (y0 + y1) / 2.0

    sorted_recs = sorted(recs, key=lambda rv: (center_y(rv), rv.bbox[0]))

    rows: List[List[RecordView]] = []
    row_y: List[float] = []
    row_band: List[int] = []

    def band_index(cy: float) -> int:
        # wie viele Separatoren liegen oberhalb? -> Band 0..n
        idx = 0
        for y in sep_y:
            if cy > y:
                idx += 1
            else:
                break
        return idx

    for r in sorted_recs:
        cy = center_y(r)
        b = band_index(cy)
        placed = False

        for i in range(len(rows)):
            if row_band[i] != b:
                continue
            if abs(cy - row_y[i]) <= y_tol:
                rows[i].append(r)
                row_y[i] = row_y[i] * 0.85 + cy * 0.15
                placed = True
                break

        if not placed:
            rows.append([r])
            row_y.append(cy)
            row_band.append(b)

    for row in rows:
        row.sort(key=lambda rv: rv.bbox[0])
    return rows

def table_to_rows(records: List[RecordView], page_width: int) -> List[List[str]]:
    # Wenn der Text explizite Trenner enthält, nutze die als "harte" Spalten,
    # statt aus BBox-Positionen eine Tabelle zu raten.
    has_pipes = any(
        (rv.text and (
                any(ch in rv.text for ch in ("|", "│", "┃")) or
                re.search(r"(?:_{2,}|\s_\s)", rv.text)  # "__" oder " _ " als Trenner
        ))
        for rv in records
    )

    if has_pipes:
        rows = group_rows_by_y(records, page_width)
        grid = []
        for row in rows:
            # links->rechts sortieren
            row = [rv for rv in row if rv.bbox]
            row.sort(key=lambda rv: rv.bbox[0] if rv.bbox else 0)

            cells: List[str] = []
            for rv in row:
                txt = (rv.text or "").strip()
                if not txt:
                    continue
                # reine Separator-Records ignorieren
                if re.fullmatch(r"[\|\u2502\u2503]+", txt):
                    continue
                # split an pipes
                if any(ch in txt for ch in ("|", "│", "┃")):
                    parts = re.split(r"\s*(?:[\|\u2502\u2503]+|_{2,}|\s_\s)\s*", txt)
                    parts = [p.strip() for p in parts if p.strip()]
                    if parts:
                        cells.extend(parts)
                else:
                    cells.append(txt)

            grid.append(cells if cells else [""])
        return grid

    # sonst: dein bestehender bbox-basierter Tabellenmodus
    rows = group_rows_by_y(records, page_width)
    cols = cluster_columns(records)

    col_x = []
    for col in cols:
        xs = [rv.bbox[0] for rv in col if rv.bbox]
        col_x.append(int(sum(xs) / max(1, len(xs))) if xs else 0)

    def nearest_col(x: int) -> int:
        if not col_x:
            return 0
        best_i = 0
        best_d = abs(col_x[0] - x)
        for i in range(1, len(col_x)):
            d = abs(col_x[i] - x)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    grid = []
    for row in rows:
        line = [""] * max(1, len(col_x))
        for rv in row:
            if not rv.bbox:
                continue
            c = nearest_col(rv.bbox[0])
            if line[c]:
                line[c] += " " + rv.text
            else:
                line[c] = rv.text
        grid.append(line)
    return grid

# -----------------------------
# RESIZABLE / MOVABLE RECT ITEM
# -----------------------------
class ResizableRectItem(QGraphicsRectItem):
    """
    Movable + resizable rect.
    Calls on_changed(idx, QRectF(scene coords)) after mouse release.
    """
    HANDLE_PAD = 6.0

    def __init__(self, rect: QRectF, idx: int, on_changed: Callable[[int, QRectF], None],
                 on_double_clicked: Optional[Callable[[int], None]] = None):
        super().__init__(rect)
        self.idx = idx
        self._on_changed = on_changed
        self._on_double_clicked = on_double_clicked

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)

        self._mode = "none"
        self._resize_edges = (False, False, False, False)  # L,T,R,B
        self._press_item_pos: Optional[QPointF] = None
        self._press_rect: Optional[QRectF] = None

    def _hit_test_edges(self, pos: QPointF) -> Tuple[bool, bool, bool, bool]:
        r = self.rect()
        x, y = pos.x(), pos.y()
        l = abs(x - r.left()) <= self.HANDLE_PAD
        t = abs(y - r.top()) <= self.HANDLE_PAD
        rr = abs(x - r.right()) <= self.HANDLE_PAD
        b = abs(y - r.bottom()) <= self.HANDLE_PAD
        return l, t, rr, b

    def hoverMoveEvent(self, event):
        l, t, r, b = self._hit_test_edges(event.pos())
        if (l and t) or (r and b):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (r and t) or (l and b):
            self.setCursor(Qt.SizeBDiagCursor)
        elif l or r:
            self.setCursor(Qt.SizeHorCursor)
        elif t or b:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            l, t, r, b = self._hit_test_edges(event.pos())
            if l or t or r or b:
                self._mode = "resize"
                self._resize_edges = (l, t, r, b)
                self._press_item_pos = QPointF(event.pos())
                self._press_rect = QRectF(self.rect())
                self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
                event.accept()
                return
            else:
                self._mode = "move"
                self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
                self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mode == "resize" and self._press_item_pos is not None and self._press_rect is not None:
            delta = event.pos() - self._press_item_pos
            r = QRectF(self._press_rect)

            l, t, rr, b = self._resize_edges
            if l:
                r.setLeft(r.left() + delta.x())
            if rr:
                r.setRight(r.right() + delta.x())
            if t:
                r.setTop(r.top() + delta.y())
            if b:
                r.setBottom(r.bottom() + delta.y())

            r = r.normalized()
            if r.width() < 5:
                r.setWidth(5)
            if r.height() < 5:
                r.setHeight(5)

            self.setRect(r)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            try:
                if callable(self._on_double_clicked):
                    self._on_double_clicked(self.idx)
            except Exception:
                pass
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setCursor(Qt.OpenHandCursor)

        was_resize_or_move = (self._mode in ("resize", "move"))
        self._mode = "none"
        self._press_item_pos = None
        self._press_rect = None

        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)

        if was_resize_or_move:
            try:
                if callable(self._on_changed):
                    scene_rect = self.mapRectToScene(self.rect()).normalized()
                    self._on_changed(self.idx, scene_rect)
            except Exception:
                pass


# -----------------------------
# DROP-ENABLED QUEUE TABLE
# -----------------------------
class DropQueueTable(QTableWidget):
    files_dropped = Signal(list)
    table_resized = Signal()
    delete_pressed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.table_resized.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        files = []
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.exists(p):
                files.append(p)
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()


# -----------------------------
# LINES LIST (Delete + DnD reorder)
# -----------------------------
class LinesListWidget(QListWidget):
    delete_pressed = Signal()
    reorder_committed = Signal(list, int)  # new_order (list of old indices), current_row after drop

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropMode(QAbstractItemView.InternalMove)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dropEvent(self, event):
        super().dropEvent(event)
        order = []
        for i in range(self.count()):
            it = self.item(i)
            idx = it.data(Qt.UserRole)
            if idx is None:
                idx = i
            order.append(int(idx))
        self.reorder_committed.emit(order, self.currentRow())


# -----------------------------
# OVERLAY BOX EDIT DIALOG
# -----------------------------
class OverlayBoxDialog(QDialog):
    def __init__(self, tr, img_w: int, img_h: int, bbox: Optional[Tuple[int, int, int, int]] = None, parent=None):
        super().__init__(parent)
        self._tr = tr
        self.setWindowTitle(tr("dlg_box_title"))
        self._img_w = max(1, int(img_w))
        self._img_h = max(1, int(img_h))

        x0, y0, x1, y1 = (0, 0, min(100, self._img_w), min(30, self._img_h))
        if bbox:
            x0, y0, x1, y1 = bbox

        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.sp_x0 = QSpinBox()
        self.sp_y0 = QSpinBox()
        self.sp_x1 = QSpinBox()
        self.sp_y1 = QSpinBox()

        for sp in (self.sp_x0, self.sp_y0, self.sp_x1, self.sp_y1):
            sp.setRange(0, 1000000)

        self.sp_x0.setRange(0, self._img_w)
        self.sp_x1.setRange(0, self._img_w)
        self.sp_y0.setRange(0, self._img_h)
        self.sp_y1.setRange(0, self._img_h)

        self.sp_x0.setValue(max(0, min(self._img_w, int(x0))))
        self.sp_y0.setValue(max(0, min(self._img_h, int(y0))))
        self.sp_x1.setValue(max(0, min(self._img_w, int(x1))))
        self.sp_y1.setValue(max(0, min(self._img_h, int(y1))))

        form.addRow(tr("dlg_box_left"), self.sp_x0)
        form.addRow(tr("dlg_box_top"), self.sp_y0)
        form.addRow(tr("dlg_box_right"), self.sp_x1)
        form.addRow(tr("dlg_box_bottom"), self.sp_y1)

        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("dlg_box_apply"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        x0 = int(self.sp_x0.value())
        y0 = int(self.sp_y0.value())
        x1 = int(self.sp_x1.value())
        y1 = int(self.sp_y1.value())
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 <= x0:
            x1 = min(self._img_w, x0 + 1)
        if y1 <= y0:
            y1 = min(self._img_h, y0 + 1)
        x0 = max(0, min(self._img_w - 1, x0))
        y0 = max(0, min(self._img_h - 1, y0))
        x1 = max(1, min(self._img_w, x1))
        y1 = max(1, min(self._img_h, y1))
        return (x0, y0, x1, y1)


# -----------------------------
# IMAGE CANVAS WITH CONTEXT MENU + DOUBLE CLICK SELECTION
# -----------------------------
class ImageCanvas(QGraphicsView):
    rect_clicked = Signal(int)
    rect_changed = Signal(int, QRectF)  # idx, new rect in scene coords
    files_dropped = Signal(list)
    canvas_clicked = Signal()
    box_drawn = Signal(QRectF)

    overlay_add_draw_requested = Signal(QPointF)
    overlay_edit_requested = Signal(int)
    overlay_delete_requested = Signal(int)
    overlay_select_requested = Signal(int)

    def __init__(self, tr_func=None):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self._space_panning = False

        # Drag & Drop
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

        self._zoom = 1.0
        self._pixmap_item = None

        self._rects: Dict[int, QGraphicsRectItem] = {}
        self._labels: Dict[int, QGraphicsSimpleTextItem] = {}
        self._selected_idx: Optional[int] = None
        self._bg_color = QColor("#333")

        self._pen_normal = QPen(QColor("#ff3b30"), 2)
        self._pen_selected = QPen(QColor("#0a84ff"), 3)
        self._brush_fill = QBrush(QColor(255, 59, 48, 30))
        self._brush_selected = QBrush(QColor(10, 132, 255, 60))

        self._drop_text = None
        self.tr_func = tr_func

        # draw-box mode
        self._draw_mode = False
        self._draw_start = None
        self._draw_rect_item: Optional[QGraphicsRectItem] = None
        self._pen_draw = QPen(QColor("#00ff7f"), 2)
        self._brush_draw = QBrush(QColor(0, 255, 127, 40))

        # enabled only after OCR finished
        self._overlay_enabled = False

        self._show_drop_hint()

    def _get_view_state(self):
        """Return (transform, center_scene_point, zoom_scalar)"""
        try:
            t = self.transform()
            center = self.mapToScene(self.viewport().rect().center())
            z = float(t.m11())  # assuming uniform scaling
            return t, center, z
        except Exception:
            return None, None, None

    def _restore_view_state(self, t, center, z):
        try:
            if t is not None:
                self.setTransform(t)
            if center is not None:
                self.centerOn(center)
            # keep internal zoom in sync (wheelEvent uses it)
            if z is not None:
                self._zoom = float(z)
            else:
                self._zoom = float(self.transform().m11())
        except Exception:
            pass

    @staticmethod
    def _event_point(event) -> QPoint:
        # Works across PySide6 versions: sometimes event.position() exists, sometimes not.
        try:
            p = event.position()
            return p.toPoint()
        except Exception:
            try:
                return event.pos()
            except Exception:
                return QPoint(0, 0)

    def set_overlay_enabled(self, enabled: bool):
        self._overlay_enabled = bool(enabled)

    def set_theme(self, theme: str):
        if theme == "dark":
            self._bg_color = QColor("#1e1e1e")
            self._pen_normal.setColor(QColor("#ff3b30"))
            self._pen_selected.setColor(QColor("#0a84ff"))
        else:
            self._bg_color = QColor("#ffffff")
            self._pen_normal.setColor(QColor("#d00000"))
            self._pen_selected.setColor(QColor("#0000ff"))
        self.setBackgroundBrush(QBrush(self._bg_color))
        if self._pixmap_item and hasattr(self, "_last_recs"):
            self.refresh_overlays()
        else:
            self._show_drop_hint()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        files = []
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.exists(p):
                files.append(p)
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()

    def start_draw_box_mode(self):
        if not self._overlay_enabled:
            return
        self._draw_mode = True
        self._draw_start = None
        self.setDragMode(QGraphicsView.NoDrag)

    def stop_draw_box_mode(self):
        self._draw_mode = False
        self._draw_start = None
        if self._draw_rect_item is not None:
            try:
                if isValid(self._draw_rect_item) and self._draw_rect_item.scene() is self.scene:
                    self.scene.removeItem(self._draw_rect_item)
            except RuntimeError:
                pass
            self._draw_rect_item = None
        self.setDragMode(QGraphicsView.NoDrag)

    def contextMenuEvent(self, event):
        pos = event.pos()
        item = self.itemAt(pos)

        menu = QMenu(self)
        tr = self.tr_func

        if not self._overlay_enabled:
            disabled = menu.addAction(tr("overlay_only_after_ocr") if tr else "Overlay editing only after OCR.")
            disabled.setEnabled(False)
            menu.exec(event.globalPos())
            return

        if isinstance(item, ResizableRectItem):
            idx = item.idx
            act_edit = menu.addAction(tr("canvas_menu_edit_box") if tr else "Edit overlay box...")
            act_del = menu.addAction(tr("canvas_menu_delete_box") if tr else "Delete overlay box")
            menu.addSeparator()
            act_add_draw = menu.addAction(tr("canvas_menu_add_box_draw") if tr else "Add overlay box (draw)")

            chosen = menu.exec(event.globalPos())
            if not chosen:
                return
            elif chosen == act_edit:
                self.rect_clicked.emit(idx)
                self.select_idx(idx, center=True)
            elif chosen == act_del:
                self.overlay_delete_requested.emit(idx)
            elif chosen == act_add_draw:
                self.overlay_add_draw_requested.emit(self.mapToScene(pos))
            return

        act_add_draw = menu.addAction(tr("canvas_menu_add_box_draw") if tr else "Add overlay box (draw)")
        chosen = menu.exec(event.globalPos())
        if not chosen:
            return
        if chosen == act_add_draw:
            self.overlay_add_draw_requested.emit(self.mapToScene(pos))

    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton:
            sp = self.mapToScene(self._event_point(event))
            self._draw_start = sp

            if self._draw_rect_item is not None:
                try:
                    if isValid(self._draw_rect_item) and self._draw_rect_item.scene() is self.scene:
                        self.scene.removeItem(self._draw_rect_item)
                except RuntimeError:
                    pass
                self._draw_rect_item = None

            self._draw_rect_item = QGraphicsRectItem(QRectF(sp, sp))
            self._draw_rect_item.setPen(self._pen_draw)
            self._draw_rect_item.setBrush(self._brush_draw)
            self._draw_rect_item.setZValue(1000)
            self.scene.addItem(self._draw_rect_item)
            return

        item = self.itemAt(self._event_point(event))
        if isinstance(item, ResizableRectItem):
            self.rect_clicked.emit(item.idx)
            # WICHTIG: nicht return -> Item muss Maus-Events bekommen für Move/Resize
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton and not self._pixmap_item:
            self.canvas_clicked.emit()

        item = self.itemAt(self._event_point(event))
        if isinstance(item, ResizableRectItem):
            self.rect_clicked.emit(item.idx)
            return

        # Klick auf Bild/Leere Fläche: keine Drag/Pan-Aktion starten
        if event.button() == Qt.LeftButton:
            it = self.itemAt(self._event_point(event))

            # falls Pixmap oder Leere Fläche geklickt -> nur deselect + accept
            if (it is None) or (self._pixmap_item is not None and it == self._pixmap_item):
                self.select_idx(None, center=False)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(self._event_point(event))
        if isinstance(item, ResizableRectItem) and event.button() == Qt.LeftButton:
            self.rect_clicked.emit(item.idx)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_mode and self._draw_start and self._draw_rect_item is not None:
            sp = self.mapToScene(self._event_point(event))
            r = QRectF(self._draw_start, sp).normalized()
            if isValid(self._draw_rect_item):
                self._draw_rect_item.setRect(r)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton and self._draw_start and self._draw_rect_item is not None:
            rect = None
            if isValid(self._draw_rect_item):
                rect = self._draw_rect_item.rect().normalized()
            self.stop_draw_box_mode()
            if rect and rect.width() >= 5 and rect.height() >= 5:
                self.box_drawn.emit(rect)
            return
        super().mouseReleaseEvent(event)

    def clear_all(self):
        self.stop_draw_box_mode()
        self.scene.clear()
        self._pixmap_item = None
        self._rects.clear()
        self._labels.clear()
        self._selected_idx = None
        self._drop_text = None
        self.resetTransform()
        self._zoom = 1.0
        self._show_drop_hint()

    def _center_drop_hint_in_view(self):
        if not self._drop_text or self._pixmap_item:
            return

        # Mittelpunkt des sichtbaren Viewports in Scene-Koordinaten
        center = self.mapToScene(self.viewport().rect().center())

        rect = self._drop_text.boundingRect()
        self._drop_text.setPos(center.x() - rect.width() / 2, center.y() - rect.height() / 2)

        # Szene so setzen, dass der Text sicher enthalten ist (sonst kann Qt komisch scrollen)
        br = self.scene.itemsBoundingRect()
        if br.isValid():
            self.setSceneRect(br.adjusted(-50, -50, 50, 50))

    def _show_drop_hint(self):
        if self._pixmap_item:
            return

        font = QFont("Arial", 20)
        font.setItalic(True)
        txt = self.tr_func("drop_hint") if self.tr_func else "Drag & drop files here"

        c = QColor("#aaa") if self._bg_color.lightness() < 128 else QColor("#555")

        # Wenn schon vorhanden: nur aktualisieren
        if self._drop_text and isValid(self._drop_text):
            self._drop_text.setFont(font)
            self._drop_text.setPlainText(txt)
            self._drop_text.setDefaultTextColor(c)
            self._center_drop_hint_in_view()
            return

        # Sonst: neu erzeugen
        self._drop_text = self.scene.addText(txt, font)
        self._drop_text.setDefaultTextColor(c)
        self._center_drop_hint_in_view()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._pixmap_item:
            self._center_drop_hint_in_view()

    def load_pil_image(self, im: Image.Image, preserve_view: bool = False):
        # save current view state BEFORE clearing, if requested
        t = center = z = None
        if preserve_view:
            t, center, z = self._get_view_state()

        self.stop_draw_box_mode()
        self.scene.clear()
        self._pixmap_item = None
        self._rects.clear()
        self._labels.clear()
        self._selected_idx = None
        self._drop_text = None

        # IMPORTANT: don't always reset/fit if we preserve
        if not preserve_view:
            self.resetTransform()
            self._zoom = 1.0

        qim = ImageQt(im.convert("RGB"))
        pix = QPixmap.fromImage(qim)
        self._pixmap_item = self.scene.addPixmap(pix)
        self._pixmap_item.setZValue(0)
        self.setSceneRect(self.scene.itemsBoundingRect())

        if preserve_view and t is not None:
            self._restore_view_state(t, center, z)
        else:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            # sync zoom from actual transform
            try:
                self._zoom = float(self.transform().m11())
            except Exception:
                self._zoom = 1.0

    def refresh_overlays(self):
        if self._pixmap_item and hasattr(self, "_last_recs"):
            for r in list(self._rects.values()):
                try:
                    if isValid(r) and r.scene() is self.scene:
                        self.scene.removeItem(r)
                except RuntimeError:
                    pass
            for l in list(self._labels.values()):
                try:
                    if isValid(l) and l.scene() is self.scene:
                        self.scene.removeItem(l)
                except RuntimeError:
                    pass
            self._rects.clear()
            self._labels.clear()
            self.draw_overlays(self._last_recs)

    def _on_rect_item_changed(self, idx: int, scene_rect: QRectF):
        self.rect_changed.emit(idx, scene_rect)

    def _on_rect_item_double_clicked(self, idx: int):
        self.rect_clicked.emit(idx)

    def draw_overlays(self, recs: List[RecordView]):
        self._last_recs = recs
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)

        for rv in recs:
            if not rv.bbox:
                continue
            x0, y0, x1, y1 = rv.bbox
            rectf = QRectF(x0, y0, x1 - x0, y1 - y0)

            ritem = ResizableRectItem(
                rectf,
                rv.idx,
                self._on_rect_item_changed,
                on_double_clicked=self._on_rect_item_double_clicked
            )
            ritem.setPen(self._pen_normal)
            ritem.setBrush(self._brush_fill)
            ritem.setZValue(10)
            self.scene.addItem(ritem)
            self._rects[rv.idx] = ritem

            lab = QGraphicsSimpleTextItem(str(rv.idx + 1))
            lab.setFont(font)
            c_text = QColor("#fff") if self._bg_color.lightness() < 128 else QColor("#000")
            lab.setBrush(QBrush(c_text))
            lab.setZValue(11)
            lab.setPos(x0, max(0, y0 - 16))
            self.scene.addItem(lab)
            self._labels[rv.idx] = lab

    def select_idx(self, idx: Optional[int], center: bool = True):
        for rect in self._rects.values():
            if isValid(rect):
                rect.setPen(self._pen_normal)
                rect.setBrush(self._brush_fill)
        self._selected_idx = idx
        if idx is not None and idx in self._rects:
            rect = self._rects[idx]
            if isValid(rect):
                rect.setPen(self._pen_selected)
                rect.setBrush(self._brush_selected)
                if center:
                    self.centerOn(rect)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self._apply_zoom(1.25)
        else:
            self._apply_zoom(0.8)

    def _apply_zoom(self, factor: float):
        new_zoom = self._zoom * factor
        if 0.05 <= new_zoom <= 20.0:
            self.scale(factor, factor)
            self._zoom = new_zoom


# -----------------------------
# OCR WORKER
# -----------------------------
class OCRWorker(QThread):
    file_started = Signal(str)
    file_done = Signal(str, str, list, object, list)
    file_error = Signal(str, str)
    progress = Signal(int)
    finished_batch = Signal()
    failed = Signal(str)
    device_resolved = Signal(str)
    gpu_info = Signal(str)

    def __init__(self, job: OCRJob):
        super().__init__()
        self.job = job
        self._device: Optional[torch.device] = None
        self._rec_model: Any = None
        self._seg_model: Any = None
        self._device_label: str = (job.device or "cpu").lower().strip()

    def _resolve_device(self) -> torch.device:
        dev = (self.job.device or "cpu").lower().strip()
        self._device_label = dev

        if dev in ("cuda", "rocm"):
            # Both CUDA and ROCm use torch.cuda backend; ROCm is indicated by torch.version.hip
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return torch.device("cuda")

        if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        self._device_label = "cpu"
        return torch.device("cpu")

    def _emit_gpu_info(self, device: torch.device):
        try:
            if device.type == "cuda":
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
                hip_ver = getattr(torch.version, "hip", None)
                cuda_ver = getattr(torch.version, "cuda", None)

                # If user selected ROCm or HIP is present -> show ROCm/HIP info; otherwise CUDA info
                if self._device_label == "rocm" or hip_ver:
                    extra = []
                    if hip_ver:
                        extra.append(f"HIP {hip_ver}")
                    s = name + (f" ({', '.join(extra)})" if extra else " (ROCm)")
                    self.gpu_info.emit(s)
                else:
                    extra = []
                    if cuda_ver:
                        extra.append(f"CUDA {cuda_ver}")
                    s = name + (f" ({', '.join(extra)})" if extra else " (CUDA)")
                    self.gpu_info.emit(s)

            elif device.type == "mps":
                self.gpu_info.emit("Apple MPS")
            else:
                self.gpu_info.emit("CPU")
        except Exception:
            pass

    def _load_rec_model(self, path: str, device: torch.device):
        try:
            return models.load_any(path, device=device)
        except TypeError:
            return models.load_any(path)

    def _load_seg_model(self, path: str, device: torch.device):
        try:
            return vgsl.TorchVGSLModel.load_model(path, device=device)
        except TypeError:
            return vgsl.TorchVGSLModel.load_model(path)

    def _ensure_models_loaded(self):
        if self._device is None:
            self._device = self._resolve_device()
            # show chosen backend label (cuda/rocm/mps/cpu) + actual torch device
            self.device_resolved.emit(f"{self._device_label} -> {self._device}")
            self._emit_gpu_info(self._device)
        if self._rec_model is None:
            self._rec_model = self._load_rec_model(self.job.recognition_model_path, self._device)
        if getattr(self.job, "segmenter_mode", "blla") == "blla":
            if self._seg_model is None:
                if not self.job.segmentation_model_path:
                    raise ValueError("No baseline model selected.")
                self._seg_model = self._load_seg_model(self.job.segmentation_model_path, self._device)
        else:
            self._seg_model = None  # sicherheitshalber

    @staticmethod
    def _seg_expected_lines(seg: Any) -> Optional[int]:
        for attr in ("lines", "baselines"):
            v = getattr(seg, attr, None)
            if v is not None:
                try:
                    return len(v)
                except Exception:
                    pass
        return None

    def _emit_overall_progress(self, file_idx: int, total_files: int, frac_in_file: float):
        if total_files <= 0:
            self.progress.emit(0)
            return
        frac_in_file = max(0.0, min(1.0, float(frac_in_file)))
        overall = (file_idx + frac_in_file) / float(total_files)
        self.progress.emit(int(overall * 100))

    # -------------------------------------------------------
    # OCRWorker._ocr_one (ersetze deine komplette Funktion damit)
    # -------------------------------------------------------
    def _ocr_one(self, img_path: str, file_idx: int, total_files: int):
        self.file_started.emit(img_path)
        try:
            # --- load image once (RGB) ---
            im = Image.open(img_path).convert("RGB")

            # --- FIX A: zu kleine Bilder hochskalieren (verhindert Baselines < 5px) ---
            min_dim = min(im.size)
            if min_dim < 1200:
                scale = 2 if min_dim >= 700 else 3
                im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.BICUBIC)

            # --- segmentation ---
            if getattr(self.job, "segmenter_mode", "blla") == "pageseg":
                seg = pageseg.segment(im)
            else:
                seg = blla.segment(im, model=self._seg_model)

            # --- FIX B: winzige/kaputte Baselines entfernen (Baseline length below minimum 5px) ---
            try:
                if hasattr(seg, "baselines") and hasattr(seg, "lines") and seg.baselines and seg.lines:
                    new_baselines = []
                    new_lines = []
                    for bl, ln in zip(seg.baselines, seg.lines):
                        if baseline_length(bl) >= 5.0:
                            new_baselines.append(bl)
                            new_lines.append(ln)
                    seg.baselines = new_baselines
                    seg.lines = new_lines
            except Exception:
                pass

            expected = self._seg_expected_lines(seg)

            # --- recognition ---
            kr_records = []
            done = 0

            try:
                for rec in rpred.rpred(self._rec_model, im, seg):
                    kr_records.append(rec)
                    done += 1
                    if expected and expected > 0:
                        self._emit_overall_progress(file_idx, total_files, done / expected)

                    if self.isInterruptionRequested():
                        break
            except Exception as e:
                self.file_error.emit(img_path, str(e))
                return

            if self.isInterruptionRequested():
                return

            # --- sort records in reading order ---
            kr_sorted = sort_records_reading_order(
                kr_records, im.size[0], im.size[1], self.job.reading_direction
            )

            # --- WIDE LINE SPLIT: nur echte 2-Spalten-Zeilen splitten, Header NICHT ---
            def _is_header_like(bb, txt, page_w, page_h):
                x0, y0, x1, y1 = bb
                w = x1 - x0
                cx = (x0 + x1) / 2.0

                if w < 0.72 * page_w:
                    return False
                if abs(cx - (page_w / 2.0)) > 0.20 * page_w:
                    return False
                if y0 > 0.45 * page_h:
                    return False
                if len((txt or "").strip()) > 90:
                    return False
                return True

            two_col_splitter = re.compile(r"\s{4,}")

            record_views: List[RecordView] = []
            lines: List[str] = []
            out_idx = 0
            page_w, page_h = im.size

            for r in kr_sorted:
                pred = getattr(r, "prediction", None)
                if pred is None:
                    continue
                txt = str(pred)
                bb = record_bbox(r)

                if bb:
                    x0, y0, x1, y1 = bb
                    w = x1 - x0
                    if w > int(page_w * 0.80) and not _is_header_like(bb, txt, page_w, page_h):
                        parts = two_col_splitter.split(txt, maxsplit=1)
                        if len(parts) == 2:
                            left_txt, right_txt = map(str.strip, parts)
                            mid = page_w // 2
                            left_bb = clamp_bbox((0, y0, mid, y1), page_w, page_h)
                            right_bb = clamp_bbox((mid, y0, page_w, y1), page_w, page_h)

                            parts_in_order = []
                            if left_bb and left_txt:
                                parts_in_order.append((left_txt, left_bb))
                            if right_bb and right_txt:
                                parts_in_order.append((right_txt, right_bb))

                            rev_x = self.job.reading_direction in (READING_MODES["TB_RL"], READING_MODES["BT_RL"])
                            if rev_x:
                                parts_in_order = list(reversed(parts_in_order))

                            for txt_part, bb_part in parts_in_order:
                                record_views.append(RecordView(out_idx, txt_part, bb_part))
                                lines.append(txt_part)
                                out_idx += 1
                            continue

                record_views.append(RecordView(out_idx, txt, bb))
                lines.append(txt)
                out_idx += 1

            self._emit_overall_progress(file_idx, total_files, 1.0)
            text = "\n".join(lines).strip()
            self.file_done.emit(img_path, text, kr_sorted, im, record_views)

        except Exception as e:
            self.file_error.emit(img_path, str(e))

    def run(self):
        try:
            if not os.path.exists(self.job.recognition_model_path):
                raise ValueError("Recognition model not found.")
            if getattr(self.job, "segmenter_mode", "blla") == "blla":
                if not os.path.exists(self.job.segmentation_model_path or ""):
                    raise ValueError("Baseline model not found.")

            self._ensure_models_loaded()

            total = len(self.job.input_paths)
            for i, path in enumerate(self.job.input_paths):
                if self.isInterruptionRequested():
                    break
                self._emit_overall_progress(i, total, 0.0)
                self._ocr_one(path, i, total)

            self.progress.emit(100)
            self.finished_batch.emit()
        except Exception as e:
            self.failed.emit(str(e))


# -----------------------------
# EXPORT DIALOGS
# -----------------------------
class ExportModeDialog(QDialog):
    def __init__(self, tr, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("export_choose_mode_title"))
        self.choice = None

        lay = QVBoxLayout(self)
        self.rb_all = QRadioButton(tr("export_mode_all"))
        self.rb_sel = QRadioButton(tr("export_mode_selected"))
        self.rb_all.setChecked(True)

        lay.addWidget(self.rb_all)
        lay.addWidget(self.rb_sel)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def accept(self):
        self.choice = "all" if self.rb_all.isChecked() else "selected"
        super().accept()


class ExportSelectFilesDialog(QDialog):
    def __init__(self, tr, items: List[TaskItem], parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("export_select_files_title"))
        self.selected_paths: List[str] = []

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(tr("export_select_files_hint")))

        self.listw = QListWidget()
        self.listw.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for it in items:
            li = QListWidgetItem(it.display_name)
            li.setData(Qt.UserRole, it.path)
            self.listw.addItem(li)

        lay.addWidget(self.listw)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_ok(self):
        paths = [i.data(Qt.UserRole) for i in self.listw.selectedItems()]
        self.selected_paths = [p for p in paths if p]
        self.accept()

# -----------------------------
# MAIN WINDOW
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1600, 900)
        self.setAcceptDrops(True)

        self.current_lang = "de"
        self.log_lang = self._detect_system_lang()
        self.reading_direction = READING_MODES["TB_LR"]
        self.device_str = "cpu"
        self.show_overlay = True
        self.model_path = ""
        self.seg_model_path = ""
        self.current_export_dir = ""
        self.current_theme = "bright"

        # queue columns dynamic ratio
        self.queue_col_ratio = 0.75
        self._resizing_cols = False

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self._tr("status_ready"))

        self.worker: Optional[OCRWorker] = None
        self.queue_items: List[TaskItem] = []

        # Canvas
        self.canvas = ImageCanvas(tr_func=self._tr)
        self.canvas.rect_clicked.connect(self.on_rect_clicked)
        self.canvas.rect_changed.connect(self.on_overlay_rect_changed)
        self.canvas.files_dropped.connect(self.add_files_to_queue)
        self.canvas.box_drawn.connect(self.on_box_drawn)

        self.canvas.overlay_add_draw_requested.connect(self.on_canvas_add_box_draw)
        self.canvas.overlay_edit_requested.connect(self.on_canvas_edit_box)
        self.canvas.overlay_delete_requested.connect(self.on_canvas_delete_box)
        self.canvas.overlay_select_requested.connect(self.on_canvas_select_line)

        # Queue Table
        self.queue_table = DropQueueTable()
        self.queue_table.setColumnCount(2)
        self.queue_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.queue_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.queue_table.itemChanged.connect(self.on_item_changed)
        self.queue_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_table.customContextMenuRequested.connect(self.queue_context_menu)
        self.queue_table.cellDoubleClicked.connect(self.on_queue_double_click)
        self.queue_table.delete_pressed.connect(self._delete_queue_via_key)
        self.queue_table.files_dropped.connect(self.add_files_to_queue)
        self.queue_table.table_resized.connect(self._fit_queue_columns_exact)

        header = self.queue_table.horizontalHeader()
        header.setSectionsMovable(False)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.sectionResized.connect(self._on_queue_header_resized)
        self.queue_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Queue hint overlay
        self.queue_hint = QLabel(self._tr("queue_drop_hint"), self.queue_table.viewport())
        self.queue_hint.setAlignment(Qt.AlignCenter)
        self.queue_hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.queue_hint.setStyleSheet("color: rgba(180,180,180,180); font-style: italic;")
        self.queue_hint.hide()

        # Lines list
        self.list_lines = LinesListWidget()
        self.list_lines.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_lines.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.list_lines.currentRowChanged.connect(self.on_line_selected)
        self.list_lines.itemChanged.connect(self.on_line_item_edited)
        self.list_lines.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_lines.customContextMenuRequested.connect(self.lines_context_menu)
        self.list_lines.delete_pressed.connect(self._delete_current_line_via_key)
        self.list_lines.reorder_committed.connect(self.on_lines_reordered)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # idle = normale Prozentanzeige
        self.progress_bar.setValue(0)

        # Log (unter Queue)
        self.log_visible = False
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)  # damit es nicht endlos wächst
        self.log_edit.hide()

        self.lbl_queue = QLabel(self._tr("lbl_queue"))
        self.lbl_lines = QLabel(self._tr("lbl_lines"))

        # Toolbar Actions
        self.act_add = QAction(QIcon.fromTheme("document-open"), self._tr("act_add_files"), self)
        self.act_add.triggered.connect(self.choose_files)

        self.act_clear = QAction(QIcon.fromTheme("edit-clear"), self._tr("act_clear_queue"), self)
        self.act_clear.triggered.connect(self.clear_queue)

        self.act_play = QAction(QIcon.fromTheme("media-playback-start"), self._tr("act_start_ocr"), self)
        self.act_play.triggered.connect(self.start_ocr)

        self.act_stop = QAction(QIcon.fromTheme("media-playback-stop"), self._tr("act_stop_ocr"), self)
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self.stop_ocr)

        self.act_re_ocr = QAction(QIcon.fromTheme("view-refresh"), self._tr("act_re_ocr"), self)
        self.act_re_ocr.setToolTip(self._tr("act_re_ocr_tip"))
        self.act_re_ocr.triggered.connect(self.reprocess_selected)

        self.act_toggle_log = QAction(QIcon.fromTheme("document-preview"), self._tr("log_toggle_show"), self)
        self.act_toggle_log.setCheckable(True)
        self.act_toggle_log.setChecked(False)
        self.act_toggle_log.toggled.connect(self.toggle_log_area)

        # Undo / Redo actions
        self.act_undo = QAction(self._tr("act_undo"), self)
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self.undo)

        self.act_redo = QAction(self._tr("act_redo"), self)
        self.act_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_redo.triggered.connect(self.redo)

        self.addAction(self.act_undo)
        self.addAction(self.act_redo)

        self.btn_rec_model = QPushButton(self._tr("dlg_choose_rec") + " -")
        self.btn_rec_model.setIcon(QIcon.fromTheme("document-open"))
        self.btn_rec_model.clicked.connect(self.choose_rec_model)

        self.btn_seg_model = QPushButton(self._tr("dlg_choose_seg") + " -")
        self.btn_seg_model.setIcon(QIcon.fromTheme("document-open"))
        self.btn_seg_model.clicked.connect(self.choose_seg_model)

        self._pending_box_for_row: Optional[int] = None
        self._pending_new_line_box: bool = False

        self._auto_select_best_device()
        self._init_ui()
        self._init_menu()
        self.apply_theme("bright")
        self.retranslate_ui()

        QTimer.singleShot(0, self._fit_queue_columns_exact)
        QTimer.singleShot(0, self._update_queue_hint)
        QTimer.singleShot(0, self._refresh_hw_menu_availability)

        self.canvas.set_overlay_enabled(False)
        self._log(self._tr_log("log_started"))

    # -----------------------------
    # Translation
    # -----------------------------
    def _tr(self, key: str, *args):
        txt = TRANSLATIONS.get(self.current_lang, TRANSLATIONS["de"]).get(key, key)
        if args:
            return txt.format(*args)
        return txt

    def _detect_system_lang(self) -> str:
        # e.g. "de_DE", "en_US", "fr_FR"
        name = QLocale.system().name().lower()
        if name.startswith("de"):
            return "de"
        if name.startswith("fr"):
            return "fr"
        return "en"

    def _tr_in(self, lang: str, key: str, *args):
        txt = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
        if args:
            return txt.format(*args)
        return txt

    def _tr_log(self, key: str, *args):
        return self._tr_in(self.log_lang, key, *args)

    def _delete_queue_via_key(self):
        # Löscht selektierte Zeilen und setzt danach die Vorschau zurück
        self.delete_selected_queue_items(reset_preview=True)

    # -----------------------------
    # Undo helpers (snapshots)
    # -----------------------------
    @staticmethod
    def _snapshot_recs(recs: List[RecordView]) -> List[Tuple[str, Optional[Tuple[int, int, int, int]]]]:
        return [(rv.text, rv.bbox) for rv in recs]

    @staticmethod
    def _restore_recs(snapshot: List[Tuple[str, Optional[Tuple[int, int, int, int]]]]) -> List[RecordView]:
        recs: List[RecordView] = []
        for i, (t, bb) in enumerate(snapshot):
            recs.append(RecordView(i, t, bb))
        return recs

    def _push_undo(self, task: TaskItem):
        if not task.results:
            return
        _, _, _, recs = task.results
        sel = self.list_lines.currentRow()
        snap: UndoSnapshot = (self._snapshot_recs(recs), int(sel) if sel is not None else -1)
        task.undo_stack.append(snap)
        if len(task.undo_stack) > 300:
            task.undo_stack.pop(0)
        task.redo_stack.clear()

    def _apply_snapshot(self, task: TaskItem, snap: UndoSnapshot):
        if not task.results:
            return
        text, kr_records, im, _recs = task.results
        state, sel = snap
        recs = self._restore_recs(state)
        new_text = "\n".join([r.text for r in recs]).strip()
        task.results = (new_text, kr_records, im, recs)
        task.edited = True
        keep_row = sel if sel is not None else -1
        if keep_row < 0:
            keep_row = 0 if recs else None
        self._sync_ui_after_recs_change(task, keep_row=keep_row)

    def undo(self):
        task = self._current_task()
        if not task or task.status != STATUS_DONE or not task.results:
            self.status_bar.showMessage(self._tr("undo_nothing"))
            return
        if not task.undo_stack:
            self.status_bar.showMessage(self._tr("undo_nothing"))
            return

        _, _, _, recs = task.results
        cur_sel = self.list_lines.currentRow()
        task.redo_stack.append((self._snapshot_recs(recs), int(cur_sel) if cur_sel is not None else -1))

        snap = task.undo_stack.pop()
        self._apply_snapshot(task, snap)

    def redo(self):
        task = self._current_task()
        if not task or task.status != STATUS_DONE or not task.results:
            self.status_bar.showMessage(self._tr("redo_nothing"))
            return
        if not task.redo_stack:
            self.status_bar.showMessage(self._tr("redo_nothing"))
            return

        _, _, _, recs = task.results
        cur_sel = self.list_lines.currentRow()
        task.undo_stack.append((self._snapshot_recs(recs), int(cur_sel) if cur_sel is not None else -1))

        snap = task.redo_stack.pop()
        self._apply_snapshot(task, snap)

    def _auto_select_best_device(self):
        caps = self._gpu_capabilities()

        # Priorität: CUDA (echtes CUDA build) > ROCm/HIP > MPS > CPU
        for dev in ("cuda", "rocm", "mps", "cpu"):
            ok, _ = caps.get(dev, (False, ""))
            if ok:
                self.device_str = dev
                break

    # -----------------------------
    # UI
    # -----------------------------
    def _init_ui(self):
        self.toolbar = QToolBar(self._tr("toolbar_main"))
        self.addToolBar(self.toolbar)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.toolbar.addAction(self.act_add)
        self.toolbar.addAction(self.act_clear)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_play)
        self.toolbar.addAction(self.act_stop)
        self.toolbar.addAction(self.act_re_ocr)

        # NEU: Log Button direkt rechts neben Wiederholen
        self.toolbar.addAction(self.act_toggle_log)

        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.btn_rec_model)
        self.toolbar.addWidget(self.btn_seg_model)

        self._make_toolbar_buttons_pushy()  # <--- NEU

        right = QVBoxLayout()
        right.addWidget(self.lbl_queue)
        right.addWidget(self.queue_table, 2)

        # NEU: Logbereich unter der Queue
        right.addWidget(self.log_edit, 1)

        right.addWidget(self.progress_bar)
        right.addWidget(self.lbl_lines)
        right.addWidget(self.list_lines, 3)


        right_widget = QLabel()
        right_widget.setLayout(right)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.canvas)
        left_widget = QLabel()
        left_widget.setLayout(left_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([1000, 500])
        self.splitter.splitterMoved.connect(lambda *_: self._fit_queue_columns_exact())
        self.setCentralWidget(self.splitter)

    def _make_toolbar_buttons_pushy(self):
        # Alle QToolButtons, die QToolBar für QAction erstellt
        for b in self.toolbar.findChildren(QToolButton):
            b.setAutoRaise(False)  # wichtig: sonst wirkt es oft "flat"
            b.setCursor(Qt.PointingHandCursor)

        # Auch die Modell-Buttons
        self.btn_rec_model.setCursor(Qt.PointingHandCursor)
        self.btn_seg_model.setCursor(Qt.PointingHandCursor)

    def _init_menu(self):
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)

        hw_group = QActionGroup(self)
        hw_group.setExclusive(True)

        read_group = QActionGroup(self)
        read_group.setExclusive(True)

        menubar = self.menuBar()

        self.file_menu = menubar.addMenu(self._tr("menu_file"))
        self.edit_menu = menubar.addMenu(self._tr("menu_edit"))

        self.edit_menu.addAction(self.act_undo)
        self.edit_menu.addAction(self.act_redo)

        self.edit_menu.addSeparator()
        self.act_export_log = QAction(self._tr("menu_export_log"), self)
        self.act_export_log.triggered.connect(self.export_log_txt)
        self.edit_menu.addAction(self.act_export_log)

        self.act_add_files = QAction(self._tr("act_add_files"), self)
        self.act_add_files.triggered.connect(self.choose_files)
        self.file_menu.addAction(self.act_add_files)

        self.file_menu.addSeparator()
        self.export_menu = self.file_menu.addMenu(self._tr("menu_export"))

        self.formats = [
            ("Text (.txt)", "txt"),
            ("CSV (.csv)", "csv"),
            ("JSON (.json)", "json"),
            ("ALTO (.xml)", "alto"),
            ("hOCR (.html)", "hocr"),
            ("PDF (.pdf)", "pdf")
        ]
        for name, fmt in self.formats:
            act = QAction(name, self)
            act.triggered.connect(lambda checked, f=fmt: self.export_flow(f))
            self.export_menu.addAction(act)

        self.file_menu.addSeparator()
        self.act_exit = QAction(self._tr("menu_exit"), self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)
        self.file_menu.addAction(self.act_exit)

        self.models_menu = menubar.addMenu(self._tr("menu_models"))

        # Menüeinträge zeigen immer den aktuell geladenen Namen
        self.act_rec = QAction(f"{self._tr('dlg_choose_rec')}-", self)
        self.act_rec.triggered.connect(self.choose_rec_model)
        self.models_menu.addAction(self.act_rec)

        self.act_seg = QAction(f"{self._tr('dlg_choose_seg')}-", self)
        self.act_seg.triggered.connect(self.choose_seg_model)
        self.models_menu.addAction(self.act_seg)

        self.models_menu.addSeparator()
        self.act_download = QAction(self._tr("act_download_model"), self)
        self.act_download.triggered.connect(self.open_download_link)
        self.models_menu.addAction(self.act_download)

        self.options_menu = menubar.addMenu(self._tr("menu_options"))

        # Languages
        self.lang_menu = self.options_menu.addMenu(self._tr("menu_languages"))
        lang_group = QActionGroup(self)
        for key, code in [("lang_de", "de"), ("lang_en", "en"), ("lang_fr", "fr")]:
            act = QAction(self._tr(key), self)
            act.setCheckable(True)
            if code == self.current_lang:
                act.setChecked(True)
            act.triggered.connect(lambda checked, c=code: self.set_language(c))
            lang_group.addAction(act)
            self.lang_menu.addAction(act)

        # HW menu
        self.options_menu.addSeparator()
        self.hw_menu = self.options_menu.addMenu(self._tr("menu_hw"))
        hw_group = QActionGroup(self)
        self.hw_actions: Dict[str, QAction] = {}
        for key, dev in [("hw_cpu", "cpu"), ("hw_cuda", "cuda"), ("hw_rocm", "rocm"), ("hw_mps", "mps")]:
            act = QAction(self._tr(key), self)
            act.setCheckable(True)
            if dev == self.device_str:
                act.setChecked(True)
            act.triggered.connect(lambda checked, d=dev: self.set_device(d))
            hw_group.addAction(act)
            self.hw_menu.addAction(act)
            self.hw_actions[dev] = act

        # Reading direction
        self.options_menu.addSeparator()
        self.reading_menu = self.options_menu.addMenu(self._tr("menu_reading"))
        read_group = QActionGroup(self)
        self.read_actions: List[QAction] = []
        for key, mode in [
            ("reading_tb_lr", READING_MODES["TB_LR"]),
            ("reading_tb_rl", READING_MODES["TB_RL"]),
            ("reading_bt_lr", READING_MODES["BT_LR"]),
            ("reading_bt_rl", READING_MODES["BT_RL"]),
        ]:
            act = QAction(self._tr(key), self)
            act.setCheckable(True)
            if mode == self.reading_direction:
                act.setChecked(True)
            act.triggered.connect(lambda checked, m=mode: self.set_reading_direction(m))
            read_group.addAction(act)
            self.reading_menu.addAction(act)
            self.read_actions.append(act)

        # Overlay
        self.options_menu.addSeparator()
        self.act_overlay = QAction(self._tr("act_overlay_show"), self)
        self.act_overlay.setCheckable(True)
        self.act_overlay.setChecked(True)
        self.act_overlay.toggled.connect(self._on_overlay_toggled)
        self.options_menu.addAction(self.act_overlay)

        # Theme
        self.options_menu.addSeparator()
        self.theme_menu = self.options_menu.addMenu(self._tr("menu_appearance"))
        self.act_theme_bright = QAction(self._tr("theme_bright"), self)
        self.act_theme_bright.triggered.connect(lambda: self.apply_theme("bright"))
        self.theme_menu.addAction(self.act_theme_bright)
        self.act_theme_dark = QAction(self._tr("theme_dark"), self)
        self.act_theme_dark.triggered.connect(lambda: self.apply_theme("dark"))
        self.theme_menu.addAction(self.act_theme_dark)

        if self.device_str in self.hw_actions:
            self.hw_actions[self.device_str].setChecked(True)

    # -----------------------------
    # Queue columns
    # -----------------------------
    def _fit_queue_columns_exact(self):
        if self._resizing_cols:
            return
        self._resizing_cols = True
        try:
            vw = max(1, self.queue_table.viewport().width())
            w0 = int(vw * float(self.queue_col_ratio))
            w1 = vw - w0

            min0, min1 = 80, 60
            if w0 < min0:
                w0 = min0
                w1 = max(min1, vw - w0)
            if w1 < min1:
                w1 = min1
                w0 = max(min0, vw - w1)

            if w0 + w1 != vw:
                w1 = max(min1, vw - w0)

            self.queue_table.setColumnWidth(0, w0)
            self.queue_table.setColumnWidth(1, w1)

            if vw > 0:
                self.queue_col_ratio = max(0.1, min(0.9, w0 / float(vw)))

            self._update_queue_hint()
        finally:
            self._resizing_cols = False

    def _on_queue_header_resized(self, logicalIndex: int, oldSize: int, newSize: int):
        if self._resizing_cols:
            return
        w0 = self.queue_table.columnWidth(0)
        w1 = self.queue_table.columnWidth(1)
        total = max(1, w0 + w1)
        self.queue_col_ratio = max(0.1, min(0.9, w0 / float(total)))
        self._fit_queue_columns_exact()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_queue_columns_exact()

    def _update_queue_hint(self):
        empty = (self.queue_table.rowCount() == 0)
        self.queue_hint.setText(self._tr("queue_drop_hint"))
        self.queue_hint.resize(self.queue_table.viewport().size())
        self.queue_hint.move(0, 0)
        self.queue_hint.setVisible(empty)

    # -----------------------------
    # Progress helpers
    # -----------------------------
    def _set_progress_busy(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 0)  # busy animation

    def _set_progress_idle(self, value: int = 0):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, min(100, int(value))))

    def on_progress_update(self, v: int):
        v = max(0, min(100, int(v)))

        # Solange wir in busy (0,0) sind und v noch 0 ist -> Animation bleibt
        if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
            if v > 0:
                self.progress_bar.setRange(0, 100)  # sobald >0 -> normale Prozentanzeige
            else:
                return  # busy-mode ignoriert setValue sowieso; Animation bleibt

        self.progress_bar.setValue(v)

    # -----------------------------
    # Theme
    # -----------------------------
    def apply_theme(self, theme: str):
        self.current_theme = theme
        pal = QPalette()
        conf = THEMES[theme]

        pal.setColor(QPalette.Window, QColor(conf["bg"]))
        pal.setColor(QPalette.WindowText, QColor(conf["fg"]))
        pal.setColor(QPalette.Base, conf["table_base"])
        pal.setColor(QPalette.AlternateBase, conf["table_base"].lighter(110))
        pal.setColor(QPalette.ToolTipBase, Qt.white)
        pal.setColor(QPalette.ToolTipText, Qt.white)
        pal.setColor(QPalette.Text, QColor(conf["fg"]))
        pal.setColor(QPalette.Button, conf["table_base"].lighter(110))
        pal.setColor(QPalette.ButtonText, QColor(conf["fg"]))
        pal.setColor(QPalette.BrightText, Qt.red)
        pal.setColor(QPalette.Link, QColor(42, 130, 218))
        pal.setColor(QPalette.Highlight, QColor(42, 130, 218))
        pal.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.instance().setPalette(pal)

        self.canvas.set_theme(theme)

        txt = conf["toolbar_text"]
        border = conf["toolbar_border"]
        if theme == "dark":
            self.toolbar.setStyleSheet(
                f"""
                QToolBar {{
                    background: {conf["bg"]};
                    spacing: 6px;
                }}

                QToolButton {{
                    color: #000;
                    border: 1px solid rgba(255,255,255,0.85);
                    border-radius: 6px;
                    padding: 4px 8px;
                    background: rgba(255,255,255,0.92);
                }}
                QToolButton:hover {{
                    background: rgba(255,255,255,0.98);
                    border-color: rgba(255,255,255,0.98);
                }}
                QToolButton:pressed {{
                    background: rgba(235,235,235,1.0);
                }}
                
                QToolButton:checked {{
                    background: rgba(200,230,255,1.0);
                    border-color: rgba(120,190,255,1.0);
                }}

                QPushButton {{
                    color: #000;
                    border: 1px solid rgba(255,255,255,0.85);
                    border-radius: 6px;
                    padding: 4px 8px;
                    background: rgba(255,255,255,0.92);
                }}
                QPushButton:hover {{
                    background: rgba(255,255,255,0.98);
                    border-color: rgba(255,255,255,0.98);
                }}
                QPushButton:pressed {{
                    background: rgba(235,235,235,1.0);
                }}
                """
            )
        else:
            self.toolbar.setStyleSheet(
                """
                QToolBar {
                    spacing: 6px;
                }

                QToolButton, QPushButton {
                    border: 1px solid rgba(0,0,0,0.25);
                    border-radius: 6px;
                    padding: 4px 8px;
                    background: rgba(255,255,255,0.90);
                }

                QToolButton:hover, QPushButton:hover {
                    background: rgba(255,255,255,1.0);
                    border-color: rgba(0,0,0,0.35);
                }

                /* Push/Release Feedback */
                QToolButton:pressed, QPushButton:pressed {
                    background: rgba(230,230,230,1.0);
                    border-color: rgba(0,0,0,0.45);
                    padding-left: 9px;   /* “depressed” effect */
                    padding-top: 5px;
                }

                QToolButton:checked {
                    background: rgba(42,130,218,0.20);
                    border-color: rgba(42,130,218,0.45);
                }
                """
            )

    # -----------------------------
    # Language / reading
    # -----------------------------
    def set_language(self, lang):
        self.current_lang = lang
        self.retranslate_ui()
        self._refresh_hw_menu_availability()

    def _update_models_menu_labels(self):
        rec_name = os.path.basename(self.model_path) if self.model_path else "-"
        seg_name = os.path.basename(self.seg_model_path) if self.seg_model_path else "-"

        # Reiter "Modelle" (Menü) aktualisieren
        self.act_rec.setText(f"{self._tr('dlg_choose_rec')}{rec_name}")
        self.act_seg.setText(f"{self._tr('dlg_choose_seg')}{seg_name}")

    def set_reading_direction(self, mode):
        self.reading_direction = mode

    def retranslate_ui(self):
        self.setWindowTitle(self._tr("app_title"))
        self.file_menu.setTitle(self._tr("menu_file"))
        self.edit_menu.setTitle(self._tr("menu_edit"))
        self.models_menu.setTitle(self._tr("menu_models"))
        self.options_menu.setTitle(self._tr("menu_options"))
        self.lang_menu.setTitle(self._tr("menu_languages"))
        self.hw_menu.setTitle(self._tr("menu_hw"))
        self.theme_menu.setTitle(self._tr("menu_appearance"))
        self.export_menu.setTitle(self._tr("menu_export"))
        self.reading_menu.setTitle(self._tr("menu_reading"))

        self.act_export_log.setText(self._tr("menu_export_log"))

        self.act_undo.setText(self._tr("act_undo"))
        self.act_redo.setText(self._tr("act_redo"))

        self.act_add_files.setText(self._tr("act_add_files"))
        self.act_exit.setText(self._tr("menu_exit"))
        self.act_download.setText(self._tr("act_download_model"))
        self.act_overlay.setText(self._tr("act_overlay_show"))
        self.act_theme_bright.setText(self._tr("theme_bright"))
        self.act_theme_dark.setText(self._tr("theme_dark"))

        self.act_add.setText(self._tr("act_add_files"))
        self.act_clear.setText(self._tr("act_clear_queue"))
        self.act_play.setText(self._tr("act_start_ocr"))
        self.act_stop.setText(self._tr("act_stop_ocr"))
        self.act_re_ocr.setText(self._tr("act_re_ocr"))
        self.act_re_ocr.setToolTip(self._tr("act_re_ocr_tip"))

        self.lbl_queue.setText(self._tr("lbl_queue"))
        self.lbl_lines.setText(self._tr("lbl_lines"))
        self.queue_table.setHorizontalHeaderLabels([self._tr("col_file"), self._tr("col_status")])

        if self.model_path:
            self.btn_rec_model.setText(f"{self._tr('dlg_choose_rec')}{os.path.basename(self.model_path)}")
        else:
            self.btn_rec_model.setText(f"{self._tr('dlg_choose_rec')}-")

        if self.seg_model_path:
            self.btn_seg_model.setText(f"{self._tr('dlg_choose_seg')}{os.path.basename(self.seg_model_path)}")
        else:
            self.btn_seg_model.setText(f"{self._tr('dlg_choose_seg')}-")

        mapping = {"cpu": "hw_cpu", "cuda": "hw_cuda", "rocm": "hw_rocm", "mps": "hw_mps"}
        for dev, key in mapping.items():
            if dev in self.hw_actions:
                self.hw_actions[dev].setText(self._tr(key))

        read_keys = ["reading_tb_lr", "reading_tb_rl", "reading_bt_lr", "reading_bt_rl"]
        for act, key in zip(self.read_actions, read_keys):
            act.setText(self._tr(key))

        self._retranslate_queue_rows()
        self._update_queue_hint()
        self.canvas._show_drop_hint()
        self._update_models_menu_labels()

    def _retranslate_queue_rows(self):
        for it in self.queue_items:
            self._update_queue_row(it.path)

    # -----------------------------
    # GPU detection + availability
    # -----------------------------
    def _gpu_capabilities(self) -> Dict[str, Tuple[bool, str]]:
        caps: Dict[str, Tuple[bool, str]] = {"cpu": (True, "CPU")}

        cuda_avail = torch.cuda.is_available() and torch.cuda.device_count() > 0
        cuda_name = ""
        if cuda_avail:
            try:
                cuda_name = torch.cuda.get_device_name(0)
            except Exception:
                cuda_name = "GPU"

        hip_ver = getattr(torch.version, "hip", None)
        cuda_ver = getattr(torch.version, "cuda", None)

        # ROCm (HIP) availability
        rocm_avail = cuda_avail and (hip_ver is not None)
        rocm_details = ""
        if rocm_avail:
            rocm_details = f"{cuda_name} (HIP {hip_ver})" if cuda_name else f"HIP {hip_ver}"

        # CUDA availability (real CUDA build)
        cuda_true = cuda_avail and (cuda_ver is not None)
        cuda_true_details = ""
        if cuda_true:
            cuda_true_details = f"{cuda_name} (CUDA {cuda_ver})" if cuda_name else f"CUDA {cuda_ver}"

        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        mps_details = "Apple MPS" if mps_avail else ""

        caps["cuda"] = (cuda_true, cuda_true_details if cuda_true_details else "CUDA")
        caps["rocm"] = (rocm_avail, rocm_details if rocm_details else "ROCm")
        caps["mps"] = (mps_avail, mps_details if mps_details else "MPS")
        return caps

    def _refresh_hw_menu_availability(self):
        caps = self._gpu_capabilities()
        for dev, act in self.hw_actions.items():
            ok, detail = caps.get(dev, (False, ""))
            if dev == "cpu":
                act.setEnabled(True)
                act.setToolTip("CPU")
                continue
            act.setEnabled(ok)
            act.setToolTip(detail if detail else ("Not available" if self.current_lang == "en" else "Nicht verfügbar"))

        if self.device_str != "cpu":
            ok, _ = caps.get(self.device_str, (False, ""))
            if not ok:
                self.device_str = "cpu"
                if "cpu" in self.hw_actions:
                    self.hw_actions["cpu"].setChecked(True)

    def set_device(self, dev: str):
        caps = self._gpu_capabilities()
        ok, detail = caps.get(dev, (False, ""))
        if not ok:
            QMessageBox.warning(self, self._tr("warn_title"), self._tr("msg_hw_not_available"))
            dev = "cpu"
            ok, detail = caps.get("cpu", (True, "CPU"))

        self.device_str = dev
        if dev in self.hw_actions:
            self.hw_actions[dev].setChecked(True)

        if detail:
            self.status_bar.showMessage(self._tr("msg_detected_gpu", detail))
        else:
            label_key = {
                "cpu": "msg_device_cpu",
                "cuda": "msg_device_cuda",
                "rocm": "msg_device_rocm",
                "mps": "msg_device_mps",
            }.get(dev, "msg_device_cpu")
            self.status_bar.showMessage(self._tr("msg_device", self._tr(label_key)))

    # -----------------------------
    # Drag & Drop on MainWindow
    # -----------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        files = []
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.exists(p):
                files.append(p)
        if files:
            self.add_files_to_queue(files)
            event.acceptProposedAction()
        else:
            event.ignore()

    # -----------------------------
    # Queue + preview
    # -----------------------------
    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, self._tr("dlg_load_img"), "", self._tr("dlg_filter_img"))
        if files:
            self.add_files_to_queue(files)

    def add_files_to_queue(self, paths: List[str]):
        added_any = False
        last_added = None
        added_count = 0

        for p in paths:
            if not p or not os.path.exists(p):
                continue
            if any(it.path == p for it in self.queue_items):
                continue
            self._add_file_to_queue_single(p)
            added_any = True
            last_added = p
            added_count += 1

        if added_any and last_added:
            self.preview_image(last_added)

        if added_any:
            self._log(self._tr_log("log_added_files", added_count))

        self._fit_queue_columns_exact()
        self._update_queue_hint()

    def _add_file_to_queue_single(self, path: str):
        item = TaskItem(path=path, display_name=os.path.basename(path))
        self.queue_items.append(item)

        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)

        name_item = QTableWidgetItem(item.display_name)
        name_item.setData(Qt.UserRole, path)
        name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)

        status_item = QTableWidgetItem(f"{STATUS_ICONS[STATUS_WAITING]} {self._tr('status_waiting')}")
        status_item.setFlags(status_item.flags() ^ Qt.ItemIsEditable)

        self.queue_table.setItem(row, 0, name_item)
        self.queue_table.setItem(row, 1, status_item)
        self.queue_table.selectRow(row)

    def on_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            row = item.row()
            path_item = self.queue_table.item(row, 0)
            if not path_item:
                return
            path = path_item.data(Qt.UserRole)
            task_item = next((t for t in self.queue_items if t.path == path), None)
            if task_item:
                task_item.display_name = item.text()

    def open_download_link(self):
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(ZENODO_URL))

    def queue_context_menu(self, pos):
        menu = QMenu()
        rename_act = menu.addAction(self._tr("act_rename"))
        delete_act = menu.addAction(self._tr("act_delete"))

        action = menu.exec(self.queue_table.viewport().mapToGlobal(pos))
        if not action:
            return

        item = self.queue_table.itemAt(pos)
        if not item:
            return

        row = item.row()
        path = self.queue_table.item(row, 0).data(Qt.UserRole)
        task = next((t for t in self.queue_items if t.path == path), None)

        if action == rename_act and task:
            new_name, ok = QInputDialog.getText(
                self,
                self._tr("dlg_title_rename"),
                self._tr("dlg_label_name"),
                text=task.display_name
            )
            if ok:
                task.display_name = new_name
                self.queue_table.item(row, 0).setText(new_name)

        elif action == delete_act:
            self.delete_selected_queue_items()

    def delete_selected_queue_items(self, reset_preview: bool = False):
        rows = sorted(set(index.row() for index in self.queue_table.selectedIndexes()), reverse=True)
        if not rows:
            return

        current_preview_path = None
        if self.queue_table.currentRow() >= 0:
            current_preview_path = self.queue_table.item(self.queue_table.currentRow(), 0).data(Qt.UserRole)

        removed_paths = []
        for row in rows:
            path = self.queue_table.item(row, 0).data(Qt.UserRole)
            removed_paths.append(path)
            self.queue_items = [i for i in self.queue_items if i.path != path]
            self.queue_table.removeRow(row)

        if len(self.queue_items) == 0:
            self.canvas.clear_all()
            self.canvas.set_overlay_enabled(False)
            self.list_lines.clear()
            self._set_progress_idle(0)

        else:
            if current_preview_path and current_preview_path in removed_paths:
                self.queue_table.selectRow(0)
                p = self.queue_table.item(0, 0).data(Qt.UserRole)
                self.preview_image(p)

        self._fit_queue_columns_exact()
        self._update_queue_hint()

        if reset_preview:
            # Vorschau zurücksetzen wie beim Programmstart
            self.canvas.clear_all()
            self.canvas.set_overlay_enabled(False)
            self.list_lines.clear()
            self._set_progress_idle(0)

    def clear_queue(self):
        self.queue_items.clear()
        self.queue_table.setRowCount(0)
        self.canvas.clear_all()
        self.canvas.set_overlay_enabled(False)
        self.list_lines.clear()
        self._set_progress_idle(0)
        self._fit_queue_columns_exact()
        self._update_queue_hint()
        self._log(self._tr_log("log_queue_cleared"))

    def preview_image(self, path: str):
        try:
            im = Image.open(path)
            self.canvas.load_pil_image(im)
            self.list_lines.clear()

            item = next((i for i in self.queue_items if i.path == path), None)
            if item and item.status == STATUS_DONE and item.results:
                self.load_results(path)
            else:
                self.canvas.set_overlay_enabled(False)
        except Exception as e:
            QMessageBox.warning(self, self._tr("err_title"), self._tr("err_load", str(e)))

    def load_results(self, path: str):
        item = next((i for i in self.queue_items if i.path == path), None)
        if not item or not item.results:
            return

        text, kr_records, im, recs = item.results
        self.canvas.load_pil_image(im)
        self.canvas.set_overlay_enabled(item.status == STATUS_DONE)

        if self.show_overlay:
            self.canvas.draw_overlays(recs)
        self._populate_lines_list(recs)

    def _populate_lines_list(self, recs: List[RecordView], keep_row: Optional[int] = None):
        self.list_lines.blockSignals(True)
        self.list_lines.clear()
        for i, rv in enumerate(recs):
            li = QListWidgetItem(f"{i + 1:04d}  {rv.text}")
            li.setData(Qt.UserRole, i)
            li.setFlags(li.flags() | Qt.ItemIsEditable)
            self.list_lines.addItem(li)
        self.list_lines.blockSignals(False)

        if recs:
            if keep_row is None:
                self.list_lines.setCurrentRow(0)
            else:
                self.list_lines.setCurrentRow(max(0, min(self.list_lines.count() - 1, keep_row)))

    def refresh_preview(self):
        if self.queue_table.currentRow() >= 0:
            path = self.queue_table.item(self.queue_table.currentRow(), 0).data(Qt.UserRole)
            item = next((i for i in self.queue_items if i.path == path), None)
            if item and item.status == STATUS_DONE:
                self.load_results(path)
            else:
                self.preview_image(path)

    def on_queue_double_click(self, row, col):
        path = self.queue_table.item(row, 0).data(Qt.UserRole)
        self.preview_image(path)

    def choose_rec_model(self):
        p, _ = QFileDialog.getOpenFileName(self, self._tr("dlg_choose_rec"), "", self._tr("dlg_filter_model"))
        if p:
            self.model_path = p
            name = os.path.basename(p)
            self.btn_rec_model.setText(f"{self._tr('dlg_choose_rec')}{name}")
            self.status_bar.showMessage(self._tr("msg_loaded_rec", name))
            self._update_models_menu_labels()

    def choose_seg_model(self):
        p, _ = QFileDialog.getOpenFileName(self, self._tr("dlg_choose_seg"), "", self._tr("dlg_filter_model"))
        if p:
            self.seg_model_path = p
            name = os.path.basename(p)
            self.btn_seg_model.setText(f"{self._tr('dlg_choose_seg')}{name}")
            self.status_bar.showMessage(self._tr("msg_loaded_seg", name))
            self._update_models_menu_labels()

    def _log(self, msg: str):
        ts = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        line = f"[{ts}] {msg}"
        try:
            self.log_edit.appendPlainText(line)
        except Exception:
            pass

    def toggle_log_area(self, checked: bool):
        self.log_visible = bool(checked)
        self.log_edit.setVisible(self.log_visible)

        # Button-Text umschalten
        if self.act_toggle_log.isChecked():
            self.act_toggle_log.setText(self._tr("log_toggle_hide"))
        else:
            self.act_toggle_log.setText(self._tr("log_toggle_show"))

    def export_log_txt(self):
        base_dir = self.current_export_dir or os.getcwd()
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("dlg_save_log"),
            os.path.join(base_dir, "ocr_log.txt"),
            self._tr("dlg_filter_txt")
        )
        if not dest_path:
            return
        if not dest_path.lower().endswith(".txt"):
            dest_path += ".txt"

        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(self.log_edit.toPlainText())
            self._log(self._tr_log("log_export_log_done", dest_path))
            self.status_bar.showMessage(self._tr("msg_exported", os.path.basename(dest_path)))
        except Exception as e:
            QMessageBox.critical(self, self._tr("err_title"), str(e))

    # -----------------------------
    # OCR controls
    # -----------------------------
    def start_ocr(self):
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.critical(self, self._tr("err_title"), self._tr("warn_need_rec"))
            return
        if getattr(self, "segmenter_mode", "blla") == "blla":
            if not self.seg_model_path:
                default_seg = os.path.join(os.path.dirname(__file__), "blla.mlmodel")
                if os.path.exists(default_seg):
                    self.seg_model_path = default_seg
                    name = os.path.basename(self.seg_model_path)
                    self.btn_seg_model.setText(f"{self._tr('dlg_choose_seg')}{name}")
                    self.status_bar.showMessage(self._tr("msg_loaded_seg", name))
                    self._update_models_menu_labels()

            if not self.seg_model_path or not os.path.exists(self.seg_model_path):
                QMessageBox.critical(self, self._tr("err_title"), self._tr("warn_need_seg"))
                return
        if not self.seg_model_path or not os.path.exists(self.seg_model_path):
            QMessageBox.critical(self, self._tr("err_title"), self._tr("warn_need_seg"))
            return

        tasks = [i for i in self.queue_items if i.status == STATUS_WAITING]
        if not tasks:
            QMessageBox.information(self, self._tr("info_title"), self._tr("warn_queue_empty"))
            return

        caps = self._gpu_capabilities()
        ok, _ = caps.get(self.device_str, (False, ""))
        if not ok:
            QMessageBox.warning(self, self._tr("warn_title"), self._tr("msg_hw_not_available"))
            self.device_str = "cpu"
            if "cpu" in self.hw_actions:
                self.hw_actions["cpu"].setChecked(True)

        self.act_play.setEnabled(False)
        self.act_stop.setEnabled(True)
        self._set_progress_busy()

        paths = [t.path for t in tasks]
        job = OCRJob(
            input_paths=paths,
            recognition_model_path=self.model_path,
            segmentation_model_path=self.seg_model_path,
            device=self.device_str,
            reading_direction=self.reading_direction,
            export_format="pdf",
            export_dir=self.current_export_dir,
            segmenter_mode = "blla",
        )

        self.worker = OCRWorker(job)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.file_error.connect(self.on_file_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.finished_batch.connect(self.on_batch_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.device_resolved.connect(self.on_device_resolved)
        self.worker.gpu_info.connect(self.on_gpu_info)
        self._log(self._tr_log("log_ocr_started", len(paths), self.device_str, self.reading_direction))
        self.worker.start()

    def on_device_resolved(self, dev_str: str):
        self.status_bar.showMessage(self._tr("msg_using_device", dev_str))

    def on_gpu_info(self, info: str):
        self.status_bar.showMessage(self._tr("msg_detected_gpu", info))

    def reprocess_selected(self):
        if self.queue_table.currentRow() < 0:
            QMessageBox.warning(self, self._tr("warn_title"), self._tr("warn_select_done"))
            return

        path = self.queue_table.item(self.queue_table.currentRow(), 0).data(Qt.UserRole)
        item = next((i for i in self.queue_items if i.path == path), None)

        if item:
            item.status = STATUS_WAITING
            item.results = None
            item.edited = False
            item.undo_stack.clear()
            item.redo_stack.clear()
            self._update_queue_row(path)
            self.list_lines.clear()
            self.canvas.set_overlay_enabled(False)
            self._set_progress_idle(0)
            self.start_ocr()

    def stop_ocr(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self._log(self._tr_log("log_stop_requested"))
            self.status_bar.showMessage(self._tr("msg_stopping"))

    def on_file_started(self, path):
        item = next((i for i in self.queue_items if i.path == path), None)
        if item:
            item.status = STATUS_PROCESSING
            self._update_queue_row(path)
            self._log(self._tr_log("log_file_started", os.path.basename(path)))

    def on_file_done(self, path, text, kr_records, im, recs):
        item = next((i for i in self.queue_items if i.path == path), None)
        if item:
            item.status = STATUS_DONE
            item.results = (text, kr_records, im, recs)
            item.edited = False
            item.undo_stack.clear()
            item.redo_stack.clear()
            self._update_queue_row(path)

            if self.queue_table.currentRow() >= 0:
                cur_path = self.queue_table.item(self.queue_table.currentRow(), 0).data(Qt.UserRole)
                if cur_path == path:
                    self.load_results(path)
            self._log(self._tr_log("log_file_done", os.path.basename(path), len(recs)))

    def on_file_error(self, path, msg):
        item = next((i for i in self.queue_items if i.path == path), None)
        if item:
            item.status = STATUS_ERROR
            self._update_queue_row(path)
            self._log(self._tr_log("log_file_error", os.path.basename(path), msg))

    def on_batch_finished(self):
        self.act_play.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.status_bar.showMessage(self._tr("msg_finished"))
        self.progress_bar.setValue(100)

    def on_failed(self, msg):
        QMessageBox.critical(self, self._tr("err_title"), msg)
        self.act_play.setEnabled(True)
        self.act_stop.setEnabled(False)
        self._set_progress_idle(0)

    def _update_queue_row(self, path):
        for row in range(self.queue_table.rowCount()):
            item0 = self.queue_table.item(row, 0)
            if item0 and item0.data(Qt.UserRole) == path:
                status_item = self.queue_table.item(row, 1)
                task = next((i for i in self.queue_items if i.path == path), None)
                if task and status_item:
                    status_enum = task.status
                    status_icon = STATUS_ICONS[status_enum]
                    status_key = {
                        STATUS_WAITING: "status_waiting",
                        STATUS_PROCESSING: "status_processing",
                        STATUS_DONE: "status_done",
                        STATUS_ERROR: "status_error",
                    }[status_enum]
                    status_item.setText(f"{status_icon} {self._tr(status_key)}")

                    if status_enum == STATUS_DONE:
                        status_item.setForeground(QBrush(QColor("green")))
                    elif status_enum == STATUS_ERROR:
                        status_item.setForeground(QBrush(QColor("red")))
                    else:
                        status_item.setForeground(QBrush(QColor("blue")))
                break

    # -----------------------------
    # Lines + overlays
    # -----------------------------
    def _current_task(self) -> Optional[TaskItem]:
        if self.queue_table.currentRow() < 0:
            return None
        path = self.queue_table.item(self.queue_table.currentRow(), 0).data(Qt.UserRole)
        return next((i for i in self.queue_items if i.path == path), None)

    def on_line_selected(self, row):
        task = self._current_task()
        if not task or not task.results or row < 0:
            return
        _, _, _, recs = task.results
        if 0 <= row < len(recs):
            self.canvas.select_idx(row)

    def on_rect_clicked(self, idx):
        if 0 <= idx < self.list_lines.count():
            self.list_lines.setCurrentRow(idx)
            self.list_lines.setFocus()

    @staticmethod
    def _parse_line_item_full(text: str) -> Tuple[Optional[int], str]:
        t = (text or "").rstrip("\n")
        m = re.match(r"^\s*(\d+)\s+(.*)$", t)
        if not m:
            return None, t.strip()
        num = int(m.group(1))
        rest = (m.group(2) or "").strip()
        return num - 1, rest

    def on_line_item_edited(self, item: QListWidgetItem):
        task = self._current_task()
        if not task or not task.results or task.status != STATUS_DONE:
            return

        text, kr_records, im, recs = task.results
        row = self.list_lines.row(item)
        if row is None or not (0 <= row < len(recs)):
            return

        target_idx, new_text = self._parse_line_item_full(item.text())
        new_text = (new_text or "").strip()

        if target_idx is None:
            target_idx = row

        target_idx = max(0, min(len(recs) - 1, int(target_idx)))

        def normalize_display(selected_row: int):
            self.list_lines.blockSignals(True)
            for i, rv in enumerate(recs):
                it = self.list_lines.item(i)
                if it:
                    it.setText(f"{i + 1:04d}  {rv.text}")
                    it.setData(Qt.UserRole, i)
            self.list_lines.blockSignals(False)
            self.list_lines.setCurrentRow(max(0, min(self.list_lines.count() - 1, selected_row)))

        if target_idx == row and new_text == recs[row].text:
            normalize_display(row)
            return

        self._push_undo(task)

        if target_idx != row:
            rv = recs.pop(row)
            rv.text = new_text
            recs.insert(target_idx, rv)
            task.edited = True
            self._sync_ui_after_recs_change(task, keep_row=target_idx)
            return

        old_text = recs[row].text
        recs[row].text = new_text
        task.edited = True

        if recs[row].bbox and im:
            x0, y0, x1, y1 = recs[row].bbox
            old_len = max(1, len(old_text))
            new_len = max(1, len(new_text))
            w = max(10, x1 - x0)
            avg_char = w / float(old_len)
            new_w = int(max(10, avg_char * new_len))
            img_w, img_h = im.size
            new_x1 = min(img_w, x0 + new_w)
            recs[row].bbox = (x0, y0, new_x1, y1)

        self._sync_ui_after_recs_change(task, keep_row=row)

    def _delete_current_line_via_key(self):
        task = self._current_task()
        if not task or not task.results or task.status != STATUS_DONE:
            return
        row = self.list_lines.currentRow()
        if row >= 0:
            self._delete_line(task, row)

    def on_lines_reordered(self, order: list, current_row_after_drop: int):
        task = self._current_task()
        if not task or not task.results or task.status != STATUS_DONE:
            return
        text, kr_records, im, recs = task.results

        if not order or len(order) != len(recs):
            return

        try:
            new_recs = [recs[int(i)] for i in order]
        except Exception:
            return

        self._push_undo(task)
        task.edited = True
        task.results = (text, kr_records, im, new_recs)
        self._sync_ui_after_recs_change(task, keep_row=max(0, min(len(new_recs) - 1, int(current_row_after_drop))))

    def lines_context_menu(self, pos):
        item = self.list_lines.itemAt(pos)
        if item is None:
            return
        row = self.list_lines.row(item)

        menu = QMenu()
        act_up = menu.addAction(self._tr("line_menu_move_up"))
        act_down = menu.addAction(self._tr("line_menu_move_down"))
        menu.addSeparator()
        act_move_to = menu.addAction(self._tr("line_menu_move_to"))
        menu.addSeparator()
        act_del = menu.addAction(self._tr("line_menu_delete"))
        menu.addSeparator()
        act_add_above = menu.addAction(self._tr("line_menu_add_above"))
        act_add_below = menu.addAction(self._tr("line_menu_add_below"))
        menu.addSeparator()
        act_draw = menu.addAction(self._tr("line_menu_draw_box"))
        menu.addSeparator()
        act_edit_box = menu.addAction(self._tr("line_menu_edit_box"))

        chosen = menu.exec(self.list_lines.viewport().mapToGlobal(pos))
        if not chosen:
            return

        task = self._current_task()
        if not task or not task.results or task.status != STATUS_DONE:
            return

        if chosen == act_up:
            self._move_line(task, row, -1)
        elif chosen == act_down:
            self._move_line(task, row, +1)
        elif chosen == act_move_to:
            self._move_line_to_dialog(task, row)
        elif chosen == act_del:
            self._delete_line(task, row)
        elif chosen == act_add_above:
            self._add_line(task, insert_row=row)
        elif chosen == act_add_below:
            self._add_line(task, insert_row=row + 1)
        elif chosen == act_draw:
            # Draw box FOR THIS LINE (kept as-is)
            self._pending_new_line_box = False
            self._pending_box_for_row = row
            self.canvas.start_draw_box_mode()
        elif chosen == act_edit_box:
            self.show_overlay = True
            self.act_overlay.setChecked(True)
            self.refresh_preview()

    def _sync_ui_after_recs_change(self, task: TaskItem, keep_row: Optional[int] = None):
        if not task.results:
            return
        text, kr_records, im, recs = task.results

        for i, rv in enumerate(recs):
            rv.idx = i

        new_text = "\n".join([r.text for r in recs]).strip()
        task.results = (new_text, kr_records, im, recs)

        self.canvas.load_pil_image(im, preserve_view=True)
        self.canvas.set_overlay_enabled(task.status == STATUS_DONE)
        if self.show_overlay:
            self.canvas.draw_overlays(recs)

        self._populate_lines_list(recs, keep_row=keep_row)

    def _move_line(self, task: TaskItem, row: int, direction: int):
        text, kr_records, im, recs = task.results
        new_row = row + direction
        if not (0 <= row < len(recs)) or not (0 <= new_row < len(recs)):
            return
        self._push_undo(task)
        recs[row], recs[new_row] = recs[new_row], recs[row]
        task.edited = True
        self._sync_ui_after_recs_change(task, keep_row=new_row)

    def _move_line_to_dialog(self, task: TaskItem, row: int):
        if not task.results:
            return
        _, _, _, recs = task.results
        if not (0 <= row < len(recs)):
            return

        target, ok = QInputDialog.getInt(
            self,
            self._tr("dlg_move_to_title"),
            self._tr("dlg_move_to_label"),
            value=row + 1,
            min=1,
            max=max(1, len(recs)),
            step=1
        )
        if not ok:
            return
        self._move_line_to(task, row, target - 1)

    def _move_line_to(self, task: TaskItem, from_row: int, to_row: int):
        text, kr_records, im, recs = task.results
        if not (0 <= from_row < len(recs)):
            return
        to_row = max(0, min(len(recs) - 1, int(to_row)))
        if from_row == to_row:
            self._sync_ui_after_recs_change(task, keep_row=to_row)
            return
        self._push_undo(task)
        rv = recs.pop(from_row)
        recs.insert(to_row, rv)
        task.edited = True
        self._sync_ui_after_recs_change(task, keep_row=to_row)

    def _delete_line(self, task: TaskItem, row: int):
        text, kr_records, im, recs = task.results
        if not (0 <= row < len(recs)):
            return
        self._push_undo(task)
        recs.pop(row)
        task.edited = True
        next_row = min(row, max(0, len(recs) - 1)) if recs else None
        self._sync_ui_after_recs_change(task, keep_row=next_row)

    def _add_line(self, task: TaskItem, insert_row: int):
        new_text, ok = QInputDialog.getText(self, self._tr("dlg_new_line_title"), self._tr("dlg_new_line_label"))
        if not ok:
            return
        new_text = (new_text or "").strip()
        if not new_text:
            return
        text, kr_records, im, recs = task.results
        insert_row = max(0, min(len(recs), insert_row))
        self._push_undo(task)
        recs.insert(insert_row, RecordView(insert_row, new_text, None))
        task.edited = True
        self._sync_ui_after_recs_change(task, keep_row=insert_row)

        self._pending_new_line_box = False
        self._pending_box_for_row = insert_row
        self.canvas.start_draw_box_mode()

    # -----------------------------
    # Canvas actions
    # -----------------------------
    def on_canvas_select_line(self, idx: int):
        self.on_rect_clicked(idx)

    def _ensure_overlay_possible(self) -> Optional[TaskItem]:
        task = self._current_task()
        if not task or not task.results or task.status != STATUS_DONE:
            QMessageBox.information(self, self._tr("info_title"), self._tr("overlay_only_after_ocr"))
            return None
        return task

    def on_canvas_add_box_draw(self, scene_pos: QPointF):
        # NEW BEHAVIOR: drawing a new overlay box creates a NEW line at the end.
        task = self._ensure_overlay_possible()
        if not task:
            return
        _, _, _, recs = task.results
        if recs is None:
            return

        self._pending_box_for_row = None
        self._pending_new_line_box = True
        self.canvas.start_draw_box_mode()

    def on_canvas_edit_box(self, idx: int):
        task = self._ensure_overlay_possible()
        if not task:
            return
        _, _, im, recs = task.results
        if not im:
            return
        if not (0 <= idx < len(recs)):
            return
        img_w, img_h = im.size
        dlg = OverlayBoxDialog(self._tr, img_w, img_h, bbox=recs[idx].bbox, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._push_undo(task)
        recs[idx].bbox = dlg.get_bbox()
        task.edited = True
        self._sync_ui_after_recs_change(task, keep_row=idx)

    def on_canvas_delete_box(self, idx: int):
        task = self._ensure_overlay_possible()
        if not task:
            return
        _, _, _, recs = task.results
        if not (0 <= idx < len(recs)):
            return
        self._push_undo(task)
        recs[idx].bbox = None
        task.edited = True
        self._sync_ui_after_recs_change(task, keep_row=idx)

    # -----------------------------
    # Box drawing result
    # -----------------------------
    def on_box_drawn(self, rect: QRectF):
        task = self._ensure_overlay_possible()
        if not task:
            return

        text, kr_records, im, recs = task.results

        x0 = _safe_int(rect.left())
        y0 = _safe_int(rect.top())
        x1 = _safe_int(rect.right())
        y1 = _safe_int(rect.bottom())
        x0, y0, x1, y1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        if im:
            img_w, img_h = im.size
            x0, y0 = max(0, min(img_w - 1, x0)), max(0, min(img_h - 1, y0))
            x1, y1 = max(1, min(img_w, x1)), max(1, min(img_h, y1))
            if x1 <= x0:
                x1 = min(img_w, x0 + 1)
            if y1 <= y0:
                y1 = min(img_h, y0 + 1)

        # Case A: create new line at end (canvas draw)
        if self._pending_new_line_box:
            self._pending_new_line_box = False
            self._pending_box_for_row = None

            # Optional: ask for text (optional) – user can also just edit in list afterwards.
            new_txt, ok = QInputDialog.getText(self, self._tr("new_line_from_box_title"), self._tr("new_line_from_box_label"))
            if not ok:
                new_txt = ""
            new_txt = (new_txt or "").strip()

            self._push_undo(task)
            recs.append(RecordView(len(recs), new_txt, (x0, y0, x1, y1)))
            task.edited = True
            self._sync_ui_after_recs_change(task, keep_row=len(recs) - 1)
            self.list_lines.setFocus()
            return

        # Case B: draw box for a specific existing row (line context menu)
        if self._pending_box_for_row is None:
            return

        row = self._pending_box_for_row
        self._pending_box_for_row = None

        if not (0 <= row < len(recs)):
            return

        self._push_undo(task)
        recs[row].bbox = (x0, y0, x1, y1)
        task.edited = True
        self._sync_ui_after_recs_change(task, keep_row=row)

    def on_overlay_rect_changed(self, idx: int, scene_rect: QRectF):
        task = self._ensure_overlay_possible()
        if not task:
            return

        text, kr_records, im, recs = task.results
        if not (0 <= idx < len(recs)):
            return

        if im:
            img_w, img_h = im.size
            r = scene_rect.normalized()
            x0 = max(0, min(img_w - 1, _safe_int(r.left())))
            y0 = max(0, min(img_h - 1, _safe_int(r.top())))
            x1 = max(1, min(img_w, _safe_int(r.right())))
            y1 = max(1, min(img_h, _safe_int(r.bottom())))

            if x1 <= x0:
                x1 = min(img_w, x0 + 1)
            if y1 <= y0:
                y1 = min(img_h, y0 + 1)

            old = recs[idx].bbox
            new = (x0, y0, x1, y1)
            if old != new:
                self._push_undo(task)
                recs[idx].bbox = new
                task.edited = True

                keep = self.list_lines.currentRow() if self.list_lines.currentRow() >= 0 else idx
                self._sync_ui_after_recs_change(task, keep_row=keep)

    # -----------------------------
    # Overlay toggle
    # -----------------------------
    def _on_overlay_toggled(self, checked):
        self.show_overlay = checked
        self.refresh_preview()

    # -----------------------------
    # Export
    # -----------------------------
    def export_flow(self, fmt: str):
        if len(self.queue_items) == 0:
            QMessageBox.warning(self, self._tr("warn_title"), self._tr("warn_queue_empty"))
            return

        if len(self.queue_items) == 1:
            it = self.queue_items[0]
            if len(self.queue_items) == 1:
                it = self.queue_items[0]
                if it.status != STATUS_DONE or not it.results:
                    QMessageBox.warning(self, self._tr("warn_title"), self._tr("warn_select_done"))
                    return
                self._export_single_interactive(it, fmt)
                return

        dlg = ExportModeDialog(self._tr, self)
        if dlg.exec() != QDialog.Accepted or dlg.choice is None:
            return

        if dlg.choice == "all":
            items = [it for it in self.queue_items if it.status == STATUS_DONE and it.results]
            if len(items) != len(self.queue_items):
                QMessageBox.warning(self, self._tr("warn_title"), self._tr("export_need_done"))
                return
            self._export_batch(items, fmt)
            return

        sel_dlg = ExportSelectFilesDialog(self._tr, self.queue_items, self)
        if sel_dlg.exec() != QDialog.Accepted:
            return
        paths = sel_dlg.selected_paths
        if not paths:
            QMessageBox.information(self, self._tr("info_title"), self._tr("export_none_selected"))
            return

        items = []
        for p in paths:
            it = next((x for x in self.queue_items if x.path == p), None)
            if not it or it.status != STATUS_DONE or not it.results:
                QMessageBox.warning(self, self._tr("warn_title"), self._tr("export_need_done"))
                return
            items.append(it)

        self._export_batch(items, fmt)

    def _export_single_interactive(self, item: TaskItem, fmt: str):
        base_name = os.path.splitext(item.display_name)[0]
        base_dir = self.current_export_dir or os.path.dirname(item.path)

        filters = {"txt": "Text (*.txt)", "csv": "CSV (*.csv)", "json": "JSON (*.json)",
                   "alto": "XML (*.xml)", "hocr": "HTML (*.html)", "pdf": "PDF (*.pdf)"}

        dest_path, _ = QFileDialog.getSaveFileName(
            self, self._tr("dlg_save"),
            os.path.join(base_dir, base_name),
            filters.get(fmt, "All (*.*)")
        )
        if not dest_path:
            return
        if not dest_path.lower().endswith(f".{fmt}"):
            dest_path += f".{fmt}"

        self._render_file(dest_path, fmt, item)
        self._log(self._tr_log("log_export_single", item.display_name, dest_path))
        self.status_bar.showMessage(self._tr("msg_exported", os.path.basename(dest_path)))

    def _export_batch(self, items: List[TaskItem], fmt: str):
        folder = QFileDialog.getExistingDirectory(self, self._tr("export_choose_folder"), self.current_export_dir or "")
        if not folder:
            return
        self.current_export_dir = folder

        for it in items:
            base_name = os.path.splitext(it.display_name)[0]
            dest_path = os.path.join(folder, f"{base_name}.{fmt}")
            self._render_file(dest_path, fmt, it)

        self.status_bar.showMessage(self._tr("msg_exported", folder))
        self._log(f"Export abgeschlossen: {len(items)} Datei(en) als {fmt} nach {folder}")


    def _render_file(self, path: str, fmt: str, item: TaskItem):
        if not item.results:
            return

        text, kr_records, pil_image, record_views = item.results

        if fmt == "txt":
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join([rv.text for rv in record_views]).strip())
            return

        grid = table_to_rows(record_views, pil_image.size[0]) if any(rv.bbox for rv in record_views) else [
            [rv.text] for rv in record_views
        ]

        if fmt == "csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(grid)
            return

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"rows": grid}, f, indent=2, ensure_ascii=False)
            return

        if fmt in ("alto", "hocr"):
            try:
                for i, rv in enumerate(record_views):
                    if i < len(kr_records) and hasattr(kr_records[i], "prediction"):
                        try:
                            kr_records[i].prediction = rv.text
                        except Exception:
                            pass
            except Exception:
                pass

            img_name = os.path.basename(path)
            xml = serialization.serialize(kr_records, image_name=img_name, image_size=pil_image.size, template=fmt)
            with open(path, "w", encoding="utf-8") as f:
                f.write(xml)
            return

        if fmt == "pdf":
            width, height = pil_image.size
            c = pdf_canvas.Canvas(path, pagesize=(width, height))
            c.drawImage(ImageReader(pil_image), 0, 0, width=width, height=height)

            for rv in record_views:
                if not rv.bbox or not rv.text.strip():
                    continue
                x0, y0, x1, y1 = rv.bbox
                t = rv.text
                box_h = max(1, y1 - y0)
                box_w = max(1, x1 - x0)
                font_size = max(6, min(24, box_h * 0.8))
                c.setFont("Helvetica", font_size)
                pdf_y = height - y1
                text_w = c.stringWidth(t, "Helvetica", font_size)
                scale_x = box_w / text_w if text_w > 0 else 1.0
                c.saveState()
                c.translate(x0, pdf_y)
                c.scale(scale_x, 1.0)
                c.setFillAlpha(0)
                c.drawString(0, 0, t)
                c.restoreState()

            c.save()
            return

    # -----------------------------
    # Keyboard focus handling
    # -----------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not getattr(self.canvas, "_draw_mode", False):
            self._space_panning = True
            self.canvas.setDragMode(QGraphicsView.ScrollHandDrag)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_panning = False
            self.canvas.setDragMode(QGraphicsView.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()