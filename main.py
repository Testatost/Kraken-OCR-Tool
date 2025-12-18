import os
import sys
import json
import csv
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple, Dict

warnings.filterwarnings(
    "ignore",
    message="Using legacy polygon extractor*",
    category=UserWarning
)

from PIL import Image
from PIL.ImageQt import ImageQt

from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QPen, QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QPushButton, QComboBox, QLineEdit, QPlainTextEdit,
    QProgressBar, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QListWidget, QListWidgetItem, QCheckBox, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsSimpleTextItem, QSplitter
)

# Kraken
from kraken import binarization, pageseg, blla, rpred, serialization
from kraken.lib import models, vgsl

# torch (device selection best-effort)
import torch


# -----------------------------
# Language support (Kraken multilingual helper)
# -----------------------------
LANG_LABELS: Dict[str, str] = {
    "auto": "Auto (keine Filterung)",
    "de": "Deutsch",
    "en": "Englisch",
    "fr": "Französisch",
    "la": "Latein",
}

LANG_CHARSETS: Dict[str, set] = {
    "de": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜßẞſ"),
    "en": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "fr": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàâçéèêëîïôùûüÿœÀÂÇÉÈÊËÎÏÔÙÛÜŸŒ"),
    "la": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}


def _try_get_model_alphabet(rec_model: Any) -> Optional[List[str]]:
    """
    Versucht, das Alphabet/Codec-Alphabet eines Kraken Recognition Modells auszulesen.
    Kraken-Versionen/Modelle unterscheiden sich; daher best-effort.
    """
    try:
        if hasattr(rec_model, "alphabet") and rec_model.alphabet:
            return list(rec_model.alphabet)
    except Exception:
        pass

    try:
        codec = getattr(rec_model, "codec", None)
        if codec is not None and hasattr(codec, "alphabet") and codec.alphabet:
            return list(codec.alphabet)
    except Exception:
        pass

    try:
        nn = getattr(rec_model, "nn", None)
        codec = getattr(nn, "codec", None) if nn is not None else None
        if codec is not None and hasattr(codec, "alphabet") and codec.alphabet:
            return list(codec.alphabet)
    except Exception:
        pass

    return None


def detect_languages_from_alphabet(alphabet: List[str]) -> List[str]:
    """
    Heuristik: Prüft, welche Sprach-Charsets im Modellalphabet auftauchen.
    Gibt immer mindestens ["auto"] zurück.
    """
    if not alphabet:
        return ["auto"]

    chars = set(alphabet)

    langs = ["auto"]
    for code, charset in LANG_CHARSETS.items():
        if chars & charset:
            langs.append(code)

    langs = sorted(set(langs), key=lambda x: (0 if x == "auto" else 1, x))
    return langs


def normalize_text_by_language(text: str, lang: str) -> str:
    """
    Best-effort: Filtert Zeichen, die nicht zur gewählten Sprache passen.
    'auto' -> keine Filterung.
    """
    if not text:
        return text
    lang = (lang or "auto").strip().lower()
    if lang == "auto" or lang not in LANG_CHARSETS:
        return text

    allowed = LANG_CHARSETS[lang]
    out = []
    for ch in text:
        if ch.isspace():
            out.append(ch)
        elif ch in allowed:
            out.append(ch)
        else:
            pass
    return "".join(out).strip()


# -----------------------------
# Data
# -----------------------------
@dataclass
class OCRJob:
    input_paths: List[str]
    recognition_model_path: str
    segmentation_mode: str             # "legacy" or "baseline"
    segmentation_model_path: Optional[str]
    device: str                        # "cpu" | "cuda" | "mps"
    language: str                      # "auto"|"de"|"en"|"fr"|"la"
    table_mode: bool
    export_format: str                 # "txt"|"csv"|"json"|"alto"|"hocr"
    export_dir: Optional[str]


@dataclass
class RecordView:
    idx: int
    text: str
    bbox: Optional[Tuple[int, int, int, int]]  # (x0,y0,x1,y1)


# -----------------------------
# Helpers: reading order & geometry
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


def record_sort_key(r: Any) -> Tuple[int, int]:
    bbox = getattr(r, "bbox", None)
    if bbox:
        try:
            x0, y0, _, _ = bbox
            return int(y0), int(x0)
        except Exception:
            pass

    baseline = getattr(r, "baseline", None)
    if baseline:
        pts = _coerce_points(baseline)
        if pts:
            return int(min(p[1] for p in pts)), int(min(p[0] for p in pts))

    boundary = getattr(r, "boundary", None) or getattr(r, "polygon", None)
    if boundary:
        pts = _coerce_points(boundary)
        if pts:
            return int(min(p[1] for p in pts)), int(min(p[0] for p in pts))

    return (0, 0)


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

    boundary = getattr(r, "boundary", None)
    if boundary:
        pts = _coerce_points(boundary)
        bb = _bbox_from_points(pts, pad=2)
        if bb:
            return bb

    poly = getattr(r, "polygon", None)
    if poly:
        pts = _coerce_points(poly)
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


def clamp_bbox(bb: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bb
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


# -----------------------------
# Table / register structuring (simple heuristic)
# -----------------------------
def cluster_columns(records: List[RecordView], x_threshold: int = 45) -> List[List[RecordView]]:
    cols: List[Dict[str, Any]] = []
    for r in records:
        if not r.bbox:
            continue
        x0, _, _, _ = r.bbox
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


def group_rows_by_y(records: List[RecordView], y_threshold: int = 22) -> List[List[RecordView]]:
    rows: List[List[RecordView]] = []
    for r in sorted(records, key=lambda rv: (rv.bbox[1] if rv.bbox else 0, rv.bbox[0] if rv.bbox else 0)):
        if not r.bbox:
            continue
        y0 = r.bbox[1]
        placed = False
        for row in rows:
            ry0 = row[0].bbox[1]
            if abs(ry0 - y0) <= y_threshold:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])
    for row in rows:
        row.sort(key=lambda rv: rv.bbox[0] if rv.bbox else 0)
    return rows


def table_to_rows(records: List[RecordView]) -> List[List[str]]:
    rows = group_rows_by_y(records)
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

    grid: List[List[str]] = []
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
# Graphics Items for interaction
# -----------------------------
class ClickableRect(QGraphicsRectItem):
    def __init__(self, rect: QRectF, idx: int, text: str):
        super().__init__(rect)
        self.idx = idx
        self.text = text
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"<b>Zeile {idx + 1}</b><br/>{text}")


class ImageCanvas(QGraphicsView):
    rect_clicked = Signal(int)

    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self._zoom = 1.0
        self._pixmap_item = None
        self._rects: Dict[int, ClickableRect] = {}
        self._labels: Dict[int, QGraphicsSimpleTextItem] = {}
        self._selected_idx: Optional[int] = None

        self._pen_normal = QPen(QColor("#ff3b30"), 2)
        self._pen_selected = QPen(QColor("#0a84ff"), 3)
        self._brush_fill = QBrush(QColor(255, 59, 48, 30))

    def clear_all(self):
        self.scene.clear()
        self._pixmap_item = None
        self._rects.clear()
        self._labels.clear()
        self._selected_idx = None
        self.resetTransform()
        self._zoom = 1.0

    def load_pil_image(self, im: Image.Image):
        self.clear_all()
        qim = ImageQt(im.convert("RGB"))
        pix = QPixmap.fromImage(qim)
        self._pixmap_item = self.scene.addPixmap(pix)
        self._pixmap_item.setZValue(0)
        self.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def draw_overlays(self, recs: List[RecordView]):
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)

        for rv in recs:
            if not rv.bbox:
                continue
            x0, y0, x1, y1 = rv.bbox
            rectf = QRectF(x0, y0, x1 - x0, y1 - y0)

            ritem = ClickableRect(rectf, rv.idx, rv.text)
            ritem.setPen(self._pen_normal)
            ritem.setBrush(self._brush_fill)
            ritem.setZValue(10)
            self.scene.addItem(ritem)
            self._rects[rv.idx] = ritem

            lab = QGraphicsSimpleTextItem(str(rv.idx + 1))
            lab.setFont(font)
            lab.setBrush(QBrush(QColor("#111")))
            lab.setZValue(11)
            lab.setPos(x0, max(0, y0 - 16))
            self.scene.addItem(lab)
            self._labels[rv.idx] = lab

    def select_idx(self, idx: Optional[int], center: bool = True):
        for rect in self._rects.values():
            rect.setPen(self._pen_normal)
            rect.setBrush(self._brush_fill)

        self._selected_idx = idx
        if idx is not None and idx in self._rects:
            rect = self._rects[idx]
            rect.setPen(self._pen_selected)
            rect.setBrush(QBrush(QColor(10, 132, 255, 60)))
            if center:
                self.centerOn(rect)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, ClickableRect):
            self.rect_clicked.emit(item.idx)
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(0.8)

    def zoom_reset(self):
        if not self.sceneRect().isValid():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def _apply_zoom(self, factor: float):
        new_zoom = self._zoom * factor
        if 0.05 <= new_zoom <= 20.0:
            self.scale(factor, factor)
            self._zoom = new_zoom


# -----------------------------
# OCR Worker Threads (Kraken only)
# -----------------------------
class OCRWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done_single = Signal(str, list, object, list)  # text, kraken_records, pil_image, record_views
    done_batch = Signal(list)                      # list of (path, ok, outpath_or_error)
    failed = Signal(str)

    def __init__(self, job: OCRJob):
        super().__init__()
        self.job = job
        self._device: Optional[torch.device] = None
        self._rec_model: Any = None
        self._seg_model: Any = None

    def _resolve_device(self) -> torch.device:
        dev = (self.job.device or "cpu").lower().strip()

        if dev == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            self.log.emit("⚠ CUDA gewählt, aber nicht verfügbar → CPU")

        if dev == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            self.log.emit("⚠ MPS gewählt, aber nicht verfügbar → CPU")

        return torch.device("cpu")

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
            self.log.emit(f"Device: {self._device}")

        if self._rec_model is None:
            self.log.emit("Recognition: Kraken-Modell laden (cached)…")
            self._rec_model = self._load_rec_model(self.job.recognition_model_path, self._device)

        if self.job.segmentation_mode == "baseline" and self._seg_model is None:
            if not self.job.segmentation_model_path:
                raise ValueError("Baseline gewählt, aber Segmentierungsmodell fehlt.")
            self.log.emit("Segmentierung: Baseline-Modell laden (cached)…")
            self._seg_model = self._load_seg_model(self.job.segmentation_model_path, self._device)

    def _segmentation_stats(self, seg: Any) -> Dict[str, Any]:
        stats = {}
        # Kraken objects differ across versions; best-effort.
        for attr in ("lines", "line", "baselines"):
            try:
                v = getattr(seg, attr, None)
                if v is not None and hasattr(v, "__len__"):
                    stats["lines"] = len(v)
                    break
            except Exception:
                pass
        for attr in ("regions", "region"):
            try:
                v = getattr(seg, attr, None)
                if v is not None and hasattr(v, "__len__"):
                    stats["regions"] = len(v)
                    break
            except Exception:
                pass
        return stats

    def _segment_legacy(self, im: Image.Image) -> Any:
        self.log.emit("Segmentierung: Legacy (nlbin + pageseg)…")
        bw = binarization.nlbin(im)
        seg = pageseg.segment(bw)
        st = self._segmentation_stats(seg)
        if st:
            self.log.emit(f"Legacy-Seg Stats: {st}")
        return seg

    def _segment_baseline(self, im: Image.Image) -> Any:
        self.log.emit("Segmentierung: Baseline (blla)…")
        self._ensure_models_loaded()
        seg = blla.segment(im, model=self._seg_model)
        st = self._segmentation_stats(seg)
        if st:
            self.log.emit(f"Baseline-Seg Stats: {st}")
        return seg

    def _segment(self, im: Image.Image) -> Any:
        if self.job.segmentation_mode == "baseline":
            try:
                return self._segment_baseline(im)
            except Exception as e:
                self.log.emit(f"⚠ Baseline-Segmentierung fehlgeschlagen → Fallback Legacy. Grund: {e}")
                return self._segment_legacy(im)
        return self._segment_legacy(im)

    def _ocr_one(self, img_path: str) -> Tuple[str, list, Image.Image, List[RecordView]]:
        self.log.emit(f"Bild laden: {img_path}")
        im = Image.open(img_path)

        seg = self._segment(im)

        self._ensure_models_loaded()
        self.log.emit("OCR läuft (Kraken rpred)…")

        kr_records = list(rpred.rpred(self._rec_model, im, seg))
        kr_sorted = sorted(kr_records, key=record_sort_key)

        record_views: List[RecordView] = []
        lines: List[str] = []

        for idx, r in enumerate(kr_sorted):
            pred = getattr(r, "prediction", None)
            if pred is None:
                continue
            txt = normalize_text_by_language(str(pred), self.job.language)
            bb = record_bbox(r)
            record_views.append(RecordView(idx=idx, text=txt, bbox=bb))
            lines.append(txt)

        text = "\n".join(lines).strip()
        return text, kr_sorted, im, record_views

    def run(self):
        try:
            if not self.job.recognition_model_path or not os.path.exists(self.job.recognition_model_path):
                raise ValueError("Kraken Recognition-Modell fehlt oder Pfad existiert nicht.")

            if self.job.segmentation_mode == "baseline":
                if not self.job.segmentation_model_path or not os.path.exists(self.job.segmentation_model_path):
                    raise ValueError("Baseline-Segmentierung gewählt, aber Seg-Modell fehlt oder Pfad existiert nicht.")

            paths = self.job.input_paths
            if not paths:
                raise ValueError("Keine Eingabe (Bild/Ordner) ausgewählt.")

            is_batch = len(paths) > 1

            # PRELOAD models once for speed (especially batch)
            self._ensure_models_loaded()

            if is_batch:
                if not self.job.export_dir:
                    raise ValueError("Für Batch bitte Export-Ordner auswählen.")
                os.makedirs(self.job.export_dir, exist_ok=True)

                results = []
                total = len(paths)
                for i, p in enumerate(paths, start=1):
                    self.progress.emit(int((i - 1) / total * 100))
                    try:
                        text, kr_records, im, record_views = self._ocr_one(p)
                        out = self._save_outputs(
                            src_path=p,
                            text=text,
                            kr_records=kr_records,
                            pil_image=im,
                            record_views=record_views,
                            export_dir=self.job.export_dir,
                            export_format=self.job.export_format,
                            table_mode=self.job.table_mode,
                        )
                        results.append((p, True, out))
                    except Exception as e:
                        results.append((p, False, f"{e}"))
                self.progress.emit(100)
                self.done_batch.emit(results)
            else:
                self.progress.emit(5)
                text, kr_records, im, record_views = self._ocr_one(paths[0])
                self.progress.emit(95)
                self.done_single.emit(text, kr_records, im, record_views)
                self.progress.emit(100)

        except Exception:
            self.failed.emit(traceback.format_exc())

    def _save_outputs(
        self,
        src_path: str,
        text: str,
        kr_records: list,
        pil_image: Image.Image,
        record_views: List[RecordView],
        export_dir: str,
        export_format: str,
        table_mode: bool
    ) -> str:
        base = os.path.splitext(os.path.basename(src_path))[0]
        fmt = export_format.lower().strip()

        if fmt == "txt":
            out_path = os.path.join(export_dir, base + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            return out_path

        if fmt in ("csv", "json"):
            if table_mode:
                grid = table_to_rows(record_views)
            else:
                grid = [[rv.text] for rv in record_views]

            if fmt == "csv":
                out_path = os.path.join(export_dir, base + ".csv")
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for row in grid:
                        w.writerow(row)
                return out_path

            out_path = os.path.join(export_dir, base + ".json")
            payload = {"source": src_path, "rows": grid}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return out_path

        if fmt in ("alto", "hocr"):
            ext = ".xml" if fmt == "alto" else ".html"
            out_path = os.path.join(export_dir, base + ext)
            rendered = serialization.serialize(
                kr_records,
                image_name=os.path.basename(src_path),
                image_size=pil_image.size,
                template=fmt
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(rendered)
            return out_path

        raise ValueError(f"Unbekanntes Exportformat: {export_format}")


# -----------------------------
# Main GUI
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kraken OCR – GUI (Fedora/PySide6)")
        self.resize(1500, 900)

        self.worker: Optional[OCRWorker] = None

        self.current_image_path: Optional[str] = None
        self.current_folder_path: Optional[str] = None
        self.current_export_dir: Optional[str] = None

        self.kraken_records: Optional[list] = None
        self.pil_image: Optional[Image.Image] = None
        self.record_views: List[RecordView] = []

        self._detected_lang_codes: List[str] = ["auto"]

        self.canvas = ImageCanvas()
        self.canvas.rect_clicked.connect(self.on_rect_clicked)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_reset = QPushButton("0%")
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.canvas.zoom_reset)

        zoom_bar = QHBoxLayout()
        zoom_bar.addWidget(self.btn_zoom_in)
        zoom_bar.addWidget(self.btn_zoom_out)
        zoom_bar.addWidget(self.btn_zoom_reset)
        zoom_bar.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addLayout(zoom_bar)
        left_layout.addWidget(self.canvas)
        left.setLayout(left_layout)

        self.list_lines = QListWidget()
        self.list_lines.currentRowChanged.connect(self.on_line_selected)

        self.full_text = QPlainTextEdit()
        self.full_text.setPlaceholderText("Gesamtausgabe (Plain Text)…")

        mid = QWidget()
        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel("Erkannte Zeilen (klickbar):"))
        mid_layout.addWidget(self.list_lines, 3)
        mid_layout.addWidget(QLabel("Plain-Text Gesamt:"))
        mid_layout.addWidget(self.full_text, 2)
        mid.setLayout(mid_layout)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)

        self.in_mode = QComboBox()
        self.in_mode.addItem("Einzelbild", userData="single")
        self.in_mode.addItem("Ordner (Batch)", userData="batch")
        self.in_mode.currentIndexChanged.connect(self._update_input_mode_ui)

        self.btn_choose_image = QPushButton("Bild wählen…")
        self.btn_choose_image.clicked.connect(self.choose_image)

        self.btn_choose_folder = QPushButton("Ordner wählen…")
        self.btn_choose_folder.clicked.connect(self.choose_folder)

        self.btn_choose_export_dir = QPushButton("Export-Ordner…")
        self.btn_choose_export_dir.clicked.connect(self.choose_export_dir)

        self.path_image = QLineEdit()
        self.path_image.setPlaceholderText("Pfad zum Bild…")
        self.path_image.setReadOnly(True)

        self.path_folder = QLineEdit()
        self.path_folder.setPlaceholderText("Pfad zum Ordner…")
        self.path_folder.setReadOnly(True)

        self.path_export_dir = QLineEdit()
        self.path_export_dir.setPlaceholderText("Export-Ordner (für Batch / Export)…")
        self.path_export_dir.setReadOnly(True)

        self.btn_choose_rec_model = QPushButton("Kraken Rec-Modell…")
        self.btn_choose_rec_model.clicked.connect(self.choose_rec_model)
        self.path_rec_model = QLineEdit()
        self.path_rec_model.setPlaceholderText("Pfad zum Kraken Recognition-Modell (.mlmodel/.pt/…)")

        self.lang_combo = QComboBox()
        self.lang_combo.setEnabled(False)
        self.lang_combo.addItem(LANG_LABELS["auto"], userData="auto")
        self.lang_combo.setToolTip(
            "Sprachauswahl wirkt als Post-Processing (Zeichenfilter/Normalisierung).\n"
            "Aktiv nur, wenn das geladene Modell mehrsprachig erkannt wurde."
        )

        self.seg_mode = QComboBox()
        self.seg_mode.addItem("Legacy (nlbin + pageseg)", userData="legacy")
        self.seg_mode.addItem("Baseline (blla, benötigt Seg-Modell)", userData="baseline")
        self.seg_mode.currentIndexChanged.connect(self._update_seg_ui)

        self.btn_choose_seg_model = QPushButton("Seg-Modell…")
        self.btn_choose_seg_model.clicked.connect(self.choose_seg_model)
        self.path_seg_model = QLineEdit()
        self.path_seg_model.setPlaceholderText("Pfad zum Segmentierungsmodell (Baseline)")

        self.device = QComboBox()
        self.device.addItem("CPU", userData="cpu")
        self.device.addItem("CUDA (NVIDIA)", userData="cuda")
        self.device.addItem("MPS (Apple)", userData="mps")

        self.chk_table_mode = QCheckBox("Register/Tabellen strukturieren (heuristisch)")
        self.chk_overlay = QCheckBox("Overlay (Boxen + Nummern) anzeigen")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self._on_overlay_toggled)

        self.export_format = QComboBox()
        self.export_format.addItem("Plain Text (.txt)", userData="txt")
        self.export_format.addItem("CSV (.csv)", userData="csv")
        self.export_format.addItem("JSON (.json)", userData="json")
        self.export_format.addItem("ALTO XML (.xml)", userData="alto")
        self.export_format.addItem("hOCR (.html)", userData="hocr")

        self.btn_run = QPushButton("OCR starten")
        self.btn_run.clicked.connect(self.run_ocr)

        self.btn_export_single = QPushButton("Ergebnis speichern…")
        self.btn_export_single.clicked.connect(self.export_single)
        self.btn_export_single.setEnabled(False)

        controls = QGroupBox("Steuerung")
        form = QFormLayout()

        form.addRow("Eingabe:", self.in_mode)
        form.addRow("Bild:", self._hrow(self.path_image, self.btn_choose_image))
        form.addRow("Ordner:", self._hrow(self.path_folder, self.btn_choose_folder))
        form.addRow("Export-Ordner:", self._hrow(self.path_export_dir, self.btn_choose_export_dir))

        form.addRow("Kraken Rec-Modell:", self._hrow(self.path_rec_model, self.btn_choose_rec_model))
        form.addRow("Sprache (nur bei multilingual):", self.lang_combo)

        form.addRow("Segmentierung:", self.seg_mode)
        form.addRow("Seg-Modell (Baseline):", self._hrow(self.path_seg_model, self.btn_choose_seg_model))
        form.addRow("Device:", self.device)
        form.addRow(self.chk_table_mode)
        form.addRow(self.chk_overlay)
        form.addRow("Exportformat:", self.export_format)
        form.addRow(self.btn_run)
        form.addRow(self.progress)
        form.addRow(self.btn_export_single)

        controls.setLayout(form)

        right = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(controls)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self.log, 1)
        right.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(mid)
        splitter.addWidget(right)
        splitter.setSizes([800, 450, 350])

        self.setCentralWidget(splitter)

        self._update_seg_ui()
        self._update_input_mode_ui()

    def _hrow(self, *widgets):
        w = QWidget()
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            lay.addWidget(x)
        w.setLayout(lay)
        return w

    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _err(self, msg: str):
        QMessageBox.critical(self, "Fehler", msg)

    def _update_seg_ui(self):
        baseline = (self.seg_mode.currentData() == "baseline")
        self.path_seg_model.setEnabled(baseline)
        self.btn_choose_seg_model.setEnabled(baseline)

    def _update_input_mode_ui(self):
        mode = self.in_mode.currentData()
        is_single = (mode == "single")
        self.path_image.setEnabled(is_single)
        self.btn_choose_image.setEnabled(is_single)

        self.path_folder.setEnabled(not is_single)
        self.btn_choose_folder.setEnabled(not is_single)

        self.path_export_dir.setEnabled(True)
        self.btn_choose_export_dir.setEnabled(True)

    def _on_overlay_toggled(self):
        if not self.pil_image:
            return
        self.canvas.load_pil_image(self.pil_image)
        if self.chk_overlay.isChecked():
            self.canvas.draw_overlays(self.record_views)

    def _set_language_choices(self, codes: List[str]):
        codes = codes or ["auto"]
        if "auto" not in codes:
            codes = ["auto"] + list(codes)

        uniq = []
        for c in codes:
            if c not in uniq:
                uniq.append(c)

        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        for code in uniq:
            label = LANG_LABELS.get(code, code)
            self.lang_combo.addItem(label, userData=code)

        multilingual = any(c != "auto" for c in uniq) and len(uniq) > 1
        self.lang_combo.setEnabled(multilingual)

        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.blockSignals(False)

        self._detected_lang_codes = uniq

        if multilingual:
            self._log("Sprachen erkannt (multilingual): " + ", ".join(LANG_LABELS.get(c, c) for c in uniq))
        else:
            self._log("Keine multilingualen Sprachen erkannt – Sprache bleibt auf Auto.")

    def _detect_languages_for_model_path(self, model_path: str):
        try:
            rec_model = models.load_any(model_path)
        except Exception as e:
            self._log(f"Warnung: konnte Modell-Metadaten nicht laden (Sprachen): {e}")
            self._set_language_choices(["auto"])
            return

        alphabet = _try_get_model_alphabet(rec_model) or []
        langs = detect_languages_from_alphabet(alphabet)
        self._set_language_choices(langs)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild wählen",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All files (*.*)"
        )
        if not path:
            return
        self.current_image_path = path
        self.path_image.setText(path)
        self._log(f"Bild gewählt: {path}")

        try:
            im = Image.open(path)
            self.pil_image = im
            self.canvas.load_pil_image(im)
        except Exception as e:
            self._log(f"Vorschau fehlgeschlagen: {e}")

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Ordner wählen")
        if not path:
            return
        self.current_folder_path = path
        self.path_folder.setText(path)
        self._log(f"Ordner gewählt: {path}")

    def choose_export_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Export-Ordner wählen")
        if not path:
            return
        self.current_export_dir = path
        self.path_export_dir.setText(path)
        self._log(f"Export-Ordner: {path}")

    def choose_rec_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Kraken Recognition-Modell wählen",
            "",
            "Model (*.mlmodel *.pt *.pth *.ckpt);;All files (*.*)"
        )
        if not path:
            return
        self.path_rec_model.setText(path)
        self._log(f"Kraken Rec-Modell: {path}")
        self._detect_languages_for_model_path(path)

    def choose_seg_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Segmentierungsmodell wählen",
            "",
            "Model (*.mlmodel *.pt *.pth *.ckpt);;All files (*.*)"
        )
        if not path:
            return
        self.path_seg_model.setText(path)
        self._log(f"Seg-Modell: {path}")

    def _collect_input_paths(self) -> List[str]:
        mode = self.in_mode.currentData()
        if mode == "single":
            if not self.current_image_path or not os.path.exists(self.current_image_path):
                raise ValueError("Bitte ein Bild auswählen.")
            return [self.current_image_path]

        if not self.current_folder_path or not os.path.isdir(self.current_folder_path):
            raise ValueError("Bitte einen Ordner auswählen.")
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        paths = []
        for name in sorted(os.listdir(self.current_folder_path)):
            p = os.path.join(self.current_folder_path, name)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                paths.append(p)
        if not paths:
            raise ValueError("Im Ordner wurden keine Bilder gefunden (png/jpg/tif/…).")
        return paths

    def run_ocr(self):
        try:
            self.progress.setValue(0)
            self.log.clear()
            self.list_lines.clear()
            self.full_text.clear()
            self.btn_export_single.setEnabled(False)

            input_paths = self._collect_input_paths()

            fmt = self.export_format.currentData()

            rec_model = self.path_rec_model.text().strip()
            if not rec_model:
                raise ValueError("Bitte ein Kraken Recognition-Modell auswählen.")
            if not os.path.exists(rec_model):
                raise ValueError("Kraken Recognition-Modellpfad existiert nicht.")

            seg_mode = self.seg_mode.currentData()
            seg_model = self.path_seg_model.text().strip() if seg_mode == "baseline" else None
            if seg_mode == "baseline":
                if not seg_model:
                    raise ValueError("Baseline-Segmentierung benötigt ein Segmentierungsmodell.")
                if not os.path.exists(seg_model):
                    raise ValueError("Segmentierungsmodellpfad existiert nicht.")

            dev = self.device.currentData()
            table_mode = self.chk_table_mode.isChecked()

            export_dir = self.current_export_dir or ""
            is_batch = len(input_paths) > 1
            if is_batch and not export_dir:
                raise ValueError("Für Batch bitte einen Export-Ordner wählen.")

            lang = self.lang_combo.currentData() or "auto"

            job = OCRJob(
                input_paths=input_paths,
                recognition_model_path=rec_model,
                segmentation_mode=seg_mode,
                segmentation_model_path=seg_model,
                device=dev,
                language=lang,
                table_mode=table_mode,
                export_format=fmt,
                export_dir=(export_dir if export_dir else None)
            )

            self.btn_run.setEnabled(False)

            self.worker = OCRWorker(job)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.log.connect(self._log)
            self.worker.done_single.connect(self.on_done_single)
            self.worker.done_batch.connect(self.on_done_batch)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()

        except Exception as e:
            self._err(str(e))

    def on_failed(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_export_single.setEnabled(False)
        self._err("OCR fehlgeschlagen:\n\n" + msg)

    def on_done_batch(self, results: list):
        self.btn_run.setEnabled(True)
        ok = sum(1 for _, success, _ in results if success)
        fail = len(results) - ok
        self._log(f"\nBatch fertig: {ok} ok, {fail} fehlgeschlagen.")
        for p, success, out in results:
            if success:
                self._log(f"✔ {os.path.basename(p)} -> {out}")
            else:
                self._log(f"✖ {os.path.basename(p)} -> {out}")

    def on_done_single(self, text: str, kr_records: list, pil_image: Image.Image, record_views: list):
        self.btn_run.setEnabled(True)
        self.btn_export_single.setEnabled(True)

        self.kraken_records = kr_records
        self.pil_image = pil_image
        self.record_views = record_views

        self.canvas.load_pil_image(pil_image)
        if self.chk_overlay.isChecked():
            self.canvas.draw_overlays(record_views)

        self.full_text.setPlainText(text)

        self.list_lines.blockSignals(True)
        self.list_lines.clear()
        for rv in record_views:
            item = QListWidgetItem(f"{rv.idx + 1:04d}  {rv.text}")
            item.setData(Qt.UserRole, rv.idx)
            self.list_lines.addItem(item)
        self.list_lines.blockSignals(False)

        if record_views:
            self.list_lines.setCurrentRow(0)

    def on_line_selected(self, row: int):
        if row < 0 or row >= self.list_lines.count():
            self.canvas.select_idx(None)
            return
        item = self.list_lines.item(row)
        idx = item.data(Qt.UserRole)

        if self.chk_overlay.isChecked() and self.pil_image and not self.canvas._rects:
            self.canvas.draw_overlays(self.record_views)

        self.canvas.select_idx(idx, center=True)

    def on_rect_clicked(self, idx: int):
        for row in range(self.list_lines.count()):
            item = self.list_lines.item(row)
            if item.data(Qt.UserRole) == idx:
                self.list_lines.setCurrentRow(row)
                break

    def export_single(self):
        if self.pil_image is None:
            return self._err("Kein OCR-Ergebnis vorhanden.")

        fmt = self.export_format.currentData()
        default_dir = self.current_export_dir or os.path.dirname(self.current_image_path or "") or os.getcwd()

        if fmt == "txt":
            path, _ = QFileDialog.getSaveFileName(self, "Speichern", default_dir, "Text (*.txt)")
            if not path:
                return
            if not path.lower().endswith(".txt"):
                path += ".txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.full_text.toPlainText() + "\n")
            self._log(f"Gespeichert: {path}")
            return

        if fmt in ("csv", "json"):
            if self.chk_table_mode.isChecked():
                grid = table_to_rows(self.record_views)
            else:
                grid = [[rv.text] for rv in self.record_views]

            if fmt == "csv":
                path, _ = QFileDialog.getSaveFileName(self, "Speichern", default_dir, "CSV (*.csv)")
                if not path:
                    return
                if not path.lower().endswith(".csv"):
                    path += ".csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for row in grid:
                        w.writerow(row)
                self._log(f"Gespeichert: {path}")
                return

            path, _ = QFileDialog.getSaveFileName(self, "Speichern", default_dir, "JSON (*.json)")
            if not path:
                return
            if not path.lower().endswith(".json"):
                path += ".json"
            payload = {
                "source": self.current_image_path,
                "rows": grid
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"Gespeichert: {path}")
            return

        if fmt in ("alto", "hocr"):
            if not self.kraken_records:
                return self._err("Kein Kraken-Export möglich: Es sind keine Kraken-Records vorhanden.")
            ext = ".xml" if fmt == "alto" else ".html"
            flt = "XML (*.xml)" if fmt == "alto" else "HTML (*.html)"
            path, _ = QFileDialog.getSaveFileName(self, "Speichern", default_dir, flt)
            if not path:
                return
            if not path.lower().endswith(ext):
                path += ext

            rendered = serialization.serialize(
                self.kraken_records,
                image_name=os.path.basename(self.current_image_path) if self.current_image_path else None,
                image_size=self.pil_image.size,
                template=fmt
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(rendered)
            self._log(f"Gespeichert: {path}")
            return

        self._err(f"Unbekanntes Format: {fmt}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
