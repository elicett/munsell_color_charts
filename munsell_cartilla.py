#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera una cartilla/atlas PDF con parches Munsell a partir de datos de renotación (h V C x y Y).

Modos:
  --label-position coords  -> Sin etiquetas por parche; muestra ejes (Value izq., Chroma abajo)
                             (ejes en tipografía 14 pt, línea divisoria y espacio extra para Chroma).
  --label-position below   -> Etiqueta por parche debajo (p.ej. "10RP 1/2"); sin ejes.

Citación: esquina inferior derecha (1 cm), alineada a la derecha, con saltos de línea.

Uso:
  pip install reportlab pandas numpy
  python munsell_cartilla.py --input renotation.txt --output munsell_atlas.pdf \
      --page-size letter --profile sRGB --label-position {coords,below}
"""

import argparse
import re
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ----------------------------
# Utilidades de color
# ----------------------------

def xyY_to_XYZ(x: float, y: float, Y: float) -> Tuple[float, float, float]:
    if y == 0:
        return 0.0, 0.0, 0.0
    X = (x * Y) / y
    Z = ((1 - x - y) * Y) / y
    return X, Y, Z

def XYZ_to_sRGB(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    # Matriz sRGB D65 (lineal)
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb_lin = M @ np.array([X, Y, Z])

    def gamma_corr(c):
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1/2.4)) - 0.055

    r, g, b = (gamma_corr(v) for v in rgb_lin)
    # Recorte [0,1]
    return (max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)))

def normalize_Y_auto(series: pd.Series) -> pd.Series:
    """
    Si Y parece > 1.0 (p.ej. 1.2, 30, 90), se asume 0–100 y se normaliza a 0–1.
    """
    maxY = float(series.max())
    return series / 100.0 if maxY > 1.5 else series

# ----------------------------
# Parsing y orden de Hues
# ----------------------------

HUE_FAMILY_ORDER = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
HUE_STEPS_ORDER = [2.5, 5.0, 7.5, 10.0]  # orden típico de pasos por familia

import collections
HueTokenBase = collections.namedtuple("HueTokenBase", ["step", "family"])
class HueToken(HueTokenBase):
    __slots__ = ()
    def sort_key(self):
        fam_idx = HUE_FAMILY_ORDER.index(self.family) if self.family in HUE_FAMILY_ORDER else 999
        step_idx = HUE_STEPS_ORDER.index(self.step) if self.step in HUE_STEPS_ORDER else 999
        return (fam_idx, step_idx)

HUE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Z]+)\s*$")

def parse_hue_token(h: str) -> HueToken:
    m = HUE_PATTERN.match(h.strip().upper())
    if not m:
        return HueToken(0.0, h.strip().upper())
    step = float(m.group(1))
    fam  = m.group(2)
    return HueToken(step, fam)

# ----------------------------
# Layout de página (ejes y etiqueta opcional)
# ----------------------------

@dataclass
class PageLayout:
    margin: float = 36.0        # 0.5 in
    gutter_x: float = 10.0      # espacio horizontal entre parches
    gutter_y: float = 10.0      # espacio vertical entre filas
    box: float = 38.0           # tamaño objetivo del cuadro (se ajusta si no cabe)

    # Texto por parche (solo si --label-position=below)
    label_font: str = "Helvetica"
    label_size: float = 6.5
    label_height: float = 10.0  # alto reservado cuando hay etiqueta debajo

    # Título
    title_font: str = "Helvetica-Bold"
    title_size: float = 12.0
    title_gap: float = 24.0     # espacio vertical reservado para el título

    # Ejes (solo si --label-position=coords)
    axis_font: str = "Helvetica"
    axis_size_coords: float = 14.0  # **doble tamaño** para Value y Chroma en modo coords
    value_axis_w: float = 18.0      # ancho reservado izquierda para Values
    chroma_axis_h: float = 12.0     # alto reserva base para Chromas
    chroma_axis_extra_gap: float = 8.0  # **espacio extra** para despegar números de la última fila

    def fit_grid(self, page_w: float, page_h: float, max_cols: int, max_rows: int, label_position: str) -> float:
        """
        Calcula el tamaño máximo del cuadro para que quepan max_cols x max_rows,
        teniendo en cuenta si hay ejes (coords) o etiquetas debajo (below).
        """
        if label_position == "coords":
            usable_w = (page_w
                        - self.margin - self.value_axis_w
                        - self.margin
                        - (max_cols - 1)*self.gutter_x)
            usable_h = (page_h
                        - self.margin - self.title_gap
                        - self.margin - (self.chroma_axis_h + self.chroma_axis_extra_gap)
                        - (max_rows - 1)*self.gutter_y)
            cell_w = usable_w / max_cols if max_cols > 0 else usable_w
            cell_h = usable_h / max_rows if max_rows > 0 else usable_h
            size = min(self.box, cell_w, cell_h)
        else:  # below
            usable_w = (page_w
                        - self.margin
                        - self.margin
                        - (max_cols - 1)*self.gutter_x)
            usable_h = (page_h
                        - self.margin - self.title_gap
                        - self.margin
                        - (max_rows - 1)*self.gutter_y)
            cell_w = usable_w / max_cols if max_cols > 0 else usable_w
            cell_h = usable_h / max_rows if max_rows > 0 else usable_h
            size = min(self.box, cell_w, cell_h - self.label_height)

        return max(4.0, size)

# ----------------------------
# Dibujo
# ----------------------------

def draw_title(c: canvas.Canvas, layout: PageLayout, page_w: float, page_h: float, title: str):
    c.setFont(layout.title_font, layout.title_size)
    c.setFillColor(colors.black)
    c.drawCentredString(page_w/2, page_h - layout.margin + 6, title)

def draw_patch(c: canvas.Canvas, x: float, y: float, box: float, rgb: Tuple[float, float, float]):
    r, g, b = rgb
    c.setFillColorRGB(r, g, b)
    c.rect(x, y, box, box, fill=1, stroke=0)

def draw_patch_below_label(c: canvas.Canvas, x: float, y: float, box: float,
                           rgb: Tuple[float, float, float], label: str,
                           layout: PageLayout):
    rect_y = y + layout.label_height
    r, g, b = rgb
    c.setFillColorRGB(r, g, b)
    c.rect(x, rect_y, box, box, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont(layout.label_font, layout.label_size)
    c.drawCentredString(x + box/2, y + (layout.label_height - layout.label_size)/2, label)

def draw_value_axis_labels(c: canvas.Canvas, layout: PageLayout,
                           start_x: float, start_y: float,
                           box: float, row_step: float,
                           values_sorted):
    """Rotula los Values (V) en el margen izquierdo, centrados por fila (modo coords, 14 pt)."""
    c.setFont(layout.axis_font, layout.axis_size_coords)
    c.setFillColor(colors.black)
    label_x = layout.margin + layout.value_axis_w - 3
    for i, v in enumerate(values_sorted):
        base_y = start_y - i * row_step
        center_y = base_y + box/2
        c.drawRightString(label_x, center_y - layout.axis_size_coords/2, f"{int(v)}")

def draw_chroma_axis_labels(c: canvas.Canvas, layout: PageLayout,
                            start_x: float, grid_bottom_y: float,
                            box: float, col_step: float,
                            chromas_sorted):
    """Rotula los Chromas (C) abajo, centrados por columna (modo coords, 14 pt)."""
    c.setFont(layout.axis_font, layout.axis_size_coords)
    c.setFillColor(colors.black)

    # Línea divisoria bajo la grilla (pelo)
    c.setStrokeColor(colors.grey)
    c.setLineWidth(0.3)
    line_y = grid_bottom_y - 2  # justo bajo los parches
    c.line(start_x, line_y, start_x + (len(chromas_sorted)-1)*col_step + box, line_y)

    # Posición de texto: más abajo para despegar números
    label_y = line_y - layout.chroma_axis_extra_gap
    for j, cc in enumerate(chromas_sorted):
        center_x = start_x + j * col_step + box/2
        c.drawCentredString(center_x, label_y - layout.axis_size_coords/2 + 2, f"{int(cc)}")

# --- Citación con salto de línea, esquina inferior derecha ---

CM_TO_PT = 28.3464567  # 1 cm en puntos

CITATION_TEXT = (
    "\n Cartilla de colores - Atlas de colores \n"
    "Procesado por: Eliezer Licett, disponible en: https://github.com/elicett/munsell_color_charts.\n"
    "https://www.rit.edu/science/munsell-color-science-lab-educational-resources y https://www.rit-mcsl.org/MunsellRenotation/real.dat.\n"
    "Citado por Rochester Institute of Technology (Sf), disponible en:\n"
    "Munsell Renotation Data (1943). Publicado por: National Institute of Standards and Technology, EE.UU.(NIST).\n"
)

def ensure_arial_or_fallback():
    try:
        pdfmetrics.registerFont(TTFont("Arial", "Arial.ttf"))
        return "Arial"
    except Exception:
        return "Helvetica"

def wrap_right_aligned_lines(text: str, font_name: str, font_size: float, max_width: float) -> List[str]:
    """Envuelve texto en múltiples líneas (manual), respetando espacios, para alineación a la derecha."""
    words = text.replace("\r", "").split()
    lines, cur = [], ""
    sw = pdfmetrics.stringWidth
    for w in words:
        test = (cur + " " + w).strip()
        if sw(test, font_name, font_size) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    # Respetar saltos de línea explícitos '\n' en text original:
    # dividimos otra vez tomando esas marcas como cortes duros.
    final_lines = []
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            final_lines.append("")
            continue
        # re-wrap chunk por si es largo:
        words = chunk.split()
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if sw(test, font_name, font_size) <= max_width:
                cur = test
            else:
                if cur:
                    final_lines.append(cur)
                cur = w
        if cur:
            final_lines.append(cur)
    return final_lines

def draw_citation_bottom_right(c: canvas.Canvas, page_w: float, page_h: float, font_name: str):
    """Cita en esquina inferior derecha, 1 cm de ambos bordes, alineada a la derecha, con wrap."""
    fs = 8.0
    c.setFont(font_name, fs)
    c.setFillColor(colors.black)

    # Área disponible (dejamos 1 cm márgenes)
    right_x = page_w - CM_TO_PT
    bottom_y = CM_TO_PT
    max_width = page_w - 2*CM_TO_PT  # ancho útil entre márgenes de 1 cm

    lines = wrap_right_aligned_lines(CITATION_TEXT, font_name, fs, max_width)
    # Dibujar desde abajo hacia arriba
    line_height = fs + 2
    y = bottom_y
    for line in lines:
        c.drawRightString(right_x, y, line)
        y += line_height

# ----------------------------
# Generación del PDF
# ----------------------------

def generate_pdf(df: pd.DataFrame, output_path: str, page_size: str = "letter", label_position: str = "coords"):
    # Tamaño de página
    if page_size.lower() == "letter":
        page_w, page_h = letter
    elif page_size.lower() == "a4":
        page_w, page_h = A4
    else:
        page_w, page_h = letter

    # Normaliza Y a 0–1 si llega en 0–100
    df["Y"] = normalize_Y_auto(df["Y"])

    # Convertir a sRGB
    def row_to_rgb(row):
        X, Y, Z = xyY_to_XYZ(row["x"], row["y"], row["Y"])
        return XYZ_to_sRGB(X, Y, Z)

    df[["R","G","B"]] = df.apply(lambda r: pd.Series(row_to_rgb(r)), axis=1)

    # Orden/paginación por hue token (familia y step)
    df["HueToken"] = df["h"].apply(parse_hue_token)
    unique_tokens = sorted(df["HueToken"].unique(), key=lambda t: t.sort_key())

    c = canvas.Canvas(output_path, pagesize=(page_w, page_h))
    layout = PageLayout()
    citation_font = ensure_arial_or_fallback()

    for token in unique_tokens:
        subset = df[df["HueToken"] == token].copy()
        if subset.empty:
            continue

        # Orden de parches: Value↓, Chroma→
        subset.sort_values(by=["V","C"], ascending=[False, True], inplace=True)

        values_sorted = sorted(subset["V"].unique(), reverse=True)
        chromas_sorted = sorted(subset["C"].unique())
        max_rows = len(values_sorted)
        max_cols = len(chromas_sorted)

        box = layout.fit_grid(page_w, page_h, max_cols, max_rows, label_position)

        # Título
        title = f"Hue: {token.step:g}{token.family}  —  (Value ↓ vs Chroma →)"
        draw_title(c, layout, page_w, page_h, title)

        if label_position == "coords":
            # Con ejes
            row_step = box + layout.gutter_y
            col_step = box + layout.gutter_x
            start_x = layout.margin + layout.value_axis_w
            top_y = page_h - layout.margin - layout.title_gap
            start_y = top_y - box
            grid_bottom_y = start_y - (max_rows - 1) * row_step

            v_index = {v: i for i, v in enumerate(values_sorted)}
            c_index = {cc: j for j, cc in enumerate(chromas_sorted)}

            # Parches
            for _, row in subset.iterrows():
                i = v_index[row["V"]]
                j = c_index[row["C"]]
                x = start_x + j * col_step
                y = start_y - i * row_step
                rgb = (float(row["R"]), float(row["G"]), float(row["B"]))
                draw_patch(c, x, y, box, rgb)

            # Ejes (Value izquierda, Chroma abajo, 14 pt + línea divisoria + gap extra)
            draw_value_axis_labels(c, layout, start_x, start_y, box, row_step, values_sorted)
            draw_chroma_axis_labels(c, layout, start_x, grid_bottom_y, box, col_step, chromas_sorted)

        else:
            # Etiqueta por parche debajo
            row_step = box + layout.label_height + layout.gutter_y
            col_step = box + layout.gutter_x
            start_x = layout.margin
            top_y = page_h - layout.margin - layout.title_gap
            start_y = top_y - box - layout.label_height

            v_index = {v: i for i, v in enumerate(values_sorted)}
            c_index = {cc: j for j, cc in enumerate(chromas_sorted)}

            for _, row in subset.iterrows():
                i = v_index[row["V"]]
                j = c_index[row["C"]]
                x = start_x + j * col_step
                y = start_y - i * row_step
                rgb = (float(row["R"]), float(row["G"]), float(row["B"]))
                label = f"{row['h']} {int(row['V'])}/{int(row['C'])}"
                draw_patch_below_label(c, x, y, box, rgb, label, layout)

        # Citación (abajo derecha, con wrap y alineación derecha)
        draw_citation_bottom_right(c, page_w, page_h, citation_font)

        c.showPage()

    c.save()

# ----------------------------
# Lectura del archivo de entrada
# ----------------------------

def read_renotation_table(path: str) -> pd.DataFrame:
    """
    Lee un archivo con columnas: h V C x y Y
    - Ignora líneas vacías y comentarios que empiezan con '#'
    - Acepta separadores por espacios o tabs
    """
    rows = []
    header_seen = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_seen and re.match(r"(?i)^\s*h\s+v\s+c\s+x\s+y\s+y\s*$", line):
                header_seen = True
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 6:
                continue
            h, V, C, x, y, Y = parts[:6]
            rows.append((h, int(float(V)), int(float(C)), float(x), float(y), float(Y)))

    if not rows:
        raise ValueError("No se encontraron filas válidas (h V C x y Y).")

    df = pd.DataFrame(rows, columns=["h","V","C","x","y","Y"])
    return df

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generar cartilla PDF Munsell desde renotation (xyY).")
    ap.add_argument("--input", required=True, help="Archivo de entrada con columnas: h V C x y Y")
    ap.add_argument("--output", default="munsell_atlas.pdf", help="Ruta del PDF de salida")
    ap.add_argument("--page-size", choices=["letter","a4"], default="letter", help="Tamaño de página")
    ap.add_argument("--profile", choices=["sRGB"], default="sRGB", help="Perfil de salida (actualmente sRGB)")
    ap.add_argument("--label-position", choices=["coords","below"], default="coords",
                    help="coords: ejes Value/Chroma, sin etiquetas por parche | below: etiqueta por parche debajo, sin ejes")
    args = ap.parse_args()

    df = read_renotation_table(args.input)
    generate_pdf(df, args.output, page_size=args.page_size, label_position=args.label_position)
    print(f"✅ PDF generado: {args.output}  (label-position={args.label_position})")

if __name__ == "__main__":
    main()
