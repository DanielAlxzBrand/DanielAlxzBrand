#!/usr/bin/env python3
"""
generate_neofetch.py
Genera assets/neofetch.svg: panel tipo neofetch con la foto convertida a ASCII
(izquierda) y la informacion del CV (derecha), para el README de perfil de GitHub.

Uso:
    python scripts/generate_neofetch.py                       # usa CONFIG
    python scripts/generate_neofetch.py --photo mi_foto.png   # otra foto
    python scripts/generate_neofetch.py --preview             # ademas escribe preview.png / preview_light.png

Dependencias:
    pip install pillow numpy opencv-python-headless
    pip install rembg onnxruntime      # opcional: aisla el sujeto (si falla, usa mascara elipse)
    pip install cairosvg               # opcional: solo para --preview
"""
import argparse
import math
import os
import sys
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

# =============================================================================
# CONFIG  (edita aqui)
# =============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHOTO = os.path.join(ROOT, "assets", "FOTO_CV_01.png")   # foto fuente (no hace falta subirla al repo)
OUT_SVG = os.path.join(ROOT, "assets", "neofetch.svg")
OUT_PREVIEW = os.path.join(ROOT, "preview.png")

# --- ASCII ---
COLS = 72                    # columnas (62-72)
ROWS = 40                    # filas (34-40)
RAMP = " .:-=+*#%@"          # de menos a mas denso
DENSITY = "glow"             # "glow": zonas claras = mas denso (ideal en fondo oscuro) | "ink": zonas oscuras = mas denso
GAMMA = 1.15                 # >1 aclara tonos medios (piel), resalta sombras
DETAIL = 1.0                 # realce de detalle fino (ojos, cejas, nariz, boca); 0 = desactivar
TONE_PCT = (1, 99)           # percentiles de luminancia (dentro del sujeto) que fijan negro y blanco
MAX_DENSITY = 0.95           # tope de densidad (evita bloques planos de @ en el pelo)
CLAHE_CLIP = 2.2             # contraste local (0 = desactivar)
SHARPEN = 0.9                # realce de bordes antes de muestrear
MASK_METHOD = "auto"         # "auto" (rembg y si falla elipse) | "rembg" | "ellipse"
REMBG_MODEL = "u2net_human_seg"
CROP_SCALE = 1.70            # ancho del recorte = ancho del rostro * CROP_SCALE
FACE_Y = 0.675               # posicion vertical del centro del rostro en el recorte (0-1)
ELLIPSE = dict(cx=0.5, cy=0.50, rx=0.50, ry=0.56, fade_start=0.78)   # silueta redondeada
BOTTOM_FADE = (0.86, 0.40)   # (desde que fraccion de alto empieza el fade, cuanto se atenua al final)
HAIR_BOTTOM = 0.40            # (solo modo glow) fraccion de alto hasta donde se considera pelo
HAIR_FLOOR = 0.40             # (solo modo glow) densidad minima del pelo oscuro, para que no desaparezca
EDGE_MIN_DENSITY = 0.30      # densidad minima en el borde de la silueta para que se lea el contorno

# --- Geometria del SVG ---
VIEW_W, VIEW_H = 900, 520
ASCII_X, ASCII_Y = 22, 44    # esquina superior izquierda del ASCII
CELL_W, CELL_H = 5.4, 10.8   # tamano de celda (px)  -> relacion 0.5
INFO_X = 430                 # x de la columna de informacion
INFO_FONT = 12
INFO_CW = 7.2                # ancho de caracter estimado para calcular columnas
LABEL_CHARS = 10             # ancho de columna de etiquetas
LINE_H = 20.5
MAX_VALUE_CHARS = 53

FONT_STACK = '"SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace'

# --- Paleta (variables CSS; ASCII: p0 = menos denso ... p7 = mas denso) ---
PALETTE = {
    "dark": dict(
        bg="#0d1117", border="#30363d", orange="#ff9e64", label="#58a6ff",
        value="#e6edf3", dim="#8b949e",
        p=["#7c5cbf", "#8f63cf", "#a66ae0", "#bd6fe8", "#d36fe8", "#e870d8", "#f576c4", "#ff8ecf"],
        c=["#484f58", "#ff7b72", "#3fb950", "#d29922", "#58a6ff", "#bc8cff", "#39c5cf", "#e6edf3"],
    ),
    "light": dict(
        bg="#ffffff", border="#d0d7de", orange="#bc4c00", label="#0969da",
        value="#24292f", dim="#57606a",
        p=["#d2a8ef", "#c590e8", "#b87ade", "#aa63d4", "#a24ac3", "#a0399f", "#9d2a86", "#86198f"],
        c=["#57606a", "#cf222e", "#1a7f37", "#9a6700", "#0969da", "#8250df", "#1b7c83", "#24292f"],
    ),
}

# --- Datos (todos tomados del CV; no se incluye telefono por privacidad) ---
HEADER = "DanielAlxzBrand@github"
ROWS_INFO = [
    ("Uptime",    "Estudiante 5.º semestre (2024 – 2028)"),
    ("Host",      "Universidad Autónoma de Occidente (Cali)"),
    ("Kernel",    "Ingeniería de Datos, IA, ETL, Modelado dimensional"),
    ("IDE.Tools", "Claude Code, Git, Docker, Power BI, Excel"),
    ("Lang.Code", "Python, SQL, JavaScript"),
    ("Lang.Data", "pandas, NumPy, dbt-core, DuckDB, Airflow, PostgreSQL, MySQL, MongoDB"),
    ("Lang.Real", "Español (nativo), English (A2)"),
    ("Proj.01",   "dbt-kimball-portfolio · dbt-core + DuckDB"),
    ("Proj.02",   "hatbot-faq-nlp · búsqueda híbrida NLP"),
    ("Proj.03",   "SIGEU_2 · MongoDB + FastAPI"),
    ("Learning",  "Power BI Avanzado (en curso)"),
]
CONTACT_TITLE = "- Contact"
CONTACT = [
    ("Email",    "danielbrand2343@gmail.com"),
    ("LinkedIn", "in/daniel-alexander-b-9a1b98210"),
    ("GitHub",   "github.com/DanielAlxzBrand"),
]
# =============================================================================


# ----------------------------- foto -> ASCII --------------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def isolate_subject(img):
    """Devuelve (alpha float 0-1 | None, metodo)."""
    if MASK_METHOD in ("auto", "rembg"):
        try:
            from rembg import new_session, remove
            out = remove(img, session=new_session(REMBG_MODEL))
            alpha = np.asarray(out.split()[-1], dtype=np.float32) / 255.0
            return alpha, "rembg(%s)" % REMBG_MODEL
        except Exception as exc:  # modelo no disponible, sin red, etc.
            print("[aviso] rembg no disponible (%s); uso mascara elipse" % exc, file=sys.stderr)
    return None, "elipse"


def detect_face(gray, alpha=None):
    """Caja (x, y, w, h) del rostro (cejas-menton).
    1) Haar de OpenCV 4.x si existe; 2) estimacion a partir de la mascara del sujeto; 3) recorte central."""
    h, w = gray.shape
    try:
        import cv2
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = casc.detectMultiScale(gray, 1.1, 5, minSize=(int(w * 0.12), int(w * 0.12)))
        if len(faces):
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3])), "opencv-haar"
    except Exception as exc:   # OpenCV 5.x ya no trae CascadeClassifier
        print("[aviso] Haar no disponible (%s)" % exc, file=sys.stderr)
    if alpha is not None:
        m = alpha > 0.5
        widths = m.sum(1)
        rows = np.where(widths > w * 0.01)[0]
        if len(rows):
            y_top = int(rows[0])
            head_w = int(widths[y_top:y_top + int(h * 0.30)].max())      # ancho maximo (pelo) de la cabeza
            ys = slice(y_top + int(0.4 * head_w), y_top + head_w)
            xs = np.where(m[ys].any(0))[0]
            cx = int((xs.min() + xs.max()) / 2)
            fw = int(0.78 * head_w)
            cy = int(y_top + 0.95 * head_w)
            return (cx - fw // 2, cy - fw // 2, fw, fw), "mascara-cabeza"
    fw = int(w * 0.46)
    return (w // 2 - fw // 2, int(h * 0.30), fw, fw), "recorte-central"


def photo_to_grid(photo_path):
    """Devuelve (chars[ROWS][COLS] de la rampa, densidad[ROWS,COLS] 0-1, info de metodos)."""
    import cv2

    img = Image.open(photo_path).convert("RGB")
    rgb = np.asarray(img)
    gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alpha, mask_method = isolate_subject(img)
    (fx, fy, fw, fh), face_method = detect_face(gray0, alpha)

    # recorte retrato (cabeza + hombros) centrado en el rostro
    crop_w = fw * CROP_SCALE
    crop_h = crop_w * (ROWS * 2.0) / COLS          # compensa celda ~0.5
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    x0, y0 = int(round(cx - crop_w / 2)), int(round(cy - crop_h * FACE_Y))
    x1, y1 = x0 + int(round(crop_w)), y0 + int(round(crop_h))
    pad = int(max(crop_w, crop_h))
    gray_p = cv2.copyMakeBorder(gray0, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = gray_p[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    if alpha is not None:
        a_p = cv2.copyMakeBorder(alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        a_crop = a_p[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    else:
        a_crop = np.ones_like(crop, dtype=np.float32)

    # supersampling + contraste local + realce
    ss = 4
    big = cv2.resize(crop, (COLS * ss, ROWS * ss), interpolation=cv2.INTER_AREA)
    if CLAHE_CLIP > 0:
        big = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(6, 6)).apply(big)
    if SHARPEN > 0:
        blur = cv2.GaussianBlur(big, (0, 0), 2.2)
        big = cv2.addWeighted(big, 1 + SHARPEN, blur, -SHARPEN, 0)
    big = big.astype(np.float32)
    if DETAIL > 0:   # diferencia de gaussianas: acentua rasgos finos
        dog = cv2.GaussianBlur(big, (0, 0), ss * 0.55) - cv2.GaussianBlur(big, (0, 0), ss * 1.7)
        big = np.clip(big + DETAIL * dog, 0, 255)
    big = big.astype(np.uint8)
    g = cv2.resize(big, (COLS, ROWS), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a = cv2.resize(a_crop.astype(np.float32), (COLS, ROWS), interpolation=cv2.INTER_AREA)

    # silueta redondeada (vignette elíptica)
    yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
    nx = (xx + 0.5) / COLS
    ny = (yy + 0.5) / ROWS
    e = ELLIPSE
    r = np.sqrt(((nx - e["cx"]) / e["rx"]) ** 2 + ((ny - e["cy"]) / e["ry"]) ** 2)
    vign = 1.0 - smoothstep(e["fade_start"], 1.0, r)
    if alpha is None:   # sin rembg: la elipse hace de silueta
        a = (r < 1.0).astype(np.float32)
    inside = (a > 0.5) & (r < 1.0)

    # normaliza dentro del sujeto
    vals = g[inside]
    lo, hi = np.percentile(vals, TONE_PCT[0]), np.percentile(vals, TONE_PCT[1])
    gn = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    d = (1.0 - gn) if DENSITY == "ink" else gn
    d = (d ** GAMMA) * MAX_DENSITY
    if DENSITY == "glow":   # el pelo oscuro conserva densidad (con textura por sus reflejos)
        hair_w = (1.0 - smoothstep(HAIR_BOTTOM - 0.06, HAIR_BOTTOM + 0.04, ny)) * (1.0 - smoothstep(0.30, 0.55, gn))
        hm = (hair_w > 0.5) & inside
        if hm.sum() > 20:   # reestira el contraste solo dentro del pelo para que se vean los rizos
            hl, hh = np.percentile(g[hm], 4), np.percentile(g[hm], 97)
            gh = np.clip((g - hl) / (hh - hl + 1e-6), 0, 1) ** 0.85
        else:
            gh = gn
        d = np.maximum(d, hair_w * (HAIR_FLOOR * 0.6 + 0.62 * gh))

    # atenua la parte baja (ropa) y el borde para que la silueta quede suave
    b0, bk = BOTTOM_FADE
    d *= 1.0 - bk * smoothstep(b0, 1.0, ny)
    d *= 0.25 + 0.75 * vign
    # contorno minimo para que la silueta se lea
    edge = inside & (vign < 0.9)
    d = np.where(edge, np.maximum(d, EDGE_MIN_DENSITY * (0.4 + 0.6 * vign)), d)
    d = np.where(inside, d, 0.0)
    d = np.clip(d, 0, 1)

    n = len(RAMP) - 1
    idx = np.rint(d * n).astype(int)
    idx[~inside] = 0
    chars = [[RAMP[i] for i in row] for row in idx]
    return chars, d, "mascara=%s; rostro=%s" % (mask_method, face_method)


# ------------------------------- SVG ----------------------------------------
NB = "\u00a0"   # espacio no separable: no se colapsa nunca


def ascii_svg(chars, dens):
    out = []
    for r, row in enumerate(chars):
        s = "".join(row)
        stripped = s.strip(" ")
        if not stripped:
            continue
        first = len(s) - len(s.lstrip(" "))
        last = len(s.rstrip(" "))
        runs, cur_lv, cur = [], None, ""
        for c in range(first, last):
            ch = row[c]
            if ch == " ":
                cur += NB
                continue
            lv = min(7, int(dens[r][c] * 8))
            if cur_lv is None or lv == cur_lv:
                cur_lv = lv if cur_lv is None else cur_lv
                cur += ch
            else:
                runs.append((cur_lv, cur))
                cur_lv, cur = lv, ch
        if cur:
            runs.append((cur_lv if cur_lv is not None else 0, cur))
        x = ASCII_X + first * CELL_W
        y = ASCII_Y + r * CELL_H + CELL_H * 0.8
        width = (last - first) * CELL_W
        tsp = "".join('<tspan class="p%d">%s</tspan>' % (lv, escape(t)) for lv, t in runs)
        out.append('<text x="%.1f" y="%.1f" textLength="%.1f" lengthAdjust="spacing" class="a">%s</text>'
                   % (x, y, width, tsp))
    return "\n".join(out)


def wrap_value(text, limit):
    if len(text) <= limit:
        return [text]
    parts = [p.strip() for p in text.split(",")]
    lines, cur = [], ""
    for p in parts:
        cand = (cur + ", " + p) if cur else p
        if len(cand) + 1 <= limit:
            cur = cand
        else:
            lines.append(cur + ",")
            cur = p
    lines.append(cur)
    return lines


def info_svg():
    out = []
    lab_x = INFO_X
    val_x = INFO_X + LABEL_CHARS * INFO_CW
    y = ASCII_Y + 14
    out.append('<text x="%.1f" y="%.1f" class="i orange b">%s</text>' % (lab_x, y, escape(HEADER)))
    y += LINE_H * 0.95
    out.append('<text x="%.1f" y="%.1f" textLength="%.1f" lengthAdjust="spacing" class="i dim">%s</text>'
               % (lab_x, y, len(HEADER) * INFO_CW, "-" * len(HEADER)))
    y += LINE_H

    def block(rows, y):
        for label, value in rows:
            lines = wrap_value(value, MAX_VALUE_CHARS)
            out.append('<text x="%.1f" y="%.1f" class="i label b">%s</text>' % (lab_x, y, escape(label)))
            for k, ln in enumerate(lines):
                out.append('<text x="%.1f" y="%.1f" class="i value">%s</text>' % (val_x, y, escape(ln)))
                if k < len(lines) - 1:
                    y += LINE_H
            y += LINE_H
        return y

    y = block(ROWS_INFO, y)
    y += LINE_H * 0.15
    out.append('<text x="%.1f" y="%.1f" class="i orange b">%s</text>' % (lab_x, y, escape(CONTACT_TITLE)))
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="rule"/>'
               % (lab_x + (len(CONTACT_TITLE) + 1) * INFO_CW, y - 4,
                  VIEW_W - 22, y - 4))
    y += LINE_H
    y = block(CONTACT, y)
    # fila de 8 cuadros de color
    y += 6
    sq, gap = 30, 4
    for i in range(8):
        out.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="2" class="c%d"/>'
                   % (lab_x + i * (sq + gap), y, sq, 18, i))
    return "\n".join(out), y + 18


def css(theme):
    """theme = None -> variables + @media; 'dark'/'light' -> valores literales (para previsualizar)."""
    P = PALETTE
    def decl(t):
        d = P[t]
        s = "--bg:%s;--border:%s;--orange:%s;--label:%s;--value:%s;--dim:%s;" % (
            d["bg"], d["border"], d["orange"], d["label"], d["value"], d["dim"])
        s += "".join("--p%d:%s;" % (i, c) for i, c in enumerate(d["p"]))
        s += "".join("--c%d:%s;" % (i, c) for i, c in enumerate(d["c"]))
        return s
    base = (
        "text{font-family:%s;white-space:pre}"
        ".a{font-size:9px}.i{font-size:%dpx}.b{font-weight:700}"
        ".orange{fill:var(--orange)}.label{fill:var(--label)}.value{fill:var(--value)}.dim{fill:var(--dim)}"
        ".panel{fill:var(--bg);stroke:var(--border);stroke-width:1.5}"
        ".rule{stroke:var(--dim);stroke-width:1;stroke-dasharray:2 3;opacity:.7}"
    ) % (("'Liberation Mono', 'DejaVu Sans Mono', monospace" if theme else FONT_STACK.replace('"', "'")), INFO_FONT)
    base += "".join(".p%d{fill:var(--p%d)}" % (i, i) for i in range(8))
    base += "".join(".c%d{fill:var(--c%d)}" % (i, i) for i in range(8))
    if theme is None:
        return ":root{%s}@media (prefers-color-scheme:light){:root{%s}}" % (decl("dark"), decl("light")) + base
    return ":root{%s}" % decl(theme) + base


def build_svg(chars, dens, theme=None):
    info, bottom = info_svg()
    style = css(theme)
    if theme is not None:   # cairosvg no resuelve var(): se sustituyen los valores
        d = PALETTE[theme]
        rep = {"--bg": d["bg"], "--border": d["border"], "--orange": d["orange"], "--label": d["label"],
               "--value": d["value"], "--dim": d["dim"]}
        rep.update({"--p%d" % i: c for i, c in enumerate(d["p"])})
        rep.update({"--c%d" % i: c for i, c in enumerate(d["c"])})
        for k, v in sorted(rep.items(), key=lambda kv: -len(kv[0])):
            style = style.replace("var(%s)" % k, v)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
        'aria-label="Panel tipo neofetch de DanielAlxzBrand con retrato en ASCII e información de perfil">\n'
        '<title>DanielAlxzBrand@github</title>\n<style>%s</style>\n'
        '<rect class="panel" x="1" y="1" width="%d" height="%d" rx="14"/>\n'
        '<g xml:space="preserve">\n%s\n</g>\n%s\n</svg>\n'
    ) % (VIEW_W, VIEW_H, VIEW_W, VIEW_H, style, VIEW_W - 2, VIEW_H - 2, ascii_svg(chars, dens), info)


def main():
    global PHOTO, OUT_SVG, OUT_PREVIEW
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default=PHOTO)
    ap.add_argument("--out", default=OUT_SVG)
    ap.add_argument("--preview", action="store_true", help="escribe preview.png (oscuro) y preview_light.png")
    ap.add_argument("--ascii-txt", default=None, help="guarda el ASCII crudo en un .txt")
    a = ap.parse_args()

    chars, dens, info = photo_to_grid(a.photo)
    print("[ok] %s | %dx%d" % (info, COLS, ROWS))
    if a.ascii_txt:
        open(a.ascii_txt, "w", encoding="utf8").write("\n".join("".join(r) for r in chars))
    svg = build_svg(chars, dens)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf8").write(svg)
    print("[ok] %s (%.1f KB)" % (a.out, len(svg.encode("utf8")) / 1024))

    if a.preview:
        import cairosvg
        base = os.path.splitext(OUT_PREVIEW)[0]
        for theme, path in (("dark", OUT_PREVIEW), ("light", base + "_light.png")):
            cairosvg.svg2png(bytestring=build_svg(chars, dens, theme).encode("utf8"),
                             write_to=path, output_width=VIEW_W * 2)
            print("[ok] preview", path)


if __name__ == "__main__":
    main()
