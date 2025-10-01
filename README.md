# munsell_color_charts

This repository provides Python tools and ready-to-print PDF atlases of the **Munsell color system**, generated from the official **Munsell Renotation Data (1943)** published by the **National Institute of Standards and Technology (NIST, USA)**.

---

## 🇬🇧 English

### Munsell Color Charts (Renotation 1943, NIST)

This repository provides Python tools and ready-to-print PDF atlases of the **Munsell color system**, generated from the official **Munsell Renotation Data (1943)** published by the National Institute of Standards and Technology (NIST, USA).

### Includes:
- Full dataset of Munsell Renotation values (hue, value, chroma, xyY).  
- Python script (`munsell_cartilla.py`) to generate printable color charts in PDF.  
- Two output modes:
  - `coords`: Charts with Value (vertical axis) and Chroma (horizontal axis).  
  - `below`: Labels under each patch (Hue Value/Chroma).  
- Example outputs: `atlas_coords.pdf`, `atlas_below.pdf`.  

These charts are useful for **botanical, agricultural, design, and scientific characterization**.

### Quick Start
With Python installed in your system, run:
```bash
pip install reportlab pandas numpy pillow

# Generate with Value/Chroma axes
python munsell_cartilla.py --input renotation.txt --output atlas_coords.pdf --label-position coords

# Generate with labels under each patch
python munsell_cartilla.py --input renotation.txt --output atlas_below.pdf --label-position below



## 🇪🇸 Español

### Cartillas de Colores Munsell (Renotación 1943, NIST)

Este repositorio ofrece herramientas en Python y atlas en PDF listos para imprimir del **sistema de colores Munsell**, generados a partir de los datos oficiales de la **Renotación de Munsell (1943)** publicados por el *National Institute of Standards and Technology (NIST, EE.UU.)*.

### Incluye:
- Conjunto completo de valores de renotación Munsell (matiz, valor, croma, xyY).  
- Script en Python (`munsell_cartilla.py`) para generar cartillas de colores imprimibles en PDF.  
- Dos modos de salida:
  - `coords`: Cartillas con Value (eje vertical) y Chroma (eje horizontal).  
  - `below`: Etiquetas debajo de cada parche (Hue Value/Chroma).  
- Ejemplos de salida: `atlas_coords.pdf`, `atlas_below.pdf`.  

Estas cartillas son útiles para la **caracterización botánica, agrícola, de diseño y científica**.

### Inicio rápido
Con Python instalado en tu sistema, ejecuta:
```bash
pip install reportlab pandas numpy pillow

# Generar con ejes Value/Chroma
python munsell_cartilla.py --input renotation.txt --output atlas_coords.pdf --label-position coords

# Generar con etiquetas debajo de cada parche
python munsell_cartilla.py --input renotation.txt --output atlas_below.pdf --label-position below
