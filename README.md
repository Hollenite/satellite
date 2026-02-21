# ☀️ AI-Assisted Rooftop Solar Pre-Assessment Tool

> **Pre-assessment only** — this tool provides rough estimates for initial screening.
> It is NOT a substitute for professional site survey or engineering design.

---

## Quick Start

### Prerequisites
- **Python 3.10+** (pyproj requires 3.10+)
- Windows / Linux / macOS

### Setup

```powershell
# 1. Create venv
cd d:\PROJECTS\AIML-Hackathon-21feb\satellite
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install core deps
pip install numpy matplotlib Pillow shapely pyproj rasterio streamlit

# 3. Install PyTorch (CPU — fastest for tonight)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install segmentation model
pip install segmentation-models-pytorch

# 5. Optional (skip if any fail)
pip install geopandas pandas scikit-image
```

#### ⚠️ Windows GDAL / rasterio issues

If `pip install rasterio` fails:

```powershell
# Option A: Use conda (recommended)
conda install -c conda-forge rasterio

# Option B: Pre-built wheel
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# pip install rasterio‑<version>‑cp310‑cp310‑win_amd64.whl
```

#### Fallback minimal install (if geopandas fails)
The pipeline works without geopandas. Core stack: `rasterio + shapely + pyproj + json`

---

## Demo Modes

### 🟢 Demo Mode A — Ground-Truth Pipeline (recommended first)

Uses existing labels/masks as "fake predictions" to prove the full product pipeline:

```
satellite image → GT mask → vectorize → solar estimate → overlay
```

**How to run:**
```powershell
streamlit run app.py
# Select "Synthetic demo" or load a SpaceNet tile with matching labels
```

**What this proves:** Geospatial pipeline correctness, UX flow, export functionality.

### 🔵 Demo Mode B — Model Inference Pipeline

Trains a U-Net on SpaceNet data, then runs inference:

```
satellite image → U-Net → predicted mask → vectorize → solar estimate → overlay
```

**How to run:**
```powershell
# Train
python -m src.train --data_dir data/raw --epochs 10

# Infer
python -m src.infer --image data/raw/images/tile.tif --checkpoint checkpoints/best.pth

# Then run Streamlit with the predicted mask
```

---

## Data Setup (SpaceNet)

### Minimal setup — one tile

```
satellite/data/raw/
├── images/
│   └── tile_001.tif          # RGB or multispectral GeoTIFF
├── masks/                     # Option A: raster masks
│   └── tile_001.tif           # Binary building mask (0/1)
└── labels/                    # Option B: vector labels
    └── tile_001.geojson       # Building footprint polygons
```

Download from: [spacenet.ai/datasets](https://spacenet.ai/datasets/)

### Synthetic mode (zero data needed)

The app includes a built-in synthetic tile generator. Select **"Synthetic demo"** in the sidebar — no downloads required.

---

## Project Structure

```
satellite/
├── app.py                  # Streamlit demo app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── data.py             # Image/mask loading, RGB prep, synthetic gen, dataset
│   ├── vectorize.py        # Mask → polygons → GeoJSON + sidecar metadata
│   ├── viz.py              # Overlay visualization, side-by-side figures
│   ├── estimate.py         # Solar estimation with configurable assumptions
│   ├── train.py            # U-Net training pipeline
│   ├── infer.py            # Model inference + mask export
│   └── utils.py            # CRS/reprojection/alignment/area helpers
├── data/raw/               # Place SpaceNet tiles here
├── outputs/                # Generated masks, GeoJSONs, PNGs
└── checkpoints/            # Model weights
```

---

## GeoJSON & CRS Handling

Output GeoJSON files are written as **standard FeatureCollections** (no embedded CRS).
A **sidecar metadata file** is saved alongside:

```
outputs/footprints.geojson       ← polygon geometries
outputs/footprints.meta.json     ← CRS, transform, coordinate type
```

> ⚠️ GeoJSON consumers (geojson.io, QGIS) may assume WGS84.
> Use the sidecar `.meta.json` to interpret coordinates correctly.

---

## Solar Estimation Assumptions

| Parameter | Default | Notes |
|-----------|---------|-------|
| Roof usability | 0.65 | Excludes tanks, stairs, shadows |
| Panel density | 0.18 kW/m² | Conservative for Indian market |
| Performance ratio | 0.78 | Inverter + wiring + soiling losses |
| Monthly gen factor | 110 kWh/kW | India avg varies 100–130 by region |

All parameters are **configurable** in the Streamlit sidebar.

> [!IMPORTANT]
> **Formula clarity:** `monthly_generation_kwh_per_kw` is a **delivered** value — it already accounts for `performance_ratio`, inverter losses, soiling, and temperature derating. Do NOT multiply by `performance_ratio` again. The formula is:
> ```
> system_kw = usable_area × panel_power_density
> monthly_kwh = system_kw × monthly_generation_kwh_per_kw
> annual_kwh = monthly_kwh × 12  (or override via annual_generation_kwh_per_kw)
> ```

### Limitations
- No shading analysis
- No roof tilt/azimuth modelling
- No structural load assessment
- No state/DISCOM net metering logic
- Irradiance values are regional averages

---

## India-Specific Adaptation Notes

### Domain Gap
SpaceNet imagery (US/European cities) differs from Indian rooftops:
- Different building styles, colours, and materials
- Rooftop clutter: water tanks, satellite dishes, parapets, washing lines
- Tighter building density in urban areas

### Fine-Tuning Path
1. Collect 50–100 annotated tiles of Indian cities (Google Earth + manual labelling)
2. Fine-tune the pretrained U-Net on this data
3. Validate on held-out Indian tiles
4. Adjust `SolarConfig` defaults per state/region

### Product Positioning
- Frame as "AI-assisted pre-assessment" — not final engineering
- Align with PM Surya Ghar Muft Bijli Yojana workflow
- State-specific customizations later: irradiance maps, net metering rates, subsidy amounts

---

## Debugging Checklist

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `rasterio` won't install | Missing GDAL | Use conda or pre-built wheel |
| Polygons don't overlap image | CRS mismatch | Check `print_raster_info()`, ensure transform is passed |
| Area in degrees² | Geographic CRS | pyproj auto-reprojects; check `pyproj` is installed |
| Model produces blank mask | Insufficient training / wrong threshold | Lower threshold to 0.3, train more epochs |
| Streamlit crashes on plot | Matplotlib threading | `matplotlib.use("Agg")` is already set |
| GeoJSON looks wrong in QGIS | CRS assumption mismatch | Read `.meta.json` sidecar for true CRS |

---

## CLI Quick Reference

```powershell
# Streamlit app
streamlit run app.py

# Train model
python -m src.train --data_dir data/raw --epochs 10 --batch_size 4

# Run inference
python -m src.infer --image data/raw/images/tile.tif --checkpoint checkpoints/best.pth

# Quick data sanity check
python -c "from src.data import load_image; img,t,c = load_image('data/raw/images/tile.tif'); print(img.shape, c, t)"

# Quick synthetic test
python -c "from src.data import generate_synthetic_tile; img,m,t,c = generate_synthetic_tile(); print(img.shape, m.shape, t, c)"
```
