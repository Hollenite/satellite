"""
Streamlit demo app — AI-Assisted Rooftop Solar Pre-Assessment.

Run with:
    cd d:\\PROJECTS\\AIML-Hackathon-21feb\\satellite
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data import (
    generate_synthetic_tile,
    load_image,
    load_or_create_mask,
    prepare_display_rgb,
)
from src.estimate import SolarConfig, estimate_all_roofs, format_report
from src.utils import (
    check_crs_units,
    ensure_output_dir,
    print_raster_info,
    validate_polygon_raster_alignment,
)
from src.vectorize import mask_to_polygons, polygons_to_geojson
from src.viz import annotate_polygons, overlay_polygons, side_by_side


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="☀️ Solar Rooftop Pre-Assessment",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ AI-Assisted Rooftop Solar Pre-Assessment")
st.caption(
    "Upload a satellite tile or use synthetic data to estimate rooftop solar potential. "
    "This is a **pre-assessment tool** — not a substitute for professional site survey."
)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

data_mode = st.sidebar.radio(
    "Data Mode",
    ["Synthetic demo", "SpaceNet tile"],
    help="Synthetic mode generates fake buildings for a zero-data demo.",
)

st.sidebar.subheader("Solar Assumptions")
usability = st.sidebar.slider("Roof usability factor", 0.3, 0.9, 0.65)
panel_density = st.sidebar.slider("Panel power density (kW/m²)", 0.10, 0.25, 0.18)
perf_ratio = st.sidebar.slider("Performance ratio", 0.60, 0.95, 0.78)
monthly_gen = st.sidebar.slider("Monthly gen (kWh/kW)", 70.0, 150.0, 110.0)

config = SolarConfig(
    roof_usability_factor=usability,
    panel_power_density_kw_per_m2=panel_density,
    performance_ratio=perf_ratio,
    monthly_generation_kwh_per_kw=monthly_gen,
)

st.sidebar.subheader("Vectorization")
min_area = st.sidebar.number_input("Min polygon area (filter)", value=25.0, min_value=1.0)
simplify_tol = st.sidebar.slider("Simplify tolerance", 0.0, 5.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

image = mask = transform = crs = None
warnings_list: list[str] = []

if data_mode == "Synthetic demo":
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0)
    n_buildings = st.sidebar.slider("Num buildings", 3, 20, 8)

    image, mask, transform, crs = generate_synthetic_tile(
        num_buildings=n_buildings, seed=int(seed)
    )
    warnings_list.append("⚠️ Using synthetic data — no real CRS. Area estimates are pixel-based.")

else:
    # SpaceNet tile mode
    st.sidebar.subheader("Data Paths")
    data_dir = Path("data/raw")

    # Auto-discover tiles
    img_dir = data_dir / "images"
    if img_dir.exists():
        available = sorted(img_dir.glob("*.tif"))
        if available:
            selected = st.sidebar.selectbox(
                "Select tile",
                available,
                format_func=lambda p: p.name,
            )
            if selected:
                image, transform, crs = load_image(selected)

                # Look for matching mask/label
                mask_path = data_dir / "masks" / selected.name
                label_stem = selected.stem
                label_path = data_dir / "labels" / f"{label_stem}.geojson"

                if mask_path.exists():
                    mask, _, _ = load_or_create_mask(selected, mask_path)
                elif label_path.exists():
                    mask, _, _ = load_or_create_mask(selected, label_path)
                else:
                    st.warning(
                        f"No mask/label found for {selected.name}. "
                        f"Looking in masks/ or labels/ directories."
                    )
        else:
            st.info("No .tif files found in `data/raw/images/`. Place SpaceNet tiles there.")
    else:
        st.info(
            "Create `data/raw/images/` and `data/raw/masks/` (or `labels/`) directories, "
            "then add SpaceNet tiles."
        )

    # Upload fallback
    uploaded = st.sidebar.file_uploader("Or upload a GeoTIFF", type=["tif", "tiff"])
    if uploaded is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        image, transform, crs = load_image(tmp_path)
        st.sidebar.info("Uploaded image loaded. Mask will need to be provided separately or use GT mode.")


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if image is not None and mask is not None:
    run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run:
        # CRS check
        crs_units = check_crs_units(crs)
        if crs is None:
            warnings_list.append("⚠️ No CRS — all coordinates are in pixel space.")

        # Vectorize
        with st.spinner("Vectorizing mask → polygons..."):
            polygons = mask_to_polygons(
                mask,
                transform=transform,
                crs=crs,
                min_area=min_area,
                simplify_tolerance=simplify_tol,
                use_pixel_coords=(crs is None),
            )

        if not polygons:
            st.error("No polygons found! Check mask threshold or min_area filter.")
        else:
            # Alignment check
            if crs is not None:
                aligned, align_warnings = validate_polygon_raster_alignment(
                    (transform.c, transform.f + transform.e * mask.shape[0],
                     transform.c + transform.a * mask.shape[1], transform.f),
                    [p["geometry"] for p in polygons],
                )
                warnings_list.extend(align_warnings)

            # Solar estimation
            with st.spinner("Computing solar estimates..."):
                per_roof, aggregate = estimate_all_roofs(polygons, config)

            # Visualization
            with st.spinner("Generating visualizations..."):
                fig = side_by_side(image, mask, polygons)

            # === Display ===
            st.subheader("📊 Results")

            # Warnings panel
            if warnings_list:
                for w_msg in warnings_list:
                    st.warning(w_msg)

            # Main figure
            st.pyplot(fig)
            plt.close(fig)

            # Metrics card
            st.subheader("☀️ Solar Estimate Summary")
            sfx = aggregate.get("label_suffix", "")
            cols = st.columns(4)
            cols[0].metric("Roofs Detected", aggregate["num_roofs"])
            cols[1].metric(
                f"Total Roof Area ({aggregate['total_roof_area_unit']})",
                f"{aggregate['total_roof_area']:,.1f}{sfx}",
            )
            cols[2].metric("System Capacity", f"{aggregate['total_system_kw']:,.2f} kW{sfx}")
            cols[3].metric("Monthly Generation", f"{aggregate['total_monthly_kwh']:,.0f} kWh{sfx}")

            st.metric("Estimated Annual Generation", f"{aggregate['total_annual_kwh']:,.0f} kWh{sfx}")

            # Per-roof table
            with st.expander("📋 Per-Roof Details"):
                for r in per_roof:
                    st.text(
                        f"Roof #{r['polygon_id']:>3}: "
                        f"area={r['roof_area']:>8.1f} {r['roof_area_unit']}, "
                        f"usable={r['usable_area']:>8.1f}, "
                        f"kW={r['estimated_system_kw']:>6.2f}, "
                        f"kWh/mo={r['estimated_monthly_kwh']:>8.1f}"
                    )

            # Assumptions
            with st.expander("🔧 Assumptions Used"):
                st.markdown(f"**Label:** {config.assumptions_label}")
                st.markdown(f"- Roof usability: {config.roof_usability_factor}")
                st.markdown(f"- Panel density: {config.panel_power_density_kw_per_m2} kW/m²")
                st.markdown(f"- Performance ratio: {config.performance_ratio}")
                st.markdown(f"- Monthly gen factor: {config.monthly_generation_kwh_per_kw} kWh/kW")

            # Limitations
            with st.expander("⚠️ Confidence / Limitations"):
                for lim in config.limitations:
                    st.markdown(f"- {lim}")

            st.info(config.pre_assessment_disclaimer)

            # === Exports ===
            st.subheader("📥 Downloads")
            col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

            # Overlay PNG
            with col_dl1:
                buf = io.BytesIO()
                fig2 = side_by_side(image, mask, polygons)
                fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig2)
                buf.seek(0)
                st.download_button(
                    "⬇️ Overlay PNG",
                    data=buf,
                    file_name="solar_overlay.png",
                    mime="image/png",
                )

            # GeoJSON
            with col_dl2:
                features = []
                for i, poly in enumerate(polygons):
                    from shapely.geometry import mapping
                    features.append({
                        "type": "Feature",
                        "id": i,
                        "properties": {
                            "id": i,
                            "area_value": round(poly["area_value"], 2),
                            "area_unit": poly["area_unit"],
                        },
                        "geometry": mapping(poly["geometry"]),
                    })
                geojson_str = json.dumps({
                    "type": "FeatureCollection",
                    "features": features,
                }, indent=2)
                st.download_button(
                    "⬇️ GeoJSON",
                    data=geojson_str,
                    file_name="roof_polygons.geojson",
                    mime="application/json",
                )

            # Meta JSON sidecar
            with col_dl3:
                meta_sidecar = json.dumps({
                    "source_raster": None,
                    "crs_epsg": crs.to_epsg() if crs else None,
                    "coordinates": "georeferenced" if crs else "pixel",
                    "num_polygons": len(polygons),
                    "warning": (
                        "GeoJSON consumers may assume WGS84. "
                        "Use this metadata sidecar to interpret coordinates."
                    ),
                }, indent=2)
                st.download_button(
                    "⬇️ Meta JSON",
                    data=meta_sidecar,
                    file_name="footprints.meta.json",
                    mime="application/json",
                )

            # Report
            with col_dl4:
                report = format_report(per_roof, aggregate, config)
                st.download_button(
                    "⬇️ Report (TXT)",
                    data=report,
                    file_name="solar_report.txt",
                    mime="text/plain",
                )

elif image is not None and mask is None:
    st.info("Image loaded but no mask available. Provide a mask/label file or switch to Synthetic mode.")

else:
    st.info("Select a data mode and tile to begin.")
