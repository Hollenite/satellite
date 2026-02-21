"""
Visualization helpers for the solar pre-assessment pipeline.

Produces clean overlay images, side-by-side comparisons, and annotated
polygon views suitable for demo screenshots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit / scripts

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection
from shapely.geometry import mapping

from src.data import prepare_display_rgb


# ---------------------------------------------------------------------------
# Basic displays
# ---------------------------------------------------------------------------

def show_rgb(
    image: np.ndarray,
    title: str = "Satellite Tile",
    ax: Optional[plt.Axes] = None,
    rgb_bands: Tuple[int, int, int] = (0, 1, 2),
) -> plt.Axes:
    """Display a raster as RGB. Handles (bands, H, W) and (H, W, bands)."""
    rgb = prepare_display_rgb(image, rgb_bands=rgb_bands)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return ax


def show_mask(
    mask: np.ndarray,
    title: str = "Building Mask",
    ax: Optional[plt.Axes] = None,
    cmap: str = "gray",
) -> plt.Axes:
    """Display a binary mask."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Polygon overlay
# ---------------------------------------------------------------------------

def overlay_polygons(
    image: np.ndarray,
    polygons: List[Dict],
    alpha: float = 0.35,
    edge_color: str = "cyan",
    face_color: str = "yellow",
    linewidth: float = 1.2,
    ax: Optional[plt.Axes] = None,
    rgb_bands: Tuple[int, int, int] = (0, 1, 2),
    title: str = "Roof Polygons Overlay",
) -> plt.Axes:
    """
    Draw polygons on top of the satellite image.

    Args:
        image: raster array (bands, H, W) or (H, W, 3)
        polygons: list of dicts with 'geometry' key (Shapely)
        alpha: fill transparency
        edge_color: polygon edge colour
        face_color: polygon fill colour
    """
    rgb = prepare_display_rgb(image, rgb_bands=rgb_bands)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb)

    for poly_dict in polygons:
        geom = poly_dict["geometry"]
        _draw_polygon(ax, geom, face_color, edge_color, alpha, linewidth)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return ax


def _draw_polygon(ax, geom, face_color, edge_color, alpha, linewidth):
    """Draw a single Shapely geometry on a matplotlib Axes."""
    if geom.geom_type == "Polygon":
        coords = np.array(geom.exterior.coords)
        patch = mpatches.Polygon(
            coords, closed=True,
            facecolor=face_color, edgecolor=edge_color,
            alpha=alpha, linewidth=linewidth,
        )
        ax.add_patch(patch)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            _draw_polygon(ax, part, face_color, edge_color, alpha, linewidth)


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def side_by_side(
    image: np.ndarray,
    mask: np.ndarray,
    polygons: List[Dict],
    save_path: Optional[str | Path] = None,
    rgb_bands: Tuple[int, int, int] = (0, 1, 2),
    figsize: Tuple[int, int] = (18, 6),
) -> plt.Figure:
    """
    Create a 3-panel figure: raw tile | binary mask | polygon overlay.

    Saves to *save_path* if provided.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    show_rgb(image, title="Satellite Tile", ax=axes[0], rgb_bands=rgb_bands)
    show_mask(mask, title="Building Mask", ax=axes[1])
    overlay_polygons(
        image, polygons,
        ax=axes[2], rgb_bands=rgb_bands,
        title="Roof Polygons Overlay",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved overlay figure → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def annotate_polygons(
    ax: plt.Axes,
    polygons: List[Dict],
    show_id: bool = True,
    show_area: bool = True,
    fontsize: int = 7,
    color: str = "white",
):
    """
    Add polygon ID and area labels at each polygon centroid.
    """
    for i, poly_dict in enumerate(polygons):
        geom = poly_dict["geometry"]
        centroid = geom.centroid
        label_parts = []
        if show_id:
            label_parts.append(f"#{i}")
        if show_area:
            area = poly_dict.get("area_value", 0)
            unit = poly_dict.get("area_unit", "")
            label_parts.append(f"{area:.0f} {unit}")
        label = "\n".join(label_parts)
        ax.text(
            centroid.x, centroid.y, label,
            fontsize=fontsize, color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )
