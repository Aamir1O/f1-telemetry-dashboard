"""
track_map.py

Renders the circuit track map, color-coded by which driver is faster at each
point on track, with corner annotations and the start/finish line.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import fastf1.core
from typing import Tuple
from utils import style_axes, get_color

def rotate_coords(x: np.ndarray, y: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotates X and Y coordinates by a given angle in degrees."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot

def plot_track_map(
    ax: plt.Axes, 
    session: fastf1.core.Session, 
    tel_a: dict, 
    tel_b: dict, 
    driver_a: str, 
    driver_b: str, 
    color_a: str, 
    color_b: str
) -> None:
    """
    Plots a single track outline colored by which driver is faster at each
    point on track (speed delta), plus start/finish line and corner markers.
    """
    # 1. Apply layout styles
    style_axes(ax, show_grid=False)
    ax.axis('equal')
    
    # Hide the axes spines and ticks completely for a clean look
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Add rounded card background for the track map (dynamic theme support)
    card_bg = get_color('card_bg')
    border_color = get_color('border_color')
    card = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        mutation_aspect=1.0,
        linewidth=1.0,
        edgecolor=border_color,
        facecolor=card_bg,
        transform=ax.transAxes,
        zorder=1
    )
    ax.add_patch(card)
    
    # 2. Extract X and Y telemetry coordinates.
    # Both drivers were interpolated onto the same common distance grid, so
    # their X/Y paths (and therefore Speed arrays) line up index-for-index.
    # We use driver A's geometry as the single reference line to draw/color.
    x_a, y_a = tel_a['X'], tel_a['Y']
    speed_a, speed_b = tel_a['Speed'], tel_b['Speed']
    
    # 3. Retrieve circuit info and rotation
    circuit_info = session.get_circuit_info()
    rotation_angle = 0.0
    if circuit_info is not None and hasattr(circuit_info, 'rotation'):
        rotation_angle = circuit_info.rotation
        
    # Rotate telemetry coordinates
    x_rot, y_rot = rotate_coords(x_a, y_a, rotation_angle)

    track_span_x = x_rot.max() - x_rot.min()
    track_span_y = y_rot.max() - y_rot.min()
    span = max(track_span_x, track_span_y)

    # 4. Build a speed-delta-colored line (positive = A faster, negative = B faster)
    speed_diff = speed_a - speed_b
    max_abs_diff = max(np.abs(speed_diff).max(), 1e-6)  # avoid a flat 0..0 norm

    gap_cmap = LinearSegmentedColormap.from_list(
        'gap_cmap', [color_b, get_color('text_panel_muted'), color_a]
    )
    norm = TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0.0, vmax=max_abs_diff)

    points = np.array([x_rot, y_rot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Color each segment by the average delta of its two endpoints
    segment_values = (speed_diff[:-1] + speed_diff[1:]) / 2.0

    lc = LineCollection(segments, cmap=gap_cmap, norm=norm, zorder=3)
    lc.set_array(segment_values)
    lc.set_linewidth(3.0)
    ax.add_collection(lc)
    
    # 5. Plot Start/Finish Line
    # Calculate perpendicular line at start (index 0)
    dx = x_rot[1] - x_rot[0]
    dy = y_rot[1] - y_rot[0]
    norm_vec = np.hypot(dx, dy)
    
    # We want a perpendicular vector
    px = -dy / norm_vec
    py = dx / norm_vec
    
    # Length of start/finish line (scale based on track size)
    line_len = span * 0.04 # 4% of track span
    
    sf_x1 = x_rot[0] - px * line_len
    sf_y1 = y_rot[0] - py * line_len
    sf_x2 = x_rot[0] + px * line_len
    sf_y2 = y_rot[0] + py * line_len
    
    ax.plot([sf_x1, sf_x2], [sf_y1, sf_y2], color=get_color('text_panel_main'), linewidth=2.0, zorder=5)
    
    # Add Start/Finish label
    # Offset label slightly in the direction of the perpendicular
    label_offset = line_len * 1.5
    ax.text(
        x_rot[0] - px * label_offset, 
        y_rot[0] - py * label_offset, 
        "S/F", 
        color=get_color('text_panel_main'), 
        fontsize=7, 
        fontweight='bold',
        ha='center', 
        va='center',
        zorder=5,
        bbox=dict(facecolor=get_color('card_bg'), edgecolor=get_color('border_color'), boxstyle='round,pad=0.2', alpha=0.8)
    )
    
    # 6. Plot Corners
    if circuit_info is not None and circuit_info.corners is not None and not circuit_info.corners.empty:
        corners = circuit_info.corners
        # Determine appropriate text offset based on track scale
        offset_distance = span * 0.08  # 8% of track span
        
        for _, corner in corners.iterrows():
            # Get original corner coords
            cx, cy = corner['X'], corner['Y']
            # Rotate corner coordinates
            cx_rot, cy_rot = rotate_coords(np.array([cx]), np.array([cy]), rotation_angle)
            cx_rot, cy_rot = cx_rot[0], cy_rot[0]
            
            # Corner angle is relative to original orientation
            # So the offset angle in rotated frame is corner angle + rotation angle
            offset_angle_deg = corner['Angle'] + rotation_angle
            offset_angle_rad = np.radians(offset_angle_deg)
            
            # Calculate offset coordinates
            ox = offset_distance * np.cos(offset_angle_rad)
            oy = offset_distance * np.sin(offset_angle_rad)
            
            tx = cx_rot + ox
            ty = cy_rot + oy
            
            # Draw a tiny dot at the corner peak
            ax.scatter(cx_rot, cy_rot, color=get_color('text_panel_muted'), s=6, zorder=4)
            
            # Write the corner number label
            corner_str = f"{int(corner['Number'])}{corner['Letter']}"
            ax.text(
                tx, ty, 
                corner_str, 
                color=get_color('text_panel_main'), 
                fontsize=7, 
                ha='center', 
                va='center',
                zorder=5,
                bbox=dict(
                    facecolor=get_color('card_bg'), 
                    edgecolor=get_color('border_color'), 
                    boxstyle='circle,pad=0.2', 
                    alpha=0.8
                )
            )

    # Make sure the LineCollection is actually included in the axes' data limits,
    # since add_collection() doesn't autoscale the view the way ax.plot() does.
    ax.set_xlim(x_rot.min() - span * 0.05, x_rot.max() + span * 0.05)
    ax.set_ylim(y_rot.min() - span * 0.05, y_rot.max() + span * 0.05)

    # 7. Custom legend: two color swatches for "driver faster here", not line labels
    legend_handles = [
        Line2D([0], [0], color=color_a, linewidth=3, label=f"{driver_a} faster"),
        Line2D([0], [0], color=color_b, linewidth=3, label=f"{driver_b} faster"),
    ]
    leg = ax.legend(
        handles=legend_handles, loc='lower center', frameon=True,
        facecolor=get_color('card_bg'), edgecolor=get_color('border_color'), fontsize=7
    )
    for text in leg.get_texts():
        text.set_color(get_color('text_panel_main'))