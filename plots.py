"""
plots.py

Renders individual telemetry panels (Speed, Throttle, Brake, RPM, Delta)
and the corner-by-corner analysis table.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from utils import (
    style_axes, get_color
)

def plot_speed(
    ax: plt.Axes,
    dist: np.ndarray,
    speed_a: np.ndarray,
    speed_b: np.ndarray,
    brake_a: np.ndarray,
    brake_b: np.ndarray,
    color_a: str,
    color_b: str,
    driver_a: str,
    driver_b: str,
    corners: pd.DataFrame
) -> None:
    """Plots the speed trace, highlights braking zones, and shades corners."""
    style_axes(ax, show_grid=True)
    
    grid_color = get_color('grid_color')
    text_muted = get_color('text_panel_muted')
    
    # Plot speed traces
    ax.plot(dist, speed_b, color=color_b, alpha=0.5, linewidth=1.2, label=driver_b)
    ax.plot(dist, speed_a, color=color_a, alpha=0.9, linewidth=1.5, label=driver_a)
    
    # Highlight braking zones
    # Shade under the speed curve where brake is applied
    ax.fill_between(dist, 0, speed_a, where=(brake_a > 0.5), color=color_a, alpha=0.15, zorder=1)
    ax.fill_between(dist, 0, speed_b, where=(brake_b > 0.5), color=color_b, alpha=0.08, zorder=1)
    
    # Shade corners and add markers
    y_min, y_max = speed_a.min() * 0.8, max(speed_a.max(), speed_b.max()) * 1.05
    ax.set_ylim(y_min, y_max)
    
    for _, corner in corners.iterrows():
        c_dist = corner['Distance']
        # Draw a vertical line at the corner apex distance
        ax.axvline(c_dist, color=grid_color, linestyle='--', linewidth=0.6, alpha=0.6)
        
        # Shade the corner zone (approx 50m before and after the apex)
        ax.axvspan(c_dist - 30, c_dist + 30, color='#FFFFFF', alpha=0.02, zorder=0)
        
        # Label the corner number at the top of the plot
        corner_label = f"T{int(corner['Number'])}{corner['Letter']}"
        ax.text(
            c_dist, y_max * 0.96, 
            corner_label, 
            color=text_muted, 
            fontsize=6.5, 
            ha='center', 
            va='top'
        )
        
    ax.set_ylabel("SPEED (km/h)", fontweight='bold')
    ax.legend(loc='lower left', frameon=False, fontsize=7.5)

def plot_throttle(
    ax: plt.Axes,
    dist: np.ndarray,
    throttle_a: np.ndarray,
    throttle_b: np.ndarray,
    color_a: str,
    color_b: str,
    driver_a: str,
    driver_b: str
) -> None:
    """Plots the throttle trace (0-100%)."""
    style_axes(ax, show_grid=True)
    
    ax.plot(dist, throttle_b, color=color_b, alpha=0.5, linewidth=1.0)
    ax.plot(dist, throttle_a, color=color_a, alpha=0.9, linewidth=1.2)
    
    ax.set_ylim(-5, 105)
    ax.set_ylabel("THROTTLE %", fontweight='bold')

def plot_brake(
    ax: plt.Axes,
    dist: np.ndarray,
    brake_a: np.ndarray,
    brake_b: np.ndarray,
    color_a: str,
    color_b: str,
    driver_a: str,
    driver_b: str
) -> None:
    """
    Plots the brake trace. Uses filled horizontal tracks for each driver 
    to prevent overlapping lines and show braking periods clearly.
    """
    style_axes(ax, show_grid=False, remove_y=True)
    
    card_bg = get_color('card_bg')
    border_color = get_color('border_color')
    text_light = get_color('text_light')
    
    # We will plot Driver A in the top track [1.2, 1.8]
    # and Driver B in the bottom track [0.2, 0.8]
    ax.fill_between(dist, 1.2, 1.8, color=card_bg, edgecolor=border_color, linewidth=0.5)
    ax.fill_between(dist, 1.2, 1.8, where=(brake_a > 0.5), color=color_a, alpha=0.9, label=driver_a)
    
    ax.fill_between(dist, 0.2, 0.8, color=card_bg, edgecolor=border_color, linewidth=0.5)
    ax.fill_between(dist, 0.2, 0.8, where=(brake_b > 0.5), color=color_b, alpha=0.7, label=driver_b)
    
    ax.set_ylim(-0.1, 2.1)
    
    # Add text labels on the left of each track
    ax.text(-50, 1.5, f"{driver_a} BRAKE", color=text_light, fontsize=7, fontweight='bold', ha='right', va='center')
    ax.text(-50, 0.5, f"{driver_b} BRAKE", color=text_light, fontsize=7, fontweight='bold', ha='right', va='center')
    
    ax.set_ylabel("BRAKE", fontweight='bold')



def plot_delta(
    ax: plt.Axes,
    dist: np.ndarray,
    delta: np.ndarray,
    color_a: str,
    color_b: str,
    driver_a: str,
    driver_b: str
) -> None:
    """
    Plots the cumulative time delta, coloring the trace dynamically:
    Green/Driver A color when Driver A is gaining, Red/Driver B color when Driver B is gaining.
    """
    style_axes(ax, show_grid=True)
    
    text_muted = get_color('text_muted')
    
    # Calculate derivative of delta to determine who is gaining
    diff = np.diff(delta)
    
    # Create segment lines
    points = np.array([dist, delta]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    colors = []
    for d in diff:
        if d < 0:
            colors.append('#39B54A')  # Green (Driver A gaining)
        else:
            colors.append('#E80020')  # Red (Driver B gaining)
            
    lc = LineCollection(segments, colors=colors, linewidths=1.5, zorder=2)
    ax.add_collection(lc)
    
    # Horizontal line at 0 delta
    ax.axhline(0, color=text_muted, linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Set y limits with some padding
    y_min, y_max = delta.min(), delta.max()
    margin = max(abs(y_min), abs(y_max)) * 0.15
    margin = max(margin, 0.1) # Min margin of 0.1s
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Add text labels explaining color code
    ax.text(
        dist[0] + 50, y_max + margin * 0.5, 
        f"▲ {driver_a} GAINING (GREEN)", 
        color='#39B54A', fontsize=7, fontweight='bold'
    )
    ax.text(
        dist[0] + 50, y_min - margin * 0.5, 
        f"▼ {driver_b} GAINING (RED)", 
        color='#E80020', fontsize=7, fontweight='bold',
        va='bottom'
    )
    
    ax.set_ylabel("DELTA TIME (s)", fontweight='bold')
    ax.set_xlabel("DISTANCE (m)", fontweight='bold')

def draw_corner_table(
    ax: plt.Axes,
    dist: np.ndarray,
    speed_a: np.ndarray,
    speed_b: np.ndarray,
    corners: pd.DataFrame,
    driver_a: str,
    driver_b: str
) -> None:
    """Renders a clean, highly formatted corner speed comparison table as text."""
    ax.axis('off')
    
    card_bg = get_color('card_bg')
    border_color = get_color('border_color')
    text_light = get_color('text_panel_main')
    text_muted = get_color('text_panel_muted')
    row_divider = get_color('row_divider')
    
    # Table background box
    from matplotlib.patches import FancyBboxPatch
    bg_box = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=border_color,
        facecolor=card_bg,
        zorder=1
    )
    ax.add_patch(bg_box)
    
    # Header titles
    ax.text(0.05, 0.92, "CORNER BY CORNER COMPARISON", fontsize=9, fontweight='bold', color=text_light, zorder=2)
    
    # Column X coordinates
    col_x = {
        'corner': 0.05,
        'speed_a': 0.22,
        'speed_b': 0.44,
        'delta': 0.66,
        'fastest': 0.85
    }
    
    # Column Headers
    ax.text(col_x['corner'], 0.83, "CORNER", color=text_muted, fontsize=7.5, fontweight='bold', zorder=2)
    ax.text(col_x['speed_a'], 0.83, f"{driver_a} (km/h)", color=text_muted, fontsize=7.5, fontweight='bold', zorder=2)
    ax.text(col_x['speed_b'], 0.83, f"{driver_b} (km/h)", color=text_muted, fontsize=7.5, fontweight='bold', zorder=2)
    ax.text(col_x['delta'], 0.83, "DELTA", color=text_muted, fontsize=7.5, fontweight='bold', zorder=2)
    ax.text(col_x['fastest'], 0.83, "FASTER", color=text_muted, fontsize=7.5, fontweight='bold', zorder=2)
    
    # Divider line under headers
    ax.plot([0.05, 0.95], [0.80, 0.80], color=border_color, linewidth=0.8, zorder=2)
    
    # Populate rows
    if corners.empty:
        ax.text(0.5, 0.5, "No corner data available", color=text_muted, fontsize=10, ha='center', va='center')
        return
        
    num_corners = len(corners)
    row_start_y = 0.74
    row_step_y = 0.70 / num_corners
    row_step_y = min(row_step_y, 0.045) # limit spacing
    
    for i, corner in corners.iterrows():
        c_num = f"T{int(corner['Number'])}{corner['Letter']}"
        c_dist = corner['Distance']
        
        # Apex speed search window: +/- 60m around corner distance
        window = (dist >= c_dist - 60) & (dist <= c_dist + 60)
        
        if np.any(window):
            v_a = speed_a[window].min()
            v_b = speed_b[window].min()
        else:
            # Fallback to nearest point if window is empty
            closest_idx = np.abs(dist - c_dist).argmin()
            v_a = speed_a[closest_idx]
            v_b = speed_b[closest_idx]
            
        speed_diff = v_a - v_b
        
        # Determine faster driver and format delta
        if abs(speed_diff) < 0.5:
            delta_str = "—"
            fastest_driver = "TIE"
            fastest_color = text_muted
        elif speed_diff > 0:
            delta_str = f"+{speed_diff:.1f}"
            fastest_driver = driver_a
            fastest_color = '#39B54A'
        else:
            delta_str = f"{speed_diff:.1f}" # Negative sign included
            fastest_driver = driver_b
            fastest_color = '#E80020'
            
        y = row_start_y - i * row_step_y
        
        # Draw table text
        ax.text(col_x['corner'], y, c_num, color=text_light, fontsize=7.5, fontweight='medium', zorder=2)
        ax.text(col_x['speed_a'], y, f"{int(v_a)}", color=text_light, fontsize=7.5, zorder=2)
        ax.text(col_x['speed_b'], y, f"{int(v_b)}", color=text_light, fontsize=7.5, zorder=2)
        ax.text(col_x['delta'], y, delta_str, color=fastest_color, fontsize=7.5, fontweight='semibold', zorder=2)
        ax.text(col_x['fastest'], y, fastest_driver, color=fastest_color, fontsize=7.5, fontweight='bold', zorder=2)
        
        # Draw a very faint divider line between rows
        if i < num_corners - 1:
            ax.plot([0.05, 0.95], [y - row_step_y * 0.35, y - row_step_y * 0.35], color=row_divider, linewidth=0.4, zorder=2)
