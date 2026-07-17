"""
driver_cards.py

Calculates driver telemetry KPIs and renders the rounded driver cards.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
import fastf1.core
from utils import (
    get_color, TYRE_COLORS, format_lap_time, format_gap_time
)

def calculate_kpis(tel: pd.DataFrame) -> dict:
    """Calculates telemetry-derived KPIs (speeds, throttle%, brake%)."""
    # Speeds
    top_speed = tel['Speed'].max()
    avg_speed = tel['Speed'].mean()
    
    # Throttle % (average throttle position over the lap)
    throttle_pct = tel['Throttle'].mean()
    
    # Brake % (percentage of lap spent braking)
    brake_pct = (tel['Brake'] > 0).mean() * 100
    
    return {
        'top_speed': top_speed,
        'avg_speed': avg_speed,
        'throttle_pct': throttle_pct,
        'brake_pct': brake_pct
    }

def draw_driver_card(
    ax: plt.Axes,
    lap: fastf1.core.Lap,
    tel: pd.DataFrame,
    driver_abbr: str,
    driver_full_name: str,
    team_name: str,
    color: str,
    gap_seconds: float
) -> None:
    """
    Renders a premium editorial KPI card for a driver in the specified axes.
    """
    # 1. Clear the axes
    ax.axis('off')
    
    card_bg = get_color('card_bg')
    border_color = get_color('border_color')
    text_light = get_color('text_panel_main')
    text_muted = get_color('text_panel_muted')
    
    # 2. Add rounded card background
    card = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        mutation_aspect=1.0,
        linewidth=1.0,
        edgecolor=border_color,
        facecolor=card_bg,
        zorder=1
    )
    ax.add_patch(card)
    
    # 3. Add left accent line in team color
    ax.fill_between(
        [0.02, 0.035], 0.05, 0.95,
        color=color,
        zorder=2
    )
    
    # 4. Calculate KPIs
    kpis = calculate_kpis(tel)
    
    # Get tyre details
    compound = str(lap['Compound']).upper() if not pd.isna(lap['Compound']) else 'UNKNOWN'
    tyre_life = lap['TyreLife'] if not pd.isna(lap['TyreLife']) else 0
    tyre_color = TYRE_COLORS.get(compound, TYRE_COLORS['UNKNOWN'])
    
    # Format lap time and gap
    lap_time_str = format_lap_time(lap['LapTime'])
    gap_str = format_gap_time(gap_seconds)
    gap_color = '#39B54A' if gap_seconds == 0.0 else '#E80020' if gap_seconds > 0 else '#39B54A'
    if gap_seconds == 0.0:
        gap_str = "FASTEST LAP"
        gap_color = '#39B54A'
    
    # Get driver number
    driver_num = str(lap['DriverNumber']) if 'DriverNumber' in lap else ""
    
    # 5. Write Typography/Labels onto Card
    # Header: Driver Full Name and Team
    # Scale font size dynamically for long driver names to fit card bounds
    full_name_upper = driver_full_name.upper()
    fs = 13.0 if len(full_name_upper) < 14 else 11.5 if len(full_name_upper) < 18 else 9.5
    
    ax.text(
        0.06, 0.84, 
        f"{full_name_upper}  #{driver_num}", 
        fontsize=fs, 
        fontweight='bold', 
        color=text_light,
        zorder=3
    )
    ax.text(
        0.06, 0.76, 
        team_name.upper(), 
        fontsize=8.5, 
        fontweight='bold', 
        color=color,
        zorder=3
    )
    
    # Sector 1: Lap Time and Gap
    ax.text(0.06, 0.63, "LAP TIME", fontsize=7.5, color=text_muted, fontweight='semibold', zorder=3)
    ax.text(0.06, 0.50, lap_time_str, fontsize=15, color=text_light, fontweight='bold', zorder=3)
    ax.text(0.06, 0.40, gap_str, fontsize=8.0, color=gap_color, fontweight='bold', zorder=3)
    
    # Sector 2: Top Speed and Avg Speed
    ax.text(0.38, 0.63, "TOP SPEED", fontsize=7.5, color=text_muted, fontweight='semibold', zorder=3)
    ax.text(0.38, 0.50, f"{int(kpis['top_speed'])} km/h", fontsize=11, color=text_light, fontweight='bold', zorder=3)
    ax.text(0.38, 0.40, f"AVG: {int(kpis['avg_speed'])} km/h", fontsize=8.0, color=text_muted, fontweight='semibold', zorder=3)
    
    # Sector 3: Tyre compound and life
    ax.text(0.70, 0.63, "TYRE DETAILS", fontsize=7.5, color=text_muted, fontweight='semibold', zorder=3)
    ax.text(0.70, 0.50, compound, fontsize=12, color=tyre_color, fontweight='bold', zorder=3)
    ax.text(0.70, 0.40, f"{int(tyre_life)} LAPS USED", fontsize=8.0, color=text_muted, fontweight='medium', zorder=3)
    
    # Divider line
    ax.plot([0.06, 0.94], [0.33, 0.33], color=border_color, linewidth=0.8, zorder=3)
    
    # Bottom Row: Telemetry Stats Grid
    # Col 1: Throttle %
    ax.text(0.06, 0.24, "THROTTLE", fontsize=7.5, color=text_muted, fontweight='semibold', zorder=3)
    ax.text(0.06, 0.10, f"{kpis['throttle_pct']:.1f}%", fontsize=11, color=text_light, fontweight='bold', zorder=3)
    
    # Col 2: Brake %
    ax.text(0.38, 0.24, "BRAKE", fontsize=7.5, color=text_muted, fontweight='semibold', zorder=3)
    ax.text(0.38, 0.10, f"{kpis['brake_pct']:.1f}%", fontsize=11, color=text_light, fontweight='bold', zorder=3)
