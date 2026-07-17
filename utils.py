"""
utils.py

Utility functions for styling, colors, and formatting in the F1 telemetry comparison dashboard.
Supports multiple visual themes: Carbon (default), Slate, and Light (Hybrid style with white panels on dark canvas).
"""

from typing import Union
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fastf1.plotting

# Standard tyre compound color mapping
TYRE_COLORS = {
    'SOFT': '#FF3333',         # Soft (Red)
    'MEDIUM': '#FFD100',       # Medium (Yellow)
    'HARD': '#FFFFFF',         # Hard (White)
    'INTERMEDIATE': '#39B54A', # Inter (Green)
    'WET': '#00AEEF',          # Wet (Blue)
    'UNKNOWN': '#8A8A8A'
}

# Editorial theme color palettes
THEMES = {
    'carbon': {
        'bg_dark': '#121212',
        'axes_bg': '#121212',
        'card_bg': '#1A1A1A',
        'grid_color': '#2A2A2A',
        'text_light': '#F5F5F5',
        'text_muted_outer': '#8A8A8A',
        'text_panel_main': '#F5F5F5',
        'text_panel_muted': '#8A8A8A',
        'border_color': '#333333',
        'row_divider': '#222222'
    },
    'slate': {
        'bg_dark': '#0E1117',
        'axes_bg': '#0E1117',
        'card_bg': '#1A1F2C',
        'grid_color': '#282F42',
        'text_light': '#E2E8F0',
        'text_muted_outer': '#94A3B8',
        'text_panel_main': '#E2E8F0',
        'text_panel_muted': '#94A3B8',
        'border_color': '#323C52',
        'row_divider': '#202635'
    },
    'light': {
        'bg_dark': '#121212',           # Keep the main dashboard background dark charcoal
        'axes_bg': '#F8F9FA',           # Soft white background for plot axes
        'card_bg': '#FFFFFF',           # White background for driver cards
        'grid_color': '#E2E8F0',         # Light grey grid inside white plots
        'text_light': '#F5F5F5',         # White text for titles on the dark background
        'text_muted_outer': '#8A8A8A',   # Muted white/grey for ticks/axis labels on dark background
        'text_panel_main': '#1A202C',   # High-contrast dark grey/black inside white cards/tables
        'text_panel_muted': '#718096',   # Dark grey for labels inside white cards/tables
        'border_color': '#CBD5E0',       # Soft grey border for white panels
        'row_divider': '#EDF2F7'         # Light row divider inside white table
    },
    'retro': {
        'bg_dark': '#F5EFE3',           # Warm vintage cream background
        'axes_bg': '#FFFFFF',           # White background for plot axes
        'card_bg': '#FFFFFF',           # White background for driver cards
        'grid_color': '#EADFCD',         # Soft warm brown grid inside white plots
        'text_light': '#2C1D11',         # Dark brown text for titles on the light background
        'text_muted_outer': '#705E52',   # Muted brown for ticks/axis labels on light background
        'text_panel_main': '#2C1D11',   # Dark brown inside white cards/tables
        'text_panel_muted': '#705E52',   # Muted brown inside white cards/tables
        'border_color': '#C8B195',       # Sand/brown borders
        'row_divider': '#EADFCD'         # Soft warm divider inside white table
    }
}

class ThemeState:
    """Holds global theme state for rendering."""
    current = THEMES['carbon']

def set_active_theme(name: str) -> None:
    """Sets the active dashboard theme."""
    name_clean = name.strip().lower()
    if name_clean in THEMES:
        ThemeState.current = THEMES[name_clean]
    else:
        ThemeState.current = THEMES['carbon']

def get_color(color_name: str) -> str:
    """Gets a color from the currently active theme."""
    return ThemeState.current.get(color_name, '#FFFFFF')

def setup_matplotlib_params() -> None:
    """Configures global Matplotlib styles based on the active theme colors."""
    fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')
    
    bg_dark = get_color('bg_dark')
    axes_bg = get_color('axes_bg')
    text_light = get_color('text_light')
    text_muted_outer = get_color('text_muted_outer')
    grid_color = get_color('grid_color')
    
    plt.rcParams.update({
        'figure.facecolor': bg_dark,
        'axes.facecolor': axes_bg,
        'savefig.facecolor': bg_dark,
        'text.color': text_light,
        'axes.labelcolor': text_muted_outer,
        'xtick.color': text_muted_outer,
        'ytick.color': text_muted_outer,
        'grid.color': grid_color,
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.titlesize': 12
    })

def style_axes(ax: plt.Axes, show_grid: bool = True, remove_y: bool = False) -> None:
    """Applies a minimal editorial style to a Matplotlib axes object."""
    border_color = get_color('border_color')
    grid_color = get_color('grid_color')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(border_color)
    ax.spines['bottom'].set_color(border_color)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    ax.tick_params(axis='both', which='both', length=3, width=0.8, color=border_color)
    
    if show_grid:
        ax.grid(True, which='both', linestyle=':', color=grid_color, linewidth=0.5)
    else:
        ax.grid(False)
        
    if remove_y:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)

def format_lap_time(time_val: Union[pd.Timedelta, datetime.timedelta, float]) -> str:
    """Formats a lap time into MM:SS.mmm format."""
    if pd.isna(time_val):
        return "N/A"
        
    if isinstance(time_val, (pd.Timedelta, datetime.timedelta)):
        total_seconds = time_val.total_seconds()
    else:
        total_seconds = float(time_val)
        
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    
    if minutes > 0:
        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{seconds}.{milliseconds:03d}"

def format_gap_time(gap_val: float) -> str:
    """Formats a time gap into +S.mmm or -S.mmm format."""
    if pd.isna(gap_val) or gap_val == 0.0:
        return "—"
    prefix = "+" if gap_val > 0 else "-"
    return f"{prefix}{abs(gap_val):.3f}s"

def get_driver_color_safe(driver: str, session) -> str:
    """Safely retrieves a driver's team color from FastF1, falling back to a neutral gray."""
    try:
        color = fastf1.plotting.get_driver_color(driver, session=session)
        if color:
            return color
    except Exception:
        pass
    
    driver_defaults = {
        'NOR': '#ff8000', # McLaren
        'PIA': '#ff8000',
        'VER': '#0600ef', # Red Bull
        'HAD': '#0600ef',
        'HAM': '#e80020', # Ferrari
        'LEC': '#e80020',
        'RUS': '#27f4d2', # Mercedes
        'ANT': '#27f4d2',
        'ALO': '#229971', # Aston Martin
        'STR': '#229971',
        'ALB': '#37bedd', # Williams
        'SAI': '#37bedd',
        'GAS': '#0078c1', # Alpine
        'COL': '#0078c1',
        'HUL': '#52e252', # Audi
        'BOR': '#52e252',
        'PER': '#c5a059', # Cadillac
        'BOT': '#c5a059',
        'LAW': '#03002f', # Racing Bulls
        'LIN': '#03002f',
        'OCO': '#ffffff', # Haas
        'BEA': '#ffffff'
    }
    return driver_defaults.get(driver, '#9E9E9E')
