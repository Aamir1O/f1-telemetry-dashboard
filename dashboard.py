"""
dashboard.py

The main orchestration script for the F1 telemetry comparison dashboard.
Loads data, performs telemetry interpolation, triggers plotting, and saves both desktop and mobile PNGs.
Supports both non-interactive CLI arguments and interactive terminal selections.
Supports custom theme styling (Carbon, Slate, Light) and driver full name lookup.
"""

import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fastf1.core

from utils import (
    get_driver_color_safe, format_lap_time, setup_matplotlib_params,
    set_active_theme, get_color
)
from load_data import load_session_data, get_driver_fastest_lap, get_circuit_corners
from telemetry import interpolate_telemetry, calculate_delta_time
from layout import create_dashboard_layout, create_mobile_dashboard_layout
from track_map import plot_track_map
from driver_cards import draw_driver_card
from plots import (
    plot_speed, plot_throttle, plot_brake, plot_delta, draw_corner_table
)

def render_dashboard_elements(
    fig: plt.Figure,
    axes: dict,
    session: fastf1.core.Session,
    lap_a: fastf1.core.Lap,
    lap_b: fastf1.core.Lap,
    tel_a_orig: pd.DataFrame,
    tel_b_orig: pd.DataFrame,
    tel_a_interp: dict,
    tel_b_interp: dict,
    driver_a: str,
    driver_b: str,
    full_name_a: str,
    full_name_b: str,
    team_a: str,
    team_b: str,
    color_a: str,
    color_b: str,
    common_distance: np.ndarray,
    delta_time: np.ndarray,
    corners: pd.DataFrame,
    gap_a_sec: float,
    gap_b_sec: float,
    is_mobile: bool = False
) -> None:
    """Renders the standard set of cards and telemetry plots onto a given GridSpec axes mapping."""
    # 1. Render top row elements
    # KPI cards - Now displaying full names
    draw_driver_card(axes['card_a'], lap_a, tel_a_orig, driver_a, full_name_a, team_a, color_a, gap_a_sec)
    draw_driver_card(axes['card_b'], lap_b, tel_b_orig, driver_b, full_name_b, team_b, color_b, gap_b_sec)
    
    # Track map
    plot_track_map(axes['track'], session, tel_a_interp, tel_b_interp, driver_a, driver_b, color_a, color_b)
    
    # 2. Render telemetry traces
    plot_speed(
        axes['speed'], common_distance, 
        tel_a_interp['Speed'], tel_b_interp['Speed'],
        tel_a_interp['Brake'], tel_b_interp['Brake'],
        color_a, color_b, driver_a, driver_b, corners
    )
    
    plot_throttle(
        axes['throttle'], common_distance, 
        tel_a_interp['Throttle'], tel_b_interp['Throttle'], 
        color_a, color_b, driver_a, driver_b
    )
    
    plot_brake(
        axes['brake'], common_distance, 
        tel_a_interp['Brake'], tel_b_interp['Brake'], 
        color_a, color_b, driver_a, driver_b
    )
    
    # Delta Time
    plot_delta(
        axes['delta'], common_distance, delta_time, 
        color_a, color_b, driver_a, driver_b
    )
    
    # 3. Render corner speed comparison table
    draw_corner_table(
        axes['table'], common_distance, 
        tel_a_interp['Speed'], tel_b_interp['Speed'], 
        corners, driver_a, driver_b
    )
    
    # 4. Configure headers & footers
    event_name = session.event['EventName'].upper()
    session_title = session.name.upper()
    
    text_light = get_color('text_light')
    text_muted = get_color('text_muted_outer')
    
    if is_mobile:
        fig.text(0.02, 0.985, "Performance Intelligence Report (MOBILE)", fontsize=13, fontweight='black', color='#39B54A', va='top')
        
        event_str = f"2026 {event_name}\n{session_title}  •  SILVERSTONE"
        fig.text(0.98, 0.985, event_str, fontsize=8.5, fontweight='semibold', color=text_muted, ha='right', va='top')
        
        # Mobile Footer (Themed badge box)
        fig.text(
            0.97, 0.008, 
            "by Sk Aamir Ahmed", 
            fontsize=7.5, 
            color=get_color('text_panel_main'), 
            ha='right', 
            va='bottom', 
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5,rounding_size=0.2', 
                facecolor=get_color('card_bg'), 
                edgecolor=get_color('border_color'), 
                linewidth=0.8
            )
        )
    else:
        fig.text(0.02, 0.98, "Performance Intelligence Report", fontsize=13, fontweight='black', color='#39B54A', va='top')
        
        circuit_name = session.event.get('Location', 'SILVERSTONE').upper()
        event_str = f"2026 {event_name}  •  {session_title}  •  {circuit_name}"
        fig.text(0.98, 0.98, event_str, fontsize=9.5, fontweight='semibold', color=text_muted, ha='right', va='top')
        
        # Desktop Footer (Themed badge box)
        fig.text(
            0.97, 0.01, 
            "by Sk Aamir Ahmed", 
            fontsize=8.5, 
            color=get_color('text_panel_main'), 
            ha='right', 
            va='bottom', 
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5,rounding_size=0.2', 
                facecolor=get_color('card_bg'), 
                edgecolor=get_color('border_color'), 
                linewidth=0.8
            )
        )

def build_dashboard(
    year: int,
    round_idx: int,
    session_name: str,
    driver_a: str,
    driver_b: str,
    theme: str = 'carbon',
    output_filename: str = None
) -> str:
    """
    Loads telemetry, performs interpolation, plots both desktop and mobile layouts, and saves them.
    """
    # Set visual theme before rendering
    set_active_theme(theme)
    
    print(f"Starting dashboard pipeline for {year} Round {round_idx} ({session_name}): {driver_a} vs {driver_b} [Theme: {theme}]...")
    
    # 1. Load data from FastF1
    session = load_session_data(year, round_idx, session_name)
    
    # Get lap details
    lap_a, tel_a_orig = get_driver_fastest_lap(session, driver_a)
    lap_b, tel_b_orig = get_driver_fastest_lap(session, driver_b)
    
    # Extract driver full names from session records
    try:
        info_a = session.get_driver(driver_a)
        full_name_a = info_a['FullName'] if (info_a is not None and 'FullName' in info_a and info_a['FullName']) else driver_a
    except Exception:
        full_name_a = driver_a
        
    try:
        info_b = session.get_driver(driver_b)
        full_name_b = info_b['FullName'] if (info_b is not None and 'FullName' in info_b and info_b['FullName']) else driver_b
    except Exception:
        full_name_b = driver_b
    
    # Extract team names
    team_a = lap_a['Team']
    team_b = lap_b['Team']
    
    # Get driver team colors
    color_a = get_driver_color_safe(driver_a, session)
    color_b = get_driver_color_safe(driver_b, session)
    
    # 2. Interpolate telemetry onto a common distance axis
    common_distance, tel_a_interp, tel_b_interp = interpolate_telemetry(tel_a_orig, tel_b_orig)
    
    # 3. Calculate time delta
    delta_time = calculate_delta_time(tel_a_interp['Time'], tel_b_interp['Time'])
    
    # 4. Get circuit corners
    corners = get_circuit_corners(session)
    
    # Calculate lap times and gaps
    lap_time_a = lap_a['LapTime'].total_seconds()
    lap_time_b = lap_b['LapTime'].total_seconds()
    gap_a_sec = 0.0 if lap_time_a <= lap_time_b else (lap_time_a - lap_time_b)
    gap_b_sec = 0.0 if lap_time_b < lap_time_a else (lap_time_b - lap_time_a)
    
    if not output_filename:
        output_filename = f"{year}_{round_idx}_{driver_a}_{driver_b}_dashboard.png"
    
    # 5. Build Desktop Layout (Landscape 18x10)
    print("Generating Desktop Dashboard layout...")
    fig_desktop, axes_desktop = create_dashboard_layout()
    render_dashboard_elements(
        fig_desktop, axes_desktop, session, lap_a, lap_b, tel_a_orig, tel_b_orig,
        tel_a_interp, tel_b_interp, driver_a, driver_b, full_name_a, full_name_b, team_a, team_b,
        color_a, color_b, common_distance, delta_time, corners, gap_a_sec, gap_b_sec,
        is_mobile=False
    )
    
    print(f"Saving Desktop Dashboard image to: {output_filename}...")
    fig_desktop.savefig(
        output_filename, 
        dpi=300, 
        facecolor=fig_desktop.get_facecolor(), 
        edgecolor='none', 
        bbox_inches='tight'
    )
    plt.close(fig_desktop)
    
    # 6. Build Mobile Layout (Portrait 10x18)
    print("Generating Mobile Dashboard layout...")
    fig_mobile, axes_mobile = create_mobile_dashboard_layout()
    render_dashboard_elements(
        fig_mobile, axes_mobile, session, lap_a, lap_b, tel_a_orig, tel_b_orig,
        tel_a_interp, tel_b_interp, driver_a, driver_b, full_name_a, full_name_b, team_a, team_b,
        color_a, color_b, common_distance, delta_time, corners, gap_a_sec, gap_b_sec,
        is_mobile=True
    )
    
    mobile_filename = output_filename.replace("_dashboard.png", "_mobile_dashboard.png")
    print(f"Saving Mobile Dashboard image to: {mobile_filename}...")
    fig_mobile.savefig(
        mobile_filename, 
        dpi=300, 
        facecolor=fig_mobile.get_facecolor(), 
        edgecolor='none', 
        bbox_inches='tight'
    )
    plt.close(fig_mobile)
    
    print("Dashboard pipeline completed successfully.")
    return output_filename

def map_session_name(input_str: str) -> str:
    """Maps common session abbreviations to standard FastF1 session names."""
    s = input_str.strip().lower()
    if s in ('fp1', 'practice 1', 'p1', '1'):
        return 'Practice 1'
    if s in ('fp2', 'practice 2', 'p2', '2'):
        return 'Practice 2'
    if s in ('fp3', 'practice 3', 'p3', '3'):
        return 'Practice 3'
    if s in ('q', 'qualifying', 'quali'):
        return 'Qualifying'
    if s in ('s', 'sprint', 'spr'):
        return 'Sprint'
    if s in ('sq', 'sprint qualifying', 'sprint quali', 'sprint shootout', 'ss'):
        return 'Sprint Qualifying'
    if s in ('r', 'race', 'gp'):
        return 'Race'
    return input_str

def interactive_selection() -> tuple:
    """Interactively prompts the user for year, round, session, theme, and drivers."""
    print("\n" + "="*55)
    print("          F1 TELEMETRY COMPARISON SELECTOR          ")
    print("="*55)
    
    # 1. Year input
    year_input = input("Enter Year [2026]: ").strip()
    year = int(year_input) if year_input else 2026
    
    # 2. Round input
    round_input = input("Enter Round Number [9 for British GP]: ").strip()
    round_idx = int(round_input) if round_input else 9
    
    # 3. Session input
    print("\nSession options:")
    print("  - FP1 : Practice 1")
    print("  - FP2 : Practice 2")
    print("  - FP3 : Practice 3")
    print("  - Q   : Qualifying")
    print("  - S   : Sprint")
    print("  - SQ  : Sprint Qualifying")
    print("  - R   : Race")
    sess_input = input("\nEnter Session Name or Abbreviation [FP1]: ").strip()
    session_name = map_session_name(sess_input) if sess_input else 'Practice 1'
    
    # 4. Theme selection
    print("\nTheme options:")
    print("  - carbon : Stealth Dark (Matte Charcoal)")
    print("  - slate  : Cool Blue-Grey")
    print("  - light  : Hybrid White Panels on Dark Canvas")
    print("  - retro  : Vintage Cream (Retro Newsprint)")
    theme_input = input("\nEnter Theme [carbon]: ").strip().lower()
    theme = theme_input if theme_input in ('carbon', 'slate', 'light', 'retro') else 'carbon'
    
    # Enable cache and load basic session info to display drivers
    print("\nLoading session details to retrieve driver list...")
    try:
        session = load_session_data(year, round_idx, session_name)
    except Exception as e:
        print(f"Error loading session: {e}")
        sys.exit(1)
        
    print("\nAvailable drivers in this session:")
    driver_list = []
    # Print formatted driver details (abbreviations for user choice)
    print(f"{'Code':<6} | {'No.':<4} | {'Driver Name':<22} | {'Team'}")
    print("-" * 60)
    for d in session.drivers:
        try:
            info = session.get_driver(d)
            abbr = info['Abbreviation']
            num = info['DriverNumber']
            name = info['FullName']
            team = info['TeamName']
            driver_list.append(abbr)
            print(f"{abbr:<6} | #{num:<3} | {name:<22} | {team}")
        except Exception:
            pass
            
    print("-" * 60)
    
    # Flush stdin to prevent skipping if user hit Enter during data load
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except Exception:
        pass
        
    # 5. Driver A
    while True:
        driver_a = input("\nEnter Driver A abbreviation (e.g. HAM): ").strip().upper()
        if not driver_a:
            driver_a = 'HAM'
            print("Defaulting to HAM")
            break
        if driver_a in driver_list:
            break
        print(f"Driver '{driver_a}' not found in the list. Please try again.")
        
    # 6. Driver B
    while True:
        driver_b = input("Enter Driver B abbreviation (e.g. RUS): ").strip().upper()
        if not driver_b:
            driver_b = 'RUS'
            print("Defaulting to RUS")
            break
        if driver_b == driver_a:
            print("Driver B cannot be the same as Driver A. Please choose a different driver.")
            continue
        if driver_b in driver_list:
            break
        print(f"Driver '{driver_b}' not found in the list. Please try again.")
        
    return year, round_idx, session_name, driver_a, driver_b, theme

if __name__ == '__main__':
    # If run without arguments, use interactive mode
    if len(sys.argv) == 1:
        year, round_idx, session_name, driver_a, driver_b, theme = interactive_selection()
        build_dashboard(
            year=year,
            round_idx=round_idx,
            session_name=session_name,
            driver_a=driver_a,
            driver_b=driver_b,
            theme=theme
        )
    else:
        # Parse CLI arguments
        parser = argparse.ArgumentParser(description="Generate F1 Performance Intelligence Report (Desktop and Mobile)")
        parser.add_argument('--year', type=int, default=2026, help="Session Year")
        parser.add_argument('--round', type=int, default=9, help="Event Round index (British GP is 9)")
        parser.add_argument('--session', type=str, default='Practice 1', help="Session Name (Practice 1, Qualifying, Race)")
        parser.add_argument('--driver-a', type=str, default='HAM', help="Abbreviation of Driver A")
        parser.add_argument('--driver-b', type=str, default='RUS', help="Abbreviation of Driver B")
        parser.add_argument('--theme', type=str, default='carbon', choices=['carbon', 'slate', 'light', 'retro'], help="Dashboard Theme")
        parser.add_argument('--output', type=str, default=None, help="Output PNG path for Desktop (Mobile will append '_mobile')")
        
        args = parser.parse_args()
        
        build_dashboard(
            year=args.year,
            round_idx=args.round,
            session_name=args.session,
            driver_a=args.driver_a,
            driver_b=args.driver_b,
            theme=args.theme,
            output_filename=args.output
        )
