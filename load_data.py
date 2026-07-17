"""
load_data.py

Handles loading data from FastF1 API, caching, and retrieving telemetry and circuit details.
"""

import os
from typing import Tuple
import pandas as pd
import fastf1
import fastf1.core

# Silence verbose FastF1 logging for clean console interface
fastf1.set_log_level('WARNING')

def enable_fastf1_cache(cache_dir: str = ".fastf1_cache") -> None:
    """Enables caching for FastF1 data in a local folder to speed up data loading."""
    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)

def load_session_data(year: int, round_idx: int, session_name: str) -> fastf1.core.Session:
    """Loads and caches session data for a given year, round, and session."""
    enable_fastf1_cache()
    
    session = fastf1.get_session(year, round_idx, session_name)
    session.load()
    return session

def get_driver_fastest_lap(session: fastf1.core.Session, driver_abbr: str) -> Tuple[fastf1.core.Lap, pd.DataFrame]:
    """Retrieves the fastest lap and its telemetry with distance added for a given driver."""
    driver_laps = session.laps.pick_drivers(driver_abbr)
    if driver_laps.empty:
        raise ValueError(f"No laps found for driver {driver_abbr} in this session.")
        
    fastest_lap = driver_laps.pick_fastest()
    if pd.isna(fastest_lap['LapTime']):
        # If pick_fastest fails to find a valid lap time, sort manually
        valid_laps = driver_laps.dropna(subset=['LapTime'])
        if valid_laps.empty:
            raise ValueError(f"No valid lap times found for driver {driver_abbr}.")
        fastest_lap = valid_laps.sort_values(by='LapTime').iloc[0]
        
    # Get telemetry data and add distance
    telemetry = fastest_lap.get_telemetry().add_distance()
    
    return fastest_lap, telemetry

def get_circuit_corners(session: fastf1.core.Session) -> pd.DataFrame:
    """Retrieves corner locations (number, distance, X, Y, angle, letter) from the session's circuit info."""
    try:
        circuit_info = session.get_circuit_info()
        if circuit_info is not None and hasattr(circuit_info, 'corners') and circuit_info.corners is not None:
            return circuit_info.corners
    except Exception as e:
        print(f"Warning: Could not retrieve circuit info corners: {e}")
        
    # Return empty DataFrame with expected columns if it fails
    return pd.DataFrame(columns=['Number', 'Letter', 'Distance', 'X', 'Y', 'Angle'])
