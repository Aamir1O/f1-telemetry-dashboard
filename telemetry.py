"""
telemetry.py

Handles telemetry alignment, interpolation, and delta calculations between drivers.
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np

def interpolate_telemetry(tel_a: pd.DataFrame, tel_b: pd.DataFrame, step_meters: float = 2.0) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Interpolates both telemetry datasets onto a shared distance grid using linear interpolation.
    
    Args:
        tel_a: Telemetry DataFrame for Driver A (must contain 'Distance', 'Speed', etc.)
        tel_b: Telemetry DataFrame for Driver B
        step_meters: Distance step size in meters for the common grid
        
    Returns:
        common_distance: Array of distances from 0 to the minimum maximum lap distance
        interp_a: Dict containing interpolated arrays for Driver A
        interp_b: Dict containing interpolated arrays for Driver B
    """
    # 1. Ensure distance starts at 0 and adds distance column if needed
    dist_a = tel_a['Distance'].values
    dist_b = tel_b['Distance'].values
    
    max_dist_a = dist_a.max()
    max_dist_b = dist_b.max()
    
    # Common distance limit is the minimum of the two lap lengths
    limit = min(max_dist_a, max_dist_b)
    common_distance = np.arange(0, limit, step_meters)
    
    # Channels to interpolate
    channels = ['Speed', 'Throttle', 'Brake', 'RPM', 'DRS', 'X', 'Y']
    
    # We also need to interpolate Time. Time must be converted to total seconds first.
    # Note: FastF1 telemetry time can be a timedelta object.
    if 'Time' in tel_a.columns:
        if isinstance(tel_a['Time'].iloc[0], (pd.Timedelta, np.timedelta64)):
            time_sec_a = tel_a['Time'].dt.total_seconds().values
        else:
            time_sec_a = tel_a['Time'].values
    else:
        raise ValueError("Telemetry A must contain a Time column.")
        
    if 'Time' in tel_b.columns:
        if isinstance(tel_b['Time'].iloc[0], (pd.Timedelta, np.timedelta64)):
            time_sec_b = tel_b['Time'].dt.total_seconds().values
        else:
            time_sec_b = tel_b['Time'].values
    else:
        raise ValueError("Telemetry B must contain a Time column.")
        
    interp_a: Dict[str, np.ndarray] = {}
    interp_b: Dict[str, np.ndarray] = {}
    
    # Interpolate Time
    # Adjust for initial offset: align so both start at time 0
    time_sec_a = time_sec_a - time_sec_a[0]
    time_sec_b = time_sec_b - time_sec_b[0]
    
    interp_a['Time'] = np.interp(common_distance, dist_a, time_sec_a)
    interp_b['Time'] = np.interp(common_distance, dist_b, time_sec_b)
    
    # Interpolate standard channels
    for ch in channels:
        if ch not in tel_a.columns or ch not in tel_b.columns:
            # Attempt case-insensitive match
            match_a = [col for col in tel_a.columns if col.lower() == ch.lower()]
            match_b = [col for col in tel_b.columns if col.lower() == ch.lower()]
            if match_a and match_b:
                val_a = tel_a[match_a[0]].values
                val_b = tel_b[match_b[0]].values
            else:
                # If channel is completely missing, fill with zeros
                val_a = np.zeros_like(dist_a)
                val_b = np.zeros_like(dist_b)
        else:
            val_a = tel_a[ch].values
            val_b = tel_b[ch].values
            
        # Perform interpolation
        interp_a[ch] = np.interp(common_distance, dist_a, val_a)
        interp_b[ch] = np.interp(common_distance, dist_b, val_b)
        
        # Post-process specific fields to keep them appropriate types
        if ch == 'Brake':
            # Brake is binary. Values > 0.5 are True/1.0, otherwise 0.0
            interp_a[ch] = (interp_a[ch] > 0.5).astype(float)
            interp_b[ch] = (interp_b[ch] > 0.5).astype(float)
        elif ch == 'DRS':
            # DRS is binary/numeric
            interp_a[ch] = (interp_a[ch] > 8.0).astype(float) # DRS values can be numbers or active status
            interp_b[ch] = (interp_b[ch] > 8.0).astype(float)
            
    return common_distance, interp_a, interp_b

def calculate_delta_time(time_a: np.ndarray, time_b: np.ndarray) -> np.ndarray:
    """
    Calculates the time delta between two aligned drivers.
    Delta = Time_A - Time_B
    
    If Delta < 0: Driver A is ahead (faster cumulative time).
    If Delta > 0: Driver B is ahead (faster cumulative time).
    """
    return time_a - time_b
