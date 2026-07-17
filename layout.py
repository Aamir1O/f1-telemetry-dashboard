"""
layout.py

Defines the dashboard GridSpec layout, creating the figure and subplots.
"""

from typing import Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import setup_matplotlib_params, get_color

def create_dashboard_layout() -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Creates an 18x10 inch dashboard layout using Matplotlib GridSpec.
    Returns:
        fig: The created Figure object.
        axes: A dictionary containing the axes objects for each panel.
    """
    # Initialize global matplotlib parameters
    setup_matplotlib_params()
    
    # Create the main figure
    fig = plt.figure(figsize=(18, 10), facecolor=get_color('bg_dark'))
    
    # Outer GridSpec:
    # Row 0: Top Cards and Track Map (height = 2.4 inches)
    # Row 1: Telemetry and Corner Table (height = 7.6 inches)
    # Columns: Card A/Left (width = 4.8), Map/Center (width = 8.4), Card B/Right (width = 4.8)
    outer_gs = gridspec.GridSpec(
        2, 3, 
        height_ratios=[2.4, 7.6], 
        width_ratios=[4.8, 8.4, 4.8],
        hspace=0.13, 
        wspace=0.13,
        left=0.02,
        right=0.98,
        top=0.94,
        bottom=0.022
    )
    
    # Top Row Subplots
    ax_card_a = fig.add_subplot(outer_gs[0, 0])
    ax_track = fig.add_subplot(outer_gs[0, 1])
    ax_card_b = fig.add_subplot(outer_gs[0, 2])
    
    # Bottom Row Right: Corner Table
    ax_table = fig.add_subplot(outer_gs[1, 2])
    
    # Bottom Row Left & Center: Subgrid for Telemetry Stack
    # 4 rows for Speed, Throttle, Brake, Delta Time
    telemetry_gs = gridspec.GridSpecFromSubplotSpec(
        4, 1, 
        subplot_spec=outer_gs[1, 0:2],
        height_ratios=[2.3, 1.2, 0.7, 2.6],
        hspace=0.08 # tight vertical integration
    )
    
    ax_speed = fig.add_subplot(telemetry_gs[0])
    ax_throttle = fig.add_subplot(telemetry_gs[1], sharex=ax_speed)
    ax_brake = fig.add_subplot(telemetry_gs[2], sharex=ax_speed)
    ax_delta = fig.add_subplot(telemetry_gs[3], sharex=ax_speed)
    
    # Hide x-tick labels for upper plots in the stack to prevent overlap and clutter
    plt.setp(ax_speed.get_xticklabels(), visible=False)
    plt.setp(ax_throttle.get_xticklabels(), visible=False)
    plt.setp(ax_brake.get_xticklabels(), visible=False)
    
    axes_dict = {
        'card_a': ax_card_a,
        'track': ax_track,
        'card_b': ax_card_b,
        'speed': ax_speed,
        'throttle': ax_throttle,
        'brake': ax_brake,
        'delta': ax_delta,
        'table': ax_table
    }
    
    return fig, axes_dict


def create_mobile_dashboard_layout() -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Creates a 10x18 inch mobile portrait dashboard layout using Matplotlib GridSpec.
    Returns:
        fig: The created Figure object.
        axes: A dictionary containing the axes objects for each panel.
    """
    # Initialize global matplotlib parameters
    setup_matplotlib_params()
    
    # Create the mobile portrait figure
    fig = plt.figure(figsize=(10, 18), facecolor=get_color('bg_dark'))
    
    # Outer GridSpec:
    # Row 0: Driver Cards side-by-side (height = 2.2 inches)
    # Row 1: Track Map (height = 4.0 inches, spanning both columns)
    # Row 2: Telemetry Stack (height = 7.2 inches, spanning both columns)
    # Row 3: Corner Table (height = 3.6 inches, spanning both columns)
    outer_gs = gridspec.GridSpec(
        4, 2, 
        height_ratios=[2.2, 4.0, 7.2, 3.6], 
        width_ratios=[1.0, 1.0],
        hspace=0.14, 
        wspace=0.12,
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.014
    )
    
    # Subplots
    # Row 0: Driver Cards (left and right)
    ax_card_a = fig.add_subplot(outer_gs[0, 0])
    ax_card_b = fig.add_subplot(outer_gs[0, 1])
    
    # Row 1: Track Map (spanning both columns)
    ax_track = fig.add_subplot(outer_gs[1, :])
    
    # Row 3: Corner Table (spanning both columns)
    ax_table = fig.add_subplot(outer_gs[3, :])
    
    # Row 2: Telemetry Stack subgrid (spanning both columns)
    telemetry_gs = gridspec.GridSpecFromSubplotSpec(
        4, 1, 
        subplot_spec=outer_gs[2, :],
        height_ratios=[2.2, 1.1, 0.7, 2.4],
        hspace=0.08 # tight vertical integration
    )
    
    ax_speed = fig.add_subplot(telemetry_gs[0])
    ax_throttle = fig.add_subplot(telemetry_gs[1], sharex=ax_speed)
    ax_brake = fig.add_subplot(telemetry_gs[2], sharex=ax_speed)
    ax_delta = fig.add_subplot(telemetry_gs[3], sharex=ax_speed)
    
    # Hide x-tick labels for upper plots in the stack to prevent overlap and clutter
    plt.setp(ax_speed.get_xticklabels(), visible=False)
    plt.setp(ax_throttle.get_xticklabels(), visible=False)
    plt.setp(ax_brake.get_xticklabels(), visible=False)
    
    axes_dict = {
        'card_a': ax_card_a,
        'track': ax_track,
        'card_b': ax_card_b,
        'speed': ax_speed,
        'throttle': ax_throttle,
        'brake': ax_brake,
        'delta': ax_delta,
        'table': ax_table
    }
    
    return fig, axes_dict

