import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap

# Set up visual parameters
BG_COLOR = '#121212'
CARD_BG = '#1A1A1A'
BORDER_COLOR = '#333333'
TEXT_LIGHT = '#F5F5F5'
TEXT_MUTED = '#8A8A8A'
FERRARI_RED = '#E80020'
GREEN_ACCENT = '#39B54A'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'text.color': TEXT_LIGHT,
    'font.family': 'sans-serif',
    'font.size': 10
})

def draw_wrapped_text(ax, text, x, y, width, font_size=10, color=TEXT_LIGHT, fontweight='normal', line_spacing=0.025):
    lines = textwrap.wrap(text, width=width)
    curr_y = y
    for line in lines:
        ax.text(x, curr_y, line, fontsize=font_size, color=color, fontweight=fontweight, va='top', ha='left')
        curr_y -= line_spacing
    return curr_y

def main():
    fig = plt.figure(figsize=(14, 9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # 1. Main Background Card
    main_border = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        mutation_aspect=1.0,
        linewidth=1.2,
        edgecolor=BORDER_COLOR,
        facecolor=BG_COLOR,
        zorder=1
    )
    ax.add_patch(main_border)

    # 2. Header Panel
    header_bg = FancyBboxPatch(
        (0.03, 0.83), 0.94, 0.13,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        mutation_aspect=1.0,
        linewidth=1.0,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        zorder=2
    )
    ax.add_patch(header_bg)
    
    # Header Accent line
    ax.fill_between([0.035, 0.045], 0.84, 0.95, color=FERRARI_RED, zorder=3)
    
    ax.text(0.06, 0.92, "2026 BRITISH GRAND PRIX  •  SILVERSTONE", fontsize=9.5, fontweight='bold', color=TEXT_MUTED, zorder=3)
    ax.text(0.06, 0.88, "RACE Pace Analysis: Leclerc vs. Hamilton", fontsize=18, fontweight='black', color=TEXT_LIGHT, zorder=3)
    ax.text(0.06, 0.85, "Leclerc maximized race pace while Hamilton excelled in one-lap speed.", fontsize=11, color=TEXT_MUTED, fontweight='medium', style='italic', zorder=3)

    # 3. Left Panel (Headline Numbers & Driver Cards)
    left_bg = FancyBboxPatch(
        (0.03, 0.08), 0.45, 0.73,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        mutation_aspect=1.0,
        linewidth=1.0,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        zorder=2
    )
    ax.add_patch(left_bg)
    
    ax.text(0.05, 0.77, "LAP & TYRE PERFORMANCE", fontsize=10, fontweight='bold', color=TEXT_MUTED, zorder=3)
    
    # Hamilton Card
    ham_card = FancyBboxPatch(
        (0.05, 0.44), 0.41, 0.30,
        boxstyle="round,pad=0.0,rounding_size=0.01",
        mutation_aspect=1.0,
        linewidth=0.8,
        edgecolor='#444444',
        facecolor='#181818',
        zorder=3
    )
    ax.add_patch(ham_card)
    ax.fill_between([0.055, 0.062], 0.46, 0.72, color='#555555', zorder=4) # Silver/Gray accent
    ax.text(0.075, 0.69, "LEWIS HAMILTON  #44", fontsize=12, fontweight='bold', color=TEXT_LIGHT, zorder=4)
    ax.text(0.075, 0.66, "FERRARI  |  ONE-LAP SUPREMACY", fontsize=8, fontweight='bold', color=TEXT_MUTED, zorder=4)
    
    ax.text(0.075, 0.58, "LAP TIME", fontsize=8, color=TEXT_MUTED, fontweight='bold', zorder=4)
    ax.text(0.075, 0.52, "1:29.260", fontsize=18, color=GREEN_ACCENT, fontweight='black', zorder=4)
    ax.text(0.075, 0.48, "FASTEST LAP (QUALIFYING TRIM)", fontsize=7.5, color=GREEN_ACCENT, fontweight='bold', zorder=4)
    
    ax.text(0.26, 0.58, "TYRE DETAILS", fontsize=8, color=TEXT_MUTED, fontweight='bold', zorder=4)
    ax.text(0.26, 0.52, "SOFT", fontsize=14, color=FERRARI_RED, fontweight='black', zorder=4)
    ax.text(0.26, 0.48, "2 LAPS USED (FRESH)", fontsize=7.5, color=TEXT_MUTED, fontweight='medium', zorder=4)

    # Leclerc Card
    lec_card = FancyBboxPatch(
        (0.05, 0.11), 0.41, 0.30,
        boxstyle="round,pad=0.0,rounding_size=0.01",
        mutation_aspect=1.0,
        linewidth=0.8,
        edgecolor='#444444',
        facecolor='#181818',
        zorder=3
    )
    ax.add_patch(lec_card)
    ax.fill_between([0.055, 0.062], 0.13, 0.39, color=FERRARI_RED, zorder=4) # Red accent
    ax.text(0.075, 0.36, "CHARLES LECLERC  #16", fontsize=12, fontweight='bold', color=TEXT_LIGHT, zorder=4)
    ax.text(0.075, 0.33, "FERRARI  |  RACE PACE FOCUS", fontsize=8, fontweight='bold', color=FERRARI_RED, zorder=4)
    
    ax.text(0.075, 0.25, "LAP TIME", fontsize=8, color=TEXT_MUTED, fontweight='bold', zorder=4)
    ax.text(0.075, 0.19, "1:29.859", fontsize=18, color=TEXT_LIGHT, fontweight='black', zorder=4)
    ax.text(0.075, 0.15, "+0.599s GAP", fontsize=8, color=FERRARI_RED, fontweight='bold', zorder=4)
    
    ax.text(0.26, 0.25, "TYRE DETAILS", fontsize=8, color=TEXT_MUTED, fontweight='bold', zorder=4)
    ax.text(0.26, 0.19, "SOFT", fontsize=14, color=FERRARI_RED, fontweight='black', zorder=4)
    ax.text(0.26, 0.15, "4 LAPS USED (DEGRADED)", fontsize=7.5, color=TEXT_MUTED, fontweight='medium', zorder=4)

    # 4. Right Panel (Telemetry & Strategy Insights)
    right_bg = FancyBboxPatch(
        (0.50, 0.08), 0.47, 0.73,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        mutation_aspect=1.0,
        linewidth=1.0,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        zorder=2
    )
    ax.add_patch(right_bg)
    
    ax.text(0.52, 0.77, "TELEMETRY & TECHNICAL INSIGHTS", fontsize=10, fontweight='bold', color=TEXT_MUTED, zorder=3)
    
    y_ptr = 0.73
    
    # Point 1: Braking vs Throttle
    ax.text(0.52, y_ptr, "1. Late Braking vs. Early Throttle Exits", fontsize=11, fontweight='bold', color=FERRARI_RED, zorder=3)
    y_ptr -= 0.03
    
    text_braking = "* Hamilton (Late Braking): Hamilton gained time by braking later and carrying high speed into heavy braking zones. He registered huge speed deltas at T1 (+1.8 km/h), T7 (+5.8 km/h), and T12 (+27.4 km/h)."
    y_ptr = draw_wrapped_text(ax, text_braking, 0.52, y_ptr, 65, font_size=9, color=TEXT_LIGHT, line_spacing=0.022)
    y_ptr -= 0.01
    
    text_exit = "* Leclerc (Early Exit): Leclerc prioritized exits and stability. By braking slightly earlier, he stabilized the car and got on the throttle sooner, carrying superior speed out of slow/medium corners: T3 (-5.0 km/h), T5 (-2.4 km/h), T14 (-4.5 km/h), and T18 (-10.0 km/h)."
    y_ptr = draw_wrapped_text(ax, text_exit, 0.52, y_ptr, 65, font_size=9, color=TEXT_LIGHT, line_spacing=0.022)
    y_ptr -= 0.03

    # Point 2: High Speed Flow & Tires
    ax.text(0.52, y_ptr, "2. High-Speed Flow and Tire Management", fontsize=11, fontweight='bold', color=FERRARI_RED, zorder=3)
    y_ptr -= 0.03
    
    text_highspeed = "* Maggotts–Becketts: Leclerc carried higher speed stability through this fast sector, minimizing steering correction and preserving his tyres."
    y_ptr = draw_wrapped_text(ax, text_highspeed, 0.52, y_ptr, 65, font_size=9, color=TEXT_LIGHT, line_spacing=0.022)
    y_ptr -= 0.01
    
    text_tyres = "* Tyre Degradation: Leclerc managed degradation much more effectively over long stints. His fastest lap was set on softs with 4 laps of wear, compared to Hamilton's fresh 2-lap soft stint."
    y_ptr = draw_wrapped_text(ax, text_tyres, 0.52, y_ptr, 65, font_size=9, color=TEXT_LIGHT, line_spacing=0.022)
    y_ptr -= 0.03

    # Point 3: Overall Race Strategy
    ax.text(0.52, y_ptr, "3. Summary: Race Pace vs. One-Lap Glory", fontsize=11, fontweight='bold', color=GREEN_ACCENT, zorder=3)
    y_ptr -= 0.03
    
    text_summary = "While Hamilton won the battle for single-lap speed, Leclerc's smoother inputs, early throttle application, and optimized corner exits translated into superior long-run race pace and extended tyre life, giving him the race advantage."
    y_ptr = draw_wrapped_text(ax, text_summary, 0.52, y_ptr, 65, font_size=9.5, color=TEXT_LIGHT, fontweight='medium', line_spacing=0.024)

    # 5. Footer Panel
    ax.text(0.03, 0.04, "Data Sources: FastF1 Telemetry Comparison Dashboard", fontsize=8, color=TEXT_MUTED, fontweight='bold', zorder=3)


    # Save Infographic
    output_filename = "race_analysis.png"
    plt.savefig(
        output_filename,
        dpi=300,
        facecolor=BG_COLOR,
        edgecolor='none',
        bbox_inches='tight'
    )
    plt.close()
    print(f"Infographic successfully saved as: {output_filename}")

if __name__ == '__main__':
    main()
