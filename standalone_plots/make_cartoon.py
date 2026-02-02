import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle
from matplotlib import transforms

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def add_radial_inflow_arrows(ax, center=(0.0, 0.0),
                             r_outer=7.5, r_inner=3.5,
                             n_arrows=10,
                             arrow_kwargs=None):
    if arrow_kwargs is None:
        arrow_kwargs = {}

    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_arrows, endpoint=False)

    for ang in angles:
        x_out = cx + r_outer * np.cos(ang)
        y_out = cy + r_outer * np.sin(ang)
        x_in = cx + r_inner * np.cos(ang)
        y_in = cy + r_inner * np.sin(ang)

        arr = FancyArrowPatch(
            (x_out, y_out),
            (x_in, y_in),
            arrowstyle="-|>",
            mutation_scale=12.0,
            linewidth=1.8,
            **arrow_kwargs
        )
        ax.add_patch(arr)


def add_stream_with_arrow(ax, angle_deg, r_start=10.0, r_end=4.0,
                          flatten=0.3, stream_lw=2.0,
                          stream_color="#8b5fbf",
                          arrow_color="#4b2c70"):
    """GSE-like radial stream with arrow pointing inward."""
    ang = np.deg2rad(angle_deg)

    r_vals = np.linspace(r_start, r_end, 300)
    x_vals = r_vals * np.cos(ang)
    y_vals = flatten * r_vals * np.sin(ang)

    # Stream line
    ax.plot(x_vals, y_vals, linestyle="-", linewidth=stream_lw,
            color=stream_color, alpha=0.85, zorder=4)

    # Arrow halfway in
    r_mid_outer = (2 * r_start + r_end) / 3.0
    r_mid_inner = (r_start + 2 * r_end) / 3.0

    x_out = r_mid_outer * np.cos(ang)
    y_out = flatten * r_mid_outer * np.sin(ang)
    x_in = r_mid_inner * np.cos(ang)
    y_in = flatten * r_mid_inner * np.sin(ang)

    arr = FancyArrowPatch(
        (x_out, y_out), (x_in, y_in),
        arrowstyle="-|>", mutation_scale=10.0,
        linewidth=3, color=arrow_color, zorder=9
    )
    ax.add_patch(arr)


def add_bar_inflow(ax, bar_angle_deg=30.0,
                   r_start=7.5, r_end=3.5,
                   n_arrows=3, color="#0b4f6c"):
    """Arrows along bar axis pointing toward center."""
    ang = np.deg2rad(bar_angle_deg)
    positions = np.linspace(r_start, r_end, n_arrows)

    for r in positions:
        # Positive side
        x_out = r * np.cos(ang)
        y_out = r * np.sin(ang)
        x_in = (r - 2.2) * np.cos(ang)
        y_in = (r - 2.2) * np.sin(ang)

        arr1 = FancyArrowPatch(
            (x_out, y_out), (x_in, y_in),
            arrowstyle="-|>", mutation_scale=11.0,
            linewidth=1.6, color=color, zorder=8
        )
        ax.add_patch(arr1)

        # Negative side
        arr2 = FancyArrowPatch(
            (-x_out, -y_out), (-x_in, -y_in),
            arrowstyle="-|>", mutation_scale=11.0,
            linewidth=1.6, color=color, zorder=8
        )
        ax.add_patch(arr2)


# ---------------------------------------------------------------------
# Create figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
ax1, ax2 = axes

for ax in axes:
    ax.set_xlim(-12, 12)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal", "box")
    ax.axis("off")

# ---------------------------------------------------------------------
# PANEL 1: FIRST INFALL (t ~ 0.1 Gyr) - Primordial Collapse
# ---------------------------------------------------------------------

# Faint primordial gas cloud
primordial_gas = Circle(
    (0.0, 0.0), radius=9.0,
    facecolor="#e8f4f8", edgecolor="#90c4de",
    linewidth=1.5, linestyle="--", alpha=0.5, zorder=1
)
ax1.add_patch(primordial_gas)

# Central bulge (forming)
bulge1 = Ellipse(
    (0.0, 0.0), width=5.2, height=5.2,
    facecolor="#f2b35c", edgecolor="black",
    linewidth=2.0, zorder=5
)
ax1.add_patch(bulge1)

# Radial inflow arrows (symmetric collapse)
add_radial_inflow_arrows(
    ax1, center=(0.0, 0.0),
    r_outer=8.5, r_inner=4.2, n_arrows=16,
    arrow_kwargs={"color": "#2e7da8", "alpha": 0.85}
)

# Labels and annotations
#ax1.text(0, -10.2, "(a) First Infall", fontsize=15, weight='bold', ha='center')
#ax1.text(0, 8.5, r"$t_1 \sim 0.1$ Gyr, $\tau_1 \sim 0.09$ Gyr", fontsize=11, ha='center', style='italic')
#ax1.text(0, 7.5, r"SFE $\sim 3$ Gyr$^{-1}$, ~60% of mass", fontsize=10, ha='center')

# Info box
#ax1.text(-10, 5.5, "Rapid primordial\ncollapse", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='#2e7da8', linewidth=1.5), ha='left', va='top')

# ---------------------------------------------------------------------
# PANEL 2: SECOND INFALL (t ~ 5.1 Gyr) - Bar + GSE accretion
# ---------------------------------------------------------------------

# Thick disk
thick_disk = Ellipse(
    (0.0, 0.0), width=24.0, height=4.0,
    facecolor="#c8c8c8", edgecolor="black",
    linewidth=1.3, alpha=0.8, zorder=1
)
ax2.add_patch(thick_disk)

# Thin disk
thin_disk = Ellipse(
    (0.0, 0.0), width=23.0, height=1.2,
    facecolor="#fff46a", edgecolor="black",
    linewidth=1.3, alpha=0.9, zorder=2
)
ax2.add_patch(thin_disk)

# Bulge (already formed)
bulge2 = Ellipse(
    (0.0, 0.0), width=5.2, height=5.2,
    facecolor="#f2b35c", edgecolor="black",
    linewidth=2.0, zorder=5
)
ax2.add_patch(bulge2)

# Bar
bar_length = 9.0
bar_width = 1.5
bar_angle_deg = 30.0

bar = Rectangle(
    (-bar_length / 2.0, -bar_width / 2.0),
    bar_length, bar_width,
    facecolor="#e07b39", edgecolor="black",
    linewidth=1.5, zorder=6
)
bar.set_transform(transforms.Affine2D().rotate_deg(bar_angle_deg) + ax2.transData)
ax2.add_patch(bar)

# Bar-driven inflows
add_bar_inflow(
    ax2, bar_angle_deg=bar_angle_deg,
    r_start=7.2, r_end=3.8, n_arrows=3,
    color="#0b4f6c"
)

# GSE-like streams
stream_angles = [140.0, -40.0]
for ang in stream_angles:
    add_stream_with_arrow(
        ax2, angle_deg=ang, r_start=21.0, r_end=2.5,
        flatten=0.28, stream_lw=16.2,
        stream_color="#8b5fbf", arrow_color="#4b2c70"
    )

# GSE debris particles
rng = np.random.default_rng(420)
for ang in stream_angles:
    rad = np.deg2rad(ang)
    r_samples = rng.uniform(1.5, 100.5, size=1000)
    x = r_samples * np.cos(rad) + rng.normal(scale=0.48, size=len(r_samples))
    y = 0.28 * r_samples * np.sin(rad) + rng.normal(scale=0.58, size=len(r_samples))
    ax2.scatter(x, y, s=8, color="#a33592",
               edgecolor='black', linewidths=0.3,
               alpha=0.75, zorder=5)

# Labels
#ax2.text(0, -10.2, "(b) Second Infall", fontsize=15,weight='bold', ha='center')
#ax2.text(0, 8.5, r"$t_2 \sim 5.1$ Gyr, $\tau_2 \sim 1.7$ Gyr",fontsize=11, ha='center', style='italic')
#ax2.text(0, 7.5, r"$\Delta$SFE $\sim 0.72$, ~40% of mass",fontsize=10, ha='center')

# Legend boxes
ax2.text(-10.5, 7.0, "Gas sources:", fontsize=10, weight='bold',
         ha='left', va='top')
ax2.text(-10.5, 6.0, "• Bar inflows", fontsize=9,
         ha='left', color='#0b4f6c', weight='bold')
ax2.text(-10.5, 5.2, "• GSE merger", fontsize=9,
         ha='left', color='#8b5fbf', weight='bold')
ax2.text(-10.5, 4.4, "• Disk gas", fontsize=9,
         ha='left', color='#666666')

plt.tight_layout()
plt.savefig("cartoon_two_infall_improved.png", dpi=300,
            bbox_inches="tight", facecolor='white')
#plt.show()