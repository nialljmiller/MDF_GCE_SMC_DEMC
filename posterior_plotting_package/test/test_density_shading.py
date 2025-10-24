#!/usr/bin/env python3
"""
Test script for density-based posterior shading.

This script creates a simple visualization to demonstrate the density shading
effect compared to uniform fill_between.
"""

import numpy as np
import matplotlib.pyplot as plt
from posterior_utils_density import plot_density_posterior_simple

# Generate mock data
np.random.seed(42)
x = np.linspace(0, 14, 100)

# Create median and uncertainty bands
median = -1.5 + 0.1 * x + 0.05 * np.sin(x)
lower = median - 0.2 - 0.05 * x
upper = median + 0.2 + 0.05 * x

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Traditional uniform fill_between
ax1.plot(x, median, color='crimson', lw=2.5, label='Median', zorder=3)
ax1.fill_between(x, lower, upper, color='crimson', alpha=0.25, 
                label='1σ (uniform)', zorder=2)
ax1.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
ax1.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
ax1.set_title('Traditional: Uniform Fill', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.2)

# Right: Density-based gradient shading
plot_density_posterior_simple(ax2, x, median, lower, upper, 
                             color='crimson', n_levels=20, 
                             zorder=2, label='1σ (density)')
ax2.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
ax2.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
ax2.set_title('New: Density-Based Gradient', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/ubuntu/Downloads/density_shading_comparison.png', 
           dpi=300, bbox_inches='tight')
print("✓ Comparison plot saved to: /home/ubuntu/Downloads/density_shading_comparison.png")

# Create a second figure showing the effect with multiple levels
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (ax, n_levels) in enumerate(zip(axes.flat, [5, 10, 20, 30])):
    plot_density_posterior_simple(ax, x, median, lower, upper,
                                 color='crimson', n_levels=n_levels,
                                 zorder=2, label=f'{n_levels} levels')
    ax.set_xlabel('Age (Gyr)', fontsize=11)
    ax.set_ylabel('[Fe/H]', fontsize=11)
    ax.set_title(f'n_levels = {n_levels}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/ubuntu/Downloads/density_levels_comparison.png',
           dpi=300, bbox_inches='tight')
print("✓ Levels comparison saved to: /home/ubuntu/Downloads/density_levels_comparison.png")

print("\nDensity shading test completed successfully!")
print("The gradient shading creates a smooth transition from dark (high density)")
print("at the median to light (low density) at the edges of the uncertainty band.")

