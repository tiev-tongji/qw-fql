import os
import sys
import re
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tbparse import SummaryReader
from pathlib import Path
from itertools import product

# ============================================================
# Command line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Plot parameter grid heatmap from multi-seed experiments')
parser.add_argument('base_dir', type=str, help='Path to the experiment directory (e.g. logs/antmaze-giant-navigate-singletask-v0)')
parser.add_argument('--output', type=str, default=None, help='Output PDF file path (default: base_dir/grid_heatmap.pdf)')
parser.add_argument('--tag', type=str, default=None,
                    help='Tag to read (default: auto-detect evaluation/success or evaluation/episode.normalized_return)')
args = parser.parse_args()

base_dir = Path(args.base_dir)
output_path = Path(args.output) if args.output else base_dir / 'grid_heatmap.pdf'

# ============================================================
# Style setup
# ============================================================
sns.set_theme(style="white", font_scale=2.0)

# ============================================================
# Parse parameter directories
# ============================================================
# Expected format: dfq_{alpha}_{temperature}
param_dirs = {}
alpha_set = set()
temp_set = set()

for d in sorted(base_dir.iterdir()):
    if not d.is_dir():
        continue
    match = re.match(r'dfq_([\d.]+)_([\d.]+)$', d.name)
    if match:
        alpha = float(match.group(1))
        temperature = float(match.group(2))
        alpha_set.add(alpha)
        temp_set.add(temperature)
        param_dirs[(alpha, temperature)] = d

alphas = sorted(alpha_set)
temps = sorted(temp_set)
print(f'Alpha values: {alphas}')
print(f'Temperature values: {temps}')
print(f'Found {len(param_dirs)} parameter combinations')

# ============================================================
# Data loading
# ============================================================
def load_final_values(exp_dir, tag=None):
    """Load final evaluation values from all seed directories under exp_dir."""
    all_final_values = []

    for seed_dir in sorted(exp_dir.rglob('seed*')):
        if not seed_dir.is_dir():
            continue
        try:
            reader = SummaryReader(str(seed_dir))
            df = reader.scalars
        except Exception as e:
            print(f'  Warning: failed to read {seed_dir}: {e}')
            continue

        if len(df) == 0:
            continue

        # Auto-detect tag
        if tag is not None:
            use_tag = tag
        elif "evaluation/success" in df['tag'].values:
            use_tag = "evaluation/success"
        else:
            use_tag = "evaluation/episode.normalized_return"

        tag_df = df[df['tag'] == use_tag].reset_index(drop=True)
        if len(tag_df) > 0:
            final_val = tag_df['value'].iloc[-1]
            # Convert success rate to percentage
            if use_tag == "evaluation/success":
                final_val *= 100
            all_final_values.append(final_val)

    if not all_final_values:
        return None, None
    return np.mean(all_final_values), np.std(all_final_values)


# Build grid values
grid_mean = np.full((len(alphas), len(temps)), np.nan)
grid_std = np.full((len(alphas), len(temps)), np.nan)

for (alpha, temp), exp_dir in param_dirs.items():
    i = alphas.index(alpha)
    j = temps.index(temp)
    mean_val, std_val = load_final_values(exp_dir, tag=args.tag)
    if mean_val is not None:
        grid_mean[i, j] = mean_val
        grid_std[i, j] = std_val
        print(f'  dfq_{alpha}_{temp}: mean={mean_val:.1f}, std={std_val:.1f}')
    else:
        print(f'  dfq_{alpha}_{temp}: NO DATA')

# ============================================================
# Plotting heatmap (rectangle-based with gaps between cells)
# ============================================================
fig, ax = plt.subplots(figsize=(len(temps) * 1.8, len(alphas) * 1.5))

# Create green colormap: lightest (for 0/min) -> darkest (for max)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'green_heat', ['#f0fff0', '#006400']
)
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(grid_mean) if not np.all(np.isnan(grid_mean)) else 100)

# Cell size and gap
cell_size = 1.0
gap = 0.08  # gap between cells

# Draw each cell as a rectangle (x=alpha, y=tau)
for i in range(len(alphas)):
    for j in range(len(temps)):
        mean_val = grid_mean[i, j]
        std_val = grid_std[i, j]

        # Compute cell position with gaps: x=alpha, y=tau
        x = i * (cell_size + gap)
        y = j * (cell_size + gap)  # j=0 (smallest tau) at bottom

        # Cell color
        if not np.isnan(mean_val):
            cell_color = cmap(norm(mean_val))
        else:
            cell_color = '#e0e0e0'

        # Draw rectangle
        rect = matplotlib.patches.Rectangle(
            (x, y), cell_size, cell_size,
            facecolor=cell_color, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)

        # Center text
        cx = x + cell_size / 2
        cy = y + cell_size / 2

        if not np.isnan(mean_val):
            text = f'{mean_val:.0f}±{std_val:.0f}'
            normalized = norm(mean_val)
            color = 'white' if normalized > 0.4 else 'black'
            ax.text(cx, cy, text, ha='center', va='center', fontsize=14, color=color, fontweight='bold')
        else:
            ax.text(cx, cy, 'N/A', ha='center', va='center', fontsize=10, color='gray')

# Set axis ticks at cell centers
alpha_labels = [f'{a:g}' for a in alphas]
temp_labels = [f'{t:g}' for t in temps]

xtick_pos = [i * (cell_size + gap) + cell_size / 2 for i in range(len(alphas))]
ytick_pos = [j * (cell_size + gap) + cell_size / 2 for j in range(len(temps))]

ax.set_xticks(xtick_pos)
ax.set_xticklabels(alpha_labels, fontsize=16)
ax.set_yticks(ytick_pos)
ax.set_yticklabels(temp_labels, fontsize=16)

ax.set_xlabel(r'$\alpha$', fontsize=20)
ax.set_ylabel(r'$\tau$', fontsize=20, rotation=0, labelpad=15)

# Remove outer border/spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Set limits to fit all cells with small padding
ax.set_xlim(-0.1, len(alphas) * (cell_size + gap) - gap + 0.1)
ax.set_ylim(-0.1, len(temps) * (cell_size + gap) - gap + 0.1)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
print(f'\nSaved to {output_path}')
