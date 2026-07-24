import os
import sys
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tbparse import SummaryReader
from pathlib import Path

# ============================================================
# Command line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Plot evaluation curves from multi-seed experiments')
parser.add_argument('base_dir', type=str, help='Path to the experiment directory containing seed folders')
parser.add_argument('--output', type=str, default=None, help='Output file path (default: base_dir/eval_curve.png)')
args = parser.parse_args()

base_dir = Path(args.base_dir)
output_path = Path(args.output) if args.output else base_dir
if output_path.is_dir():
    output_path /= 'eval_curve.png'

# ============================================================
# Style setup (seaborn whitegrid theme)
# ============================================================
sns.set_theme(style="whitegrid", font_scale=2.0)

# ============================================================
# Data loading
# ============================================================
all_steps_list = []
all_values_list = []

for seed_dir in sorted(base_dir.rglob('seed*')):
    if not seed_dir.is_dir():
        continue

    reader = SummaryReader(str(seed_dir))
    df = reader.scalars
    multiplier = 1
    if "evaluation/success" in df['tag'].values:
        tag = "evaluation/success"
        multiplier = 100
    else:
        tag = "evaluation/episode.normalized_return"
    tag_df = df[df['tag'] == tag].reset_index(drop=True)
    if len(tag_df) > 0:
        all_steps_list.append(tag_df['step'].values)
        all_values_list.append(tag_df['value'].values * multiplier)
        print(f'{str(seed_dir)}: {len(tag_df)} pts, final={tag_df["value"].iloc[-1]*multiplier:.1f}')

# Align to minimum length
min_len = min(len(s) for s in all_steps_list)
steps = np.array(all_steps_list[0][:min_len], dtype=int)
values = np.array([v[:min_len] for v in all_values_list])
mean = values.mean(axis=0)
std = values.std(axis=0)

print(f'\nSeeds: {len(all_steps_list)}, Points: {min_len}')
for i in range(min_len):
    print(f'  step={steps[i]}, mean={mean[i]:.2f} +/- {std[i]:.2f}')
print(f'  avrg={mean[-3:].mean():.2f} +/- {std[-3:].mean():.2f}')

# ============================================================
# Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Color palette
color = sns.color_palette("tab10")[0]

# Plot mean line
sns.lineplot(x=steps, y=mean, ax=ax, color=color, linewidth=2.5, label=base_dir.parts[2])

# Fill std band
ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

# Labels and title
ax.set_xlabel('Steps', fontsize=20)
ax.set_ylabel('Normalized Return', fontsize=20)
ax.set_title(base_dir.parts[1], fontsize=24, pad=15)

# Legend
ax.legend(loc='lower right', frameon=False, fontsize=16)

# Grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Tick formatting
ax.tick_params(axis='both', which='major', labelsize=14)

# Format x-axis with scientific notation if needed
if steps[-1] >= 1e5:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x/1e3:.0f}k' if x < 1e6 else f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'\nSaved to {output_path}')
