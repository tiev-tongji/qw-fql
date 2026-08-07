import os
import re
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
def load_method_data(method_dir: Path):
    """Load all seed data under a method directory.

    Returns a dict with:
      - steps:  (T,)   aligned env steps
      - values: (N, T) raw values of all seeds
      - mean/std: (T,) cross-seed statistics
    """
    print(f'Loading data from {str(method_dir)}')
    all_steps_list = []
    all_values_list = []

    for seed_dir in sorted(method_dir.rglob('seed*')):
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

    if not all_steps_list:
        print(f'Warning: no seed data found in {str(method_dir)}, skipping')
        return None

    # Align to minimum length
    min_len = min(len(s) for s in all_steps_list)
    steps = np.array(all_steps_list[0][:min_len], dtype=int)
    values = np.array([v[:min_len] for v in all_values_list])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return {
        'steps': steps,
        'values': values,
        'mean': mean,
        'std': std,
    }

# ============================================================
# Auto-detect directory level based on hierarchy:
#   logs/task_name/method_name    -> method level  -> aggregate all seeds
#   logs/task_name                -> task level    -> iterate method subdirs
# Also works for layouts with/without an intermediate timestamp dir:
#   direct children are seed* or all children are timestamp-like (YYYYMMDD_HHMMSS)
#   => method level; otherwise each subdir is treated as a method.
# ============================================================
TS_PATTERN = re.compile(r'^\d{8}_\d{6}$')

def contains_seeds(d: Path):
    return any(p.is_dir() for p in d.rglob('seed*'))

seed_dirs = sorted(p for p in base_dir.rglob('seed*') if p.is_dir())
if not seed_dirs:
    print(f'Error: no seed* directories found under {base_dir}')
    sys.exit(1)

subdirs = sorted(c for c in base_dir.iterdir() if c.is_dir() and contains_seeds(c))
has_direct_seeds = any(c.name.startswith('seed') for c in base_dir.iterdir() if c.is_dir())

if has_direct_seeds or all(TS_PATTERN.match(c.name) for c in subdirs):
    # method level: seeds directly inside, or only timestamp-like intermediate dirs
    all_data = {base_dir.name: load_method_data(base_dir)}
else:
    # task level: each subdir is one method
    all_data = {}
    for method_dir in subdirs:
        data = load_method_data(method_dir)
        if data is not None:
            all_data[method_dir.name] = data

if not all_data:
    print(f'Error: no valid experiment data found under {base_dir}')
    sys.exit(1)

# ============================================================
# Summary statistics per method (across all seeds)
# ============================================================
print('\n' + '=' * 60)
print(f'Summary for {base_dir}')
print('=' * 60)
for exp_name, data in all_data.items():
    values = data['values']            # shape: (n_seeds, T)
    final_vals = values[:, -1]         # last eval point of each seed
    last3_vals = values[:, -3:]        # last 3 eval points of each seed
    print(f'{exp_name} ({values.shape[0]} seeds)')
    print(f'  final step : mean={final_vals.mean():8.2f}  std={final_vals.std():8.2f}')
    print(f'  last 3 step: mean={last3_vals.mean():8.2f}  std={last3_vals.std():8.2f}')
print('=' * 60)

# ============================================================
# Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Color palette
colors = sns.color_palette("tab10")

# Plot mean line
for i, (exp_name, data) in enumerate(all_data.items()):
    steps = data['steps']
    mean = data['mean']
    std = data['std']
    color = colors[i % len(colors)]
    sns.lineplot(x=steps, y=mean, ax=ax, color=color, linewidth=2.5, label=exp_name)

    # Fill std band
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

# Labels and title
ax.set_xlabel('Steps', fontsize=20)
ax.set_ylabel('Normalized Return', fontsize=20)
ax.set_title(base_dir.name, fontsize=24, pad=15)

# Legend
ax.legend(loc='lower right', frameon=False, fontsize=16)

# Grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Tick formatting
ax.tick_params(axis='both', which='major', labelsize=14)

# Format x-axis with scientific notation if needed
max_step = max(d['steps'][-1] for d in all_data.values())
if max_step >= 1e5:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x/1e3:.0f}k' if x < 1e6 else f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f'\nSaved to {output_path}')
