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
parser = argparse.ArgumentParser(description='Plot validation/actor/data_action_rate curves from multi-seed experiments')
parser.add_argument('base_dir', type=str, help='Path to the experiment directory containing seed folders')
parser.add_argument('--output', type=str, default=None, help='Output file path (default: base_dir/data_action_rate.pdf)')
parser.add_argument('--sample_interval', type=int, default=50000, help='Sample every N steps (default: 50000)')
parser.add_argument('--smooth_window', type=int, default=5, help='Moving average window size for smoothing (default: 5, 0=disable)')
args = parser.parse_args()

base_dir = Path(args.base_dir)
output_path = Path(args.output) if args.output else base_dir
if output_path.is_dir():
    output_path /= 'data_action_rate.pdf'

# ============================================================
# Style setup (seaborn whitegrid theme)
# ============================================================
sns.set_theme(style="whitegrid", font_scale=2.0)

# ============================================================
# Data loading
# ============================================================
TAG = "training/actor/data_action_rate"

def load_one_exp_data(exp_data_dir: Path):
    print(f'Loading data from {str(exp_data_dir)}')
    all_steps_list = []
    all_values_list = []

    for seed_dir in sorted(exp_data_dir.rglob('seed*')):
        if not seed_dir.is_dir():
            continue

        reader = SummaryReader(str(seed_dir))
        df = reader.scalars
        tag_df = df[df['tag'] == TAG].reset_index(drop=True)
        if len(tag_df) > 0:
            all_steps_list.append(tag_df['step'].values)
            all_values_list.append(tag_df['value'].values)
            print(f'{str(seed_dir)}: {len(tag_df)} pts, final={tag_df["value"].iloc[-1]:.4f}')

    if not all_steps_list:
        print(f'Warning: no seed data found in {str(exp_data_dir)}, skipping')
        return None

    # Align to minimum length
    min_len = min(len(s) for s in all_steps_list)
    steps = np.array(all_steps_list[0][:min_len], dtype=int)
    values = np.array([v[:min_len] for v in all_values_list])

    # Sample every N steps
    interval = args.sample_interval
    if interval > 0 and len(steps) > 1:
        sample_mask = (steps % interval == 0) | (np.arange(len(steps)) == 0)
        steps = steps[sample_mask]
        values = values[:, sample_mask]

    mean = values.mean(axis=0)
    std = values.std(axis=0)

    # Moving average smoothing
    window = args.smooth_window
    if window > 1 and len(mean) > window:
        kernel = np.ones(window) / window
        mean = np.convolve(mean, kernel, mode='valid')
        std = np.convolve(std, kernel, mode='valid')
        steps = steps[window-1:window-1+len(mean)]

    return {
        'steps': steps,
        'mean': mean,
        'std': std,
    }

# Load data from all experiment directories
all_data = {}
for exp_dir in sorted(base_dir.iterdir()):
    if exp_dir.is_dir():
        all_data[exp_dir.name] = load_one_exp_data(exp_dir)

for exp_name, data in all_data.items():
    if data is None:
        continue
    print(exp_name)
    print(f'  final_mean={data["mean"][-1]:.4f} +/- {data["std"][-1]:.4f}')
    print(f'  last3_mean={data["mean"][-3:].mean():.4f} +/- {data["std"][-3:].mean():.4f}')

# ============================================================
# Plotting
# ============================================================
# Filter out None values
valid_data = {k: v for k, v in all_data.items() if v is not None}

if not valid_data:
    print('Error: no valid data found, cannot plot')
    sys.exit(1)

fig, ax = plt.subplots(figsize=(12, 8))

# Color palette
colors = sns.color_palette("tab10")

# Plot mean line
for i, (exp_name, data) in enumerate(valid_data.items()):
    steps = data['steps']
    mean = data['mean']
    std = data['std']
    color = colors[i % len(colors)]
    sns.lineplot(x=steps, y=mean, ax=ax, color=color, linewidth=2.5, label=exp_name)

    # Fill std band
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

# Labels and title
ax.set_xlabel('Steps', fontsize=20)
ax.set_ylabel('Data Action Rate', fontsize=20)
ax.set_title('D4RL', fontsize=24, pad=15)

# Legend
ax.legend(loc='upper right', frameon=False, fontsize=12)

# Grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Tick formatting
ax.tick_params(axis='both', which='major', labelsize=14)

# Format x-axis with scientific notation if needed
if steps[-1] >= 1e5:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x/1e3:.0f}k' if x < 1e6 else f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
print(f'\nSaved to {output_path}')
