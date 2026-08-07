import argparse
import numpy as np
from tbparse import SummaryReader
from pathlib import Path

# ============================================================
# Command line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Stat mean/std of the last step value for each environment')
parser.add_argument('base_dir', type=str, nargs='?', default='logs',
                    help='Path to the top-level log directory containing environment folders')
parser.add_argument('--output', type=str, default=None,
                    help='Optional CSV file path to save the statistics (default: print only)')
args = parser.parse_args()

base_dir = Path(args.base_dir)

# ============================================================
# Data loading: collect the last-step value of every seed
# ============================================================
def load_final_values(env_dir: Path):
    """Collect the last value of the eval metric from all seed dirs under an env."""
    final_values = []

    for seed_dir in sorted(env_dir.rglob('seed*')):
        if not seed_dir.is_dir():
            continue

        try:
            reader = SummaryReader(str(seed_dir))
            df = reader.scalars
        except Exception as e:
            print(f'  Warning: failed to read {seed_dir}: {e}')
            continue

        multiplier = 1
        if "evaluation/success" in df['tag'].values:
            tag = "evaluation/success"
            multiplier = 100
        elif "evaluation/episode.normalized_return" in df['tag'].values:
            tag = "evaluation/episode.normalized_return"
        else:
            continue

        tag_df = df[df['tag'] == tag].reset_index(drop=True)
        if len(tag_df) > 0:
            final_values.append(tag_df['value'].iloc[-1] * multiplier)

    return final_values

# ============================================================
# Statistics for each environment
# ============================================================
results = []
for env_dir in sorted(base_dir.iterdir()):
    if not env_dir.is_dir():
        continue

    values = load_final_values(env_dir)
    if not values:
        print(f'{env_dir.name}: no data found')
        continue

    values = np.array(values)
    mean = values.mean()
    std = values.std()
    results.append({'env': env_dir.name, 'n_seeds': len(values), 'mean': mean, 'std': std})
    print(f'{env_dir.name}: {mean:.2f} +/- {std:.2f}  (n_seeds={len(values)})')

# ============================================================
# Optional CSV output
# ============================================================
if args.output and results:
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('env,n_seeds,mean,std\n')
        for r in results:
            f.write(f'{r["env"]},{r["n_seeds"]},{r["mean"]:.2f},{r["std"]:.2f}\n')
    print(f'\nSaved to {output_path}')
