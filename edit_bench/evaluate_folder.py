# edit_bench/scripts/evaluate_metrics.py

import os
import argparse
import h5py
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and print metrics from EditBench inference results.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing .h5 result files')
    return parser.parse_args()

def load_metrics_from_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        metrics = f['metrics']
        return {
            'mse': metrics['mse'][()],
            'psnr': metrics['psnr'][()],
            'ssim': metrics['ssim'][()],
            'lpips': metrics['lpips'][()],
            'edit_adherance': metrics['Edit Adherance'][()]
        }

def evaluate_metrics(results_dir):
    all_metrics = {
        'mse': [],
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'edit_adherance': []
    }

    h5_files = [f for f in os.listdir(results_dir) if f.endswith('.h5')]
    if not h5_files:
        print("No .h5 files found in the directory.")
        return

    for file in h5_files:
        path = os.path.join(results_dir, file)
        metrics = load_metrics_from_h5(path)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    print(f"Aggregated metrics over {len(h5_files)} samples:")
    for k, v in all_metrics.items():
        mean_val = np.mean(v)
        print(f"{k.upper():<15}: {mean_val:.4f}")

def main():
    args = parse_args()
    evaluate_metrics(args.results_dir)
