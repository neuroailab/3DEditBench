import argparse
import os
import subprocess
import numpy as np
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel launcher for EditBench inference.")
    parser.add_argument("--gpus", nargs="+", type=int, required=True, help="List of GPU IDs to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--model_class", type=str, required=True, help="Full import path to model class.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Directory containing EditBench .hdf5 files.")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of node splits (for SLURM-style sharding).")
    parser.add_argument("--split_num", type=int, default=0, help="Index of this node's split.")
    return parser.parse_args()

def chunk_list(lst, n):
    return np.array_split(lst, n)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all hdf5 files in dataset_path
    all_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(args.dataset_path, "*.hdf5"))])
    if not all_files:
        raise RuntimeError(f"No .hdf5 files found in {args.dataset_path}")

    print(f"Found {len(all_files)} files in dataset.")

    # Split dataset among num_splits
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_files = chunk_list(all_files, args.num_splits)[args.split_num]

    if len(split_files) == 0:
        print("No files in this split.")
        return

    # Distribute image keys across GPUs
    gpu_chunks = chunk_list(split_files, len(args.gpus))
    processes = []

    for gpu_id, chunk in zip(args.gpus, gpu_chunks):
        if len(chunk) == 0:
            continue

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "editbench-infer",
            "--model_class", args.model_class,
            "--editbench_dir", args.dataset_path,
            "--output_dir", args.output_dir,
            "--gpu", str(gpu_id),
            "--image_keys", *chunk
        ]

        full_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        print(f"Launching inference on GPU {gpu_id} with {len(chunk)} files...")
        p = subprocess.Popen(full_cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()
