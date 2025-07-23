# edit_bench/inference/run_inference.py

import os
import glob
import h5py
import numpy as np
import importlib
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize

from edit_bench.utils import (
    ImageMetricInput,
    ImageMetricCalculator,
    save_metrics_in_h5
)

def load_model(model_class_str):
    module_name, class_name = model_class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    model = model_class()
    return model

def resize_image(image, size=(256, 256)):
    return (resize(image, size, preserve_range=True).astype(np.uint8)
            if image.dtype == np.uint8 else
            resize(image, size).astype(np.float32))

def save_visualization(image0, image_pred, image1, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Input (image0)', 'Prediction', 'Ground Truth (image1)']
    images = [image0, image_pred, image1]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.05)  # reduce horizontal gap between plots
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()


def run_edit_and_evaluate(model, f, image_key, h5_save_path, viz_save_path):
    image0 = f['image1'][:]
    image1 = f['image2'][:]
    R = f['R'][:]
    T = f['T'][:]
    K = f['K'][:]
    point_prompt = f['point_prompt'][:]
    pts_0 = f['object_points_image1'][:]
    pts_1 = f['object_points_image2'][:]
    gt_segment = f['GT_segment'][:]

    image_pred = model.run_forward(
        image0, image_key, point_prompt, R, T, K, gt_segment
    )

    image0_down = resize_image(image0, size=(256, 256))
    image1_down = resize_image(image1, size=(256, 256))
    pred_down = resize_image(image_pred, size=(256, 256))

    pts_0 = pts_0[0] if pts_0.ndim == 3 else pts_0
    pts_1 = pts_1[0] if pts_1.ndim == 3 else pts_1

    metric_input = ImageMetricInput(Image.fromarray(pred_down), Image.fromarray(image1_down),
                                    Image.fromarray(image0_down), pts_0, pts_1)
    metric_calc = ImageMetricCalculator()
    metrics = metric_calc.calculate_metrics(metric_input)

    save_metrics_in_h5(h5_save_path, image0_down, image1_down, pred_down, pts_0, pts_1, metrics)
    save_visualization(image0_down / 255.0, pred_down / 255.0, image1_down / 255.0, viz_save_path)

def run_inference(model_class, editbench_dir, image_keys, output_dir, gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs(output_dir, exist_ok=True)
    h5_dir = os.path.join(output_dir, 'hdf5_result_files')
    viz_dir = os.path.join(output_dir, 'viz')
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    model = load_model(model_class)

    for h5_path in tqdm(glob.glob(os.path.join(editbench_dir, "*.hdf5"))):
        image_key = os.path.basename(h5_path)
        if image_key not in image_keys:
            continue

        with h5py.File(h5_path, "r") as f:
            base_name = os.path.splitext(image_key)[0]
            h5_out_path = os.path.join(h5_dir, base_name + ".h5")
            viz_out_path = os.path.join(viz_dir, base_name + ".png")
            run_edit_and_evaluate(model, f, image_key, h5_out_path, viz_out_path)


# Optional CLI support
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, required=True)
    parser.add_argument("--editbench_dir", type=str, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    run_inference(args.model_class, args.editbench_dir, args.image_keys, args.output_dir, args.gpu)
