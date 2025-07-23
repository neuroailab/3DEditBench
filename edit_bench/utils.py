import h5py as h5

from PIL import Image, ImageOps
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

from dataclasses import dataclass
import matplotlib.pyplot as plt
import sys
import cv2
import os
from edit_bench.model_wrapper import ModelFactory
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

from PIL import Image
def compute_flow(model, img1: np.ndarray, img2: np.ndarray, device: str = "cuda"):

    """Compute optical flow (SEA‑RAFT) and a simple disparity map.

    Args
    ----
    img1, img2 : numpy H×W×3, RGB in [0,1] or uint8.
    device     : Torch device string.

    Returns
    -------
    flow_rgb   : H×W×3 uint8  – colour wheel visualisation of flow.
    disp_norm  : H×W          – normalised (0‑1) inverse‑magnitude depth proxy.
    """
    # Ensure uint8 BGR for OpenCV/ptlflow helper
    def to_bgr_uint8(x):
        # Accept PIL.Image
        if isinstance(x, Image.Image):
            x = np.array(x)

        # Accept torch.Tensor (C,H,W) or (H,W,C)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim == 3 and x.shape[0] in {1, 3}:  # assume C,H,W
                x = np.transpose(x, (1, 2, 0))

        if x.dtype != np.uint8:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Ensure 3‑channel RGB
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)

        # Convert RGB→BGR for OpenCV
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    cv1, cv2_img = map(to_bgr_uint8, (img1, img2))


    # Prepare inputs
    io_adapter = IOAdapter(model, cv1.shape[:2])
    inputs = io_adapter.prepare_inputs([cv1, cv2_img])
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        preds = model(inputs)

    # Flow tensor: BNCHW → (H,W,2)
    flows = preds["flows"][0, 0].permute(1, 2, 0)#.cpu().numpy()

    # Flow‑RGB visualisation via PTLFlow helper; BNCHW → HWC RGB float32 0‑1
    flow_rgb = flow_utils.flow_to_rgb(preds["flows"])[0, 0].permute(1, 2, 0)#.cpu().numpy()

    return flows


@dataclass
class ImageMetricInput:
    image_pred: Image
    image_gt: Image
    image_first_frame: Image
    pts_0: np.ndarray
    pts_1: np.ndarray
    save_path: str = None


@dataclass
class ImageMetricOutput:
    mse: float
    psnr: float
    ssim: float
    lpips: float
    segment_iou: float
    warning: str


class ImageMetricCalculator:
    def __init__(self, device="cuda", lpips_model_type="alex", eval_resolution=(256, 256)):
        self.device = device
        self.lpips_model_type = lpips_model_type
        self.lpips_model = lpips.LPIPS(net=lpips_model_type).to(device)
        self.eval_resolution = (eval_resolution[1], eval_resolution[0])  # ImageOps.fit use (width, height) format

        model_factory = ModelFactory()

        from segment_anything import SamPredictor, sam_model_registry

        sam_ckpt_path = model_factory.load_ckpt('sam_model.pth', only_path=True)
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path)
        sam.to(device="cuda")
        sam.eval()
        self.segmentation_model = SamPredictor(sam)

        # Instantiate model (small SeaRAFT weights trained on FlyingThings)
        self.model_flow = ptlflow.get_model("dpflow", ckpt_path="sintel").to(device)
        self.model_flow.eval()

    @torch.no_grad()
    def predict_segmentation0_from_rgb0(self, rgb0_numpy_0to1,
                                        prompt={"input_point": [[500, 375]], "input_label": [[1]]}):

        rgb0_numpy = (rgb0_numpy_0to1 * 255).astype(np.uint8)
        self.segmentation_model.set_image(rgb0_numpy)
        input_point = np.array(prompt["input_point"])
        input_label = np.array(prompt["input_label"][0])
        masks, scores, logits = self.segmentation_model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_mask_index = np.argmax(scores)
        segmentation0_numpy = masks[best_mask_index]  # HxW bool mask in numpy
        segmentation0_tensor = torch.tensor(segmentation0_numpy).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return {"segmentation0_tensor": segmentation0_tensor, "segmentation0_numpy": segmentation0_numpy,
                "masks": masks, "scores": scores, "logits": logits}

    def get_segment_from_points(self, image, points):
        '''
        image: [H, W, 3] in (0, 255)
        points: [N, 2]
        '''

        segmentation_prompt = {"input_point": points, "input_label": [[1] * len(points)]}

        segment_dict = self.predict_segmentation0_from_rgb0(image, segmentation_prompt)

        segment_map = segment_dict['segmentation0_numpy']

        return segment_map

    def compute_iou(self, seg1, seg2):
        """
        Compute the Intersection over Union (IoU) between two segmentation maps.

        Args:
            seg1 (np.ndarray): First segmentation map of shape (H, W) with binary values.
            seg2 (np.ndarray): Second segmentation map of shape (H, W) with binary values.

        Returns:
            float: The IoU value.
        """
        if seg1.shape != seg2.shape:
            raise ValueError("Both segmentation maps must have the same shape.")

        # Calculate the intersection and union areas.
        intersection = np.logical_and(seg1, seg2)
        union = np.logical_or(seg1, seg2)

        # Avoid division by zero: if union is empty, we consider IoU to be 1.
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 1.0
        return iou

    def get_iou_score(self, input: ImageMetricInput):

        image_pred = np.array(input.image_pred)
        image_gt = np.array(input.image_gt)
        image_frame0 = np.array(input.image_first_frame)

        pts_0 = input.pts_0
        pts_1 = input.pts_1

        # frame0 = torch.from_numpy(image_frame0).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)
        # frame1 = torch.from_numpy(image_pred).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)


        flow_map_pred = compute_flow(self.model_flow, image_frame0, image_pred, device=self.device).cpu().numpy()

        # flow_map_pred = flow_map_pred[0].permute(1, 2, 0).cpu().numpy()

        segment_gt = self.get_segment_from_points(image_gt, pts_1 / 4)

        segment_gt_frame0 = self.get_segment_from_points(image_frame0, pts_0 / 4)

        pts_0 = pts_0 // 4

        # breakpoint()

        coords_frame1 = pts_0 + flow_map_pred[pts_0[:, 1], pts_0[:, 0]]

        segment_pred_frame1 = self.get_segment_from_points(image_pred, coords_frame1)

        iou = self.compute_iou(segment_gt, segment_pred_frame1)

        self.segment_gt = segment_gt

        self.segment_pred = segment_pred_frame1

        return iou



    def calculate_metrics(self, input: ImageMetricInput) -> ImageMetricOutput:

        if not isinstance(input.image_pred, Image.Image) or not isinstance(input.image_gt, Image.Image):
            #convert to PIL Image
            image_pred = Image.fromarray(input.image_pred)
            image_gt =  input.image_gt
            # image_gt = Image.fromarray(input.image_gt)
        else:
            image_pred = input.image_pred
            image_gt = input.image_gt
        # breakpoint()
        image_pred = image_pred.convert("RGB")
        image_gt = image_gt.convert("RGB")
        if input.save_path is not None:
            image_pred.save(input.save_path.replace(".png", "_pred.png"))
            image_gt.save(input.save_path.replace(".png", "_gt.png"))

        image_pred = ImageOps.fit(image_pred, self.eval_resolution)
        image_gt = ImageOps.fit(image_gt, self.eval_resolution)
        if input.save_path is not None:
            image_pred.save(input.save_path.replace(".png", "_pred_metric.png"))
            image_gt.save(input.save_path.replace(".png", "_gt_metric.png"))

        mse = self.calculate_mse(image_pred, image_gt)
        psnr = self.calculate_psnr(image_pred, image_gt)
        ssim = self.calculate_ssim(image_pred, image_gt)
        lpips = self.calculate_lpips(image_pred, image_gt)

        segment_iou = self.get_iou_score(input)

        warning = f"You are using {self.lpips_model_type} model for LPIPS calculation."

        return ImageMetricOutput(mse, psnr, ssim, lpips, segment_iou, warning)

    def calculate_mse(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_numpy_float = np.array(image_pred) / 255
        image_gt_numpy_float = np.array(image_gt) / 255
        return np.mean((image_pred_numpy_float - image_gt_numpy_float) ** 2)

    def calculate_psnr(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_numpy_float = np.array(image_pred) / 255
        image_gt_numpy_float = np.array(image_gt) / 255
        mse = np.mean((image_pred_numpy_float - image_gt_numpy_float) ** 2)
        psnr = 10 * np.log10(1 / mse) if mse > 0 else 100
        return psnr

    def calculate_ssim(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_numpy_float = np.array(image_pred) / 255
        image_gt_numpy_float = np.array(image_gt) / 255
        return compute_ssim(image_pred_numpy_float, image_gt_numpy_float, channel_axis=2, data_range=1)

    def calculate_lpips(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_tensor = self.image_to_tensor(image_pred)
        image_gt_tensor = self.image_to_tensor(image_gt)
        image_pred_tensor_m1_1 = image_pred_tensor * 2 - 1
        image_gt_tensor_m1_1 = image_gt_tensor * 2 - 1
        return self.lpips_model(image_pred_tensor_m1_1, image_gt_tensor_m1_1).item()

    def image_to_tensor(self, image: Image) -> torch.Tensor:
        image_numpy = np.array(image)
        image_tensor = torch.tensor(image_numpy).permute(2, 0, 1).unsqueeze(0).float() / 255
        return image_tensor.to(self.device)


def save_metrics_in_h5(h5_path, image0_downsampled, image1_downsampled, rgb1_pred, pts_0, pts_1, metrics):

    with h5.File(h5_path, 'w') as f:
        f.create_dataset('image0', data=image0_downsampled)
        f.create_dataset('image1', data=image1_downsampled)
        f.create_dataset('edited_image', data=rgb1_pred)
        f.create_dataset('pts_0', data=pts_0)
        f.create_dataset('pts_1', data=pts_1)
        # save metrics in new field
        new_field = f.create_group('metrics')
        new_field.create_dataset('mse', data=metrics.mse)
        new_field.create_dataset('psnr', data=metrics.psnr)
        new_field.create_dataset('ssim', data=metrics.ssim)
        new_field.create_dataset('lpips', data=metrics.lpips)
        new_field.create_dataset('Edit Adherance', data=metrics.segment_iou)


def load_data(h5_file, return_orig=False):
    with h5.File(h5_file, 'r') as f:
        image0 = f['image1'][:]

        image1 = f['image2'][:]

        pts_0_orig = f['points_image1'][:]
        pts_0 = pts_0_orig[:-6].reshape(-1, 4, 2)

        pts_1_orig = f['points_image2'][:]
        pts_1 = pts_1_orig[:-6].reshape(-1, 4, 2)

        K = f['K'][:]

    if return_orig:
        return image0, image1, pts_0, pts_1, K, pts_0_orig, pts_1_orig

    return image0, image1, pts_0, pts_1, K,


def visualize_segmentation_overlay(image, points, calculator, color=[255, 0, 0], alpha=0.5):
    """
    Create a visualization with segmentation overlay using the first point as SAM prompt.
    
    Args:
        image: PIL Image or numpy array [H, W, 3] in range [0, 255]
        points: numpy array of points [N, 2] 
        calculator: ImageMetricCalculator instance with loaded SAM model
        color: RGB color for the overlay [R, G, B] in range [0, 255]
        alpha: transparency of the overlay (0.0 = transparent, 1.0 = opaque)
    
    Returns:
        PIL Image with segmentation overlay
    """
    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Ensure image is in correct format
    if image_np.dtype != np.uint8:
        image_np = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Use first point as SAM prompt
    first_point = points[0:1]  # Take first point, keep 2D shape [1, 2]
    
    # Get segmentation mask using existing functionality
    segmentation_prompt = {"input_point": first_point, "input_label": [[1]]}
    segment_dict = calculator.predict_segmentation0_from_rgb0(image_np / 255.0, segmentation_prompt)
    segment_mask = segment_dict['segmentation0_numpy']  # boolean mask [H, W]
    
    # Create colored overlay
    overlay = image_np.copy()
    overlay[segment_mask] = overlay[segment_mask] * (1 - alpha) + np.array(color) * alpha
    overlay = overlay.astype(np.uint8)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(overlay)
    
    return result_image, segment_mask


def quick_segmentation_visualization_example():
    """
    Quick example to test segmentation visualization functionality.
    """
    print("Creating segmentation visualization example...")
    
    # Initialize the calculator
    calculator = ImageMetricCalculator(device="cuda", eval_resolution=(512, 512))
    
    # Create a simple example - you can replace this with real data
    # For now, create a simple colored image for testing
    example_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    example_points = np.array([[128, 128], [100, 100], [150, 150]])  # Some example points
    
    # Create visualization
    result_image, segment_mask = visualize_segmentation_overlay(
        example_image, 
        example_points, 
        calculator,
        color=[255, 0, 0],  # Red overlay
        alpha=0.3
    )
    
    # Save results
    os.makedirs("visualization_output", exist_ok=True)
    
    # Save original image
    Image.fromarray(example_image).save("visualization_output/original.png")
    
    # Save segmentation mask
    mask_visual = (segment_mask * 255).astype(np.uint8)
    Image.fromarray(mask_visual).save("visualization_output/segmentation_mask.png")
    
    # Save overlay result
    result_image.save("visualization_output/segmented_overlay.png")
    
    print("Visualization saved to visualization_output/ directory")
    print(f"Segmentation mask shape: {segment_mask.shape}")
    print(f"Segmented pixels: {np.sum(segment_mask)} / {segment_mask.size}")
    
    return result_image, segment_mask