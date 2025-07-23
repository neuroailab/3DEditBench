from abc import ABC, abstractmethod
class ObjectEditingModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_forward(self, image, image_id, point_prompt, R, T, K, gt_segment):
        """
        Perform an edit on the input image using a 3D transformation
        derived from R (rotation) and T (translation).

        Args:
            image (np.ndarray): Input RGB image of shape [H, W, 3], with values in the range [0, 255].
            image_id (str): Unique identifier for the image.
            point_prompt (np.ndarray): 2D point of shape (2,) specifying a pixel location (x, y)
                on the object to be edited. x is horizontal, y is vertical.
            R (np.ndarray): Rotation matrix of shape [3, 3] defining the object transformation.
            T (np.ndarray): Translation vector of shape [3,] defining the object transformation.
            K (np.ndarray): Camera intrinsics matrix of shape [3, 3].
            gt_segment (np.ndarray): Ground truth segmentation mask of shape [H, W], with 1s for object pixels
                and 0s for background. If `use_gt_segment` is False, this can be a custom mask.
        Returns:
            edited_image (np.ndarray): Edited RGB image of shape [H, W, 3], in [0, 255] range.
        """
