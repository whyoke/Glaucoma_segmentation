import fastai
from fastai.vision import *
from fastai.vision.all import *
from sklearn.metrics import mean_squared_error
import torch
import torchvision.transforms as transforms
import os
import os.path as op
from tqdm.auto import tqdm
from PIL import ImageOps
import PIL
from typing import Tuple, List, Dict, Callable, Union
import subprocess

from imantics import Polygons, Mask

import shutil
import shlex

# import asyncio


class FastaiCupDiscSegmentor:
    """A utility class to get the segmented cup and disc from the image given"""

    def __init__(
        self,
        cup_learner: fastai.learner.Learner,
        disc_learner: fastai.learner.Learner,
        image_size: Tuple[int, int] = (224, 224),
        mask_value_threshold: float = 0.9,
        device: str = "cpu",
    ) -> None:
        self.image_size = image_size
        self.mask_value_threshold = mask_value_threshold
        self.zero_count = 0

        self.cup_learner = cup_learner
        self.disc_learner = disc_learner

        self.cup_learner.model.to(device)
        self.disc_learner.model.to(device)

        self.cup_learner.model.eval()
        self.disc_learner.model.eval()

    def predict_from_dl(self, dataloader: fastai.data.core.TfmdDL) -> (np.ndarray, np.ndarray):
        """Predict cups and discs from a dataloader"""
        cups = array(self.cup_learner.get_preds(dl=dataloader)[0].squeeze())
        discs = array(self.disc_learner.get_preds(dl=dataloader)[0].squeeze())
        return cups, discs

    @torch.no_grad()
    def segment(self, image: Union[str, PIL.Image.Image]) -> (np.ndarray, np.ndarray):
        """Predicts cup and disc masks from the given image / image path"""
        if isinstance(image, (str, Path)):
            image = PILImage.create(image)
        elif isinstance(image, PIL.Image.Image):
            image = image
        else:
            raise TypeError("The input must be a string or PIL.Image.Image")

        # Resize the image to the specified size.
        image = image.resize(self.image_size)
        image = image.convert("RGB")
        image = np.array(image)

        # Predict cup and disc masks.
        with self.cup_learner.no_bar(), self.cup_learner.no_logging():
            cup = self.cup_learner.predict(image)[0].squeeze()
        with self.disc_learner.no_bar(), self.disc_learner.no_logging():
            disc = self.disc_learner.predict(image)[0].squeeze()

        # Convert TensorMasks to numpy arrays.
        cup = cup.numpy()
        disc = disc.numpy()
        return cup, disc

    def predict_cdr(self, image: Union[str, PIL.Image.Image]) -> float:
        """Predicts the cup/disc ratio from the given image / image path"""
        cup, disc = self.segment(image)
        return self.get_cdr(cup, disc)

    def get_cdr(self, cup: np.ndarray, disc: np.ndarray, mask_value_threshold: float = None) -> float:
        """Gets the cup-disc ratio from the given cup and disc masks"""
        # Use default threshold if not specified.
        if mask_value_threshold is None:
            mask_value_threshold = self.mask_value_threshold

        # Find the length of the cup and disc masks.
        l_cup = self.find_mask_length(cup, mask_value_threshold)
        l_disc = self.find_mask_length(disc, mask_value_threshold)

        # Return 0 if either cup or disc is 0.
        if l_cup == 0 or l_disc == 0:
            # Increment the zero count.
            self.zero_count += 1
            return 0.0
        # CDR is the ratio of the cup length to the disc length.
        cdr = l_cup / l_disc
        return cdr

    def find_mask_length(self, mask: np.ndarray, mask_value_threshold: int = None) -> float:
        """
        Finds the center of the interested mask_value_threshold regoin and measure the height
        """
        # Convert to np.array if not already one.
        if not isinstance(mask, np.ndarray):
            mask = ImageOps.equalize(mask)
            mask = np.array(mask)
        # Use default threshold if not specified.
        if not mask_value_threshold:
            mask_value_threshold = self.mask_value_threshold
        # Ignore the color channel.
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        try:
            # Find the rows of the mask (i.e., the region with value >= mask_value_threshold).
            rows, _ = np.where(mask >= mask_value_threshold)
            # Find the center of the masked rows.
            row_center = rows.min() + int((rows.max() - rows.min()) / 2)
            # Find the columns of the mask.
            cols = np.where(mask[row_center, :] >= mask_value_threshold)[0]
            # Find the height of the mask.
            height = cols.max() - cols.min()
            return height
        except Exception as e:
            print(e)
            return 0

    def evaluate_cdr(
        self,
        image_paths: List[Path],
        cup_mask_path: str,
        disc_mask_path: str,
        loss_fn: Callable = mean_squared_error,
    ) -> float:
        """Evaluates the predicted cup/disc ratio against the truth from the given image paths"""
        true_cdr_list = []
        predicted_cdr_list = []
        for image_path in tqdm(image_paths):
            ## get true masks and true cdr
            cup = PILImage.create(op.join(cup_mask_path, image_path.stem + ".jpg"))
            disc = PILImage.create(op.join(disc_mask_path, image_path.stem + ".jpg"))
            true_cdr = self.get_cdr(cup=cup, disc=disc)
            ## get predicted cdr from path
            predicted_cdr = self.predict_cdr(image_path)
            true_cdr_list.append(true_cdr)
            predicted_cdr_list.append(predicted_cdr)
        return loss_fn(predicted_cdr_list, true_cdr_list)

    def get_mask_boundary_points(self, mask: np.ndarray) -> List[Dict[str, int]]:
        polygons = Mask(mask).polygons()
        points = polygons.points[0].tolist()
        # Convert to int to avoid json serialization error.
        return [{"x": int(x), "y": int(y)} for x, y in points]


class YoloCupDiscSegmentor:
    def __init__(
        self,
        cup_model_path: str,
        disc_model_path: str,
        image_size: Tuple[int, int] = (1024, 1024),
        measure_mask_length_from_height: bool = True,
        temp_save_dir: str = "temp_predicted_masks",
        clear_temp_dir_after: bool = True,
        device: str = "cuda",  # unused for now.
    ):
        self.cup_model_path = cup_model_path
        self.disc_model_path = disc_model_path
        self.image_size = image_size

        self.temp_save_dir = temp_save_dir
        self.cup_temp_save_dir = op.join(self.temp_save_dir, "cup")
        self.disc_temp_save_dir = op.join(self.temp_save_dir, "disc")
        self.clear_temp_dir_after = clear_temp_dir_after

        self.measure_mask_length_from_height = measure_mask_length_from_height
        self.device = device

    def predict(self, image_dir: str, via_cli: bool = True) -> dict:
        """Predicts cup and disc masks from the given image path list"""
        if via_cli:
            return self._predict_cli(image_dir)
        else:
            image_path_list = glob(op.join(image_dir, "*"))
            return self._predict(image_path_list)

    def _predict_cli(
        self,
        image_dir: str,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts cup and disc masks from the given image path via CLI.
        Args:
            image_dir (str): The path to the image directory or image file.
        Returns:
            A dictionary of predicted CDR and masks with the image name(s) as the key(s).
            Each key's value is a dictionary with the following keys: "cup_disc_ratio", "cup", "disc".
        Example:
        >>> image_dir = "path/to/image_dir"
        >>> segmentor.predict(image_dir, via_cli=True)
        >>> # Returns the following dictionary:
        >>> {"first_image": {"cup_disc_ratio": 0.5, "cup": {'x': ..., 'y': ...}, "disc": {'x': ..., 'y': ...}, "second_image": {...}}
        """

        # Get image name list if the given image_dir is a directory. Otherwise, wrap the image path in a list.
        image_path_list = op.join(image_dir, "*") if op.isdir(image_dir) else [image_dir]
        image_name_list = os.listdir(image_dir) if op.isdir(image_dir) else [op.basename(image_dir)]

        # Create temporary save directories.
        project_name = self.temp_save_dir
        cup_subproject_name = op.basename(self.cup_temp_save_dir)
        disc_subproject_name = op.basename(self.disc_temp_save_dir)

        cup_command = f"yolo task=segment mode=predict model={self.cup_model_path} source={image_dir} imgsz={self.image_size[0]} half=True save_txt=True project={project_name} name={cup_subproject_name} exist_ok=True save=True"
        disc_command = f"yolo task=segment mode=predict model={self.disc_model_path} source={image_dir} imgsz={self.image_size[0]} half=True save_txt=True project={project_name} name={disc_subproject_name} exist_ok=True save=True"

        # print(*shlex.split(cup_command))
        # print(*shlex.split(disc_command))

        # # Run the commands asynchronously.
        # cup_process = await asyncio.create_subprocess_exec(*shlex.split(cup_command))
        # disc_process = await asyncio.create_subprocess_exec(*shlex.split(disc_command))

        # # Wait for the processes to finish.
        # await cup_process.wait()
        # await disc_process.wait()

        # Run the commands synchronously.
        cup_process = os.system(cup_command)
        disc_process = os.system(disc_command)

        result_mask_dict = {}
        for image_path, image_name in zip(image_path_list, image_name_list):
            original_image_size = Image.open(image_path).size

            # Default predicted values in case the segmentation fails.
            cup_length, disc_length, cdr = "N/A", "N/A", "N/A"
            cup_coordinate_dict, disc_coordinate_dict = {"prediction": "N/A"}, {"prediction": "N/A"}

            # Get the image name and extension.
            image_name, image_extension = op.splitext(image_name)

            # Get the predicted cup and disc mask paths.
            cup_txt_path = op.join(self.cup_temp_save_dir, "labels", image_name + ".txt")
            disc_txt_path = op.join(self.disc_temp_save_dir, "labels", image_name + ".txt")

            # Get the cup and disc mask lengths if the masks exist.
            if op.exists(cup_txt_path):
                cup_length, cup_coordinate_dict = self.get_mask_length_from_textfile(
                    cup_txt_path,
                    return_axis_coordinate_dict=True,
                    original_image_size=original_image_size,
                )
            if op.exists(disc_txt_path):
                disc_length, disc_coordinate_dict = self.get_mask_length_from_textfile(
                    disc_txt_path,
                    return_axis_coordinate_dict=True,
                    original_image_size=original_image_size,
                )
            # Calculate the cup/disc ratio if both the cup and disc masks exist.
            if op.exists(cup_txt_path) and op.exists(disc_txt_path):
                # Calculate the cup/disc ratio.
                cdr = cup_length / disc_length

            # Save the results in a dictionary.
            result_mask_dict[image_name] = {
                "cup_disc_ratio": cdr,
                "cup": cup_coordinate_dict,
                "disc": disc_coordinate_dict,
            }
        # Clear the temporary save directories.
        if self.clear_temp_dir_after:
            self.clear_temp_dir()
        return result_mask_dict

    def get_mask_length_from_textfile(
        self,
        text_file_path: str,
        return_axis_coordinate_dict: bool = True,
        original_image_size: Tuple[int, int] = None,
        ) -> int:
        """
        Returns the length of the predicted mask from the text file.
        Args:
            text_file_path (str): The path to the text file.
            return_axis_coordinate_dict (bool): Whether to return the dictionary containing the x and y coordinates of the mask.
        """
        x_coordinate_list, y_coordinate_list = self.get_axis_coordinates_from_textfile(text_file_path)
        target_axis_coordinate_list = y_coordinate_list if self.measure_mask_length_from_height else x_coordinate_list
        mask_length = self.find_mask_length(target_axis_coordinate_list)
        if return_axis_coordinate_dict:
            xy_coordinate_list = list(zip(x_coordinate_list, y_coordinate_list))
            # Scale the mask coordinates if the original image size is given.
            if original_image_size:
                xy_coordinate_list = self.scale_mask_coordinates(xy_coordinate_list, original_image_size)
            return mask_length, xy_coordinate_list
        return mask_length

    def get_axis_coordinates_from_textfile(
        self, text_file_path: str, separator: str = " ", class_id_is_first_index: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Returns the x and y coordinates of the bounding box from the text file.
        The text file is in the format of [class_id, x1, y1, x2, y2, x3, y3, x4, y4].
        Args:
            text_file_path (str): The path to the text file.
            separator (str): The separator used in the text file.
            class_id_is_first_index (bool): Whether the class_id is the first index in the text file.
        Returns:
            x_coordinate_list (List[float]): The x coordinates of the bounding box.
            y_coordinate_list (List[float]): The y coordinates of the bounding box.
        """
        # Read only the first line of the text file as the text file should contain only one class.
        coordinate_list = open(text_file_path).readline().strip("\n").split(separator)
        # Convert to float.
        coordinate_list = [float(coordinate) for coordinate in coordinate_list]
        # Remove the first element (class_id).
        coordinate_list = coordinate_list[1:]

        # The current list is in the format of [x1, y1, x2, y2, x3, y3, x4, y4]. Separate the x and y coordinates.
        x_coordinate_list = coordinate_list[::2]
        y_coordinate_list = coordinate_list[1::2]
        return x_coordinate_list, y_coordinate_list

    def find_mask_length(self, axis_coordinate_list: List[float]) -> float:
        """
        Finds the maximum distance between the minimum and maximum values of the coordinates.
        Max width is returned if axis_coordinates is the x coordinates.
        Max height is returned if axis_coordinates is the y coordinates.
        """
        maximum_value = min(axis_coordinate_list)
        minimum_value = max(axis_coordinate_list)
        return maximum_value - minimum_value

    def _predict(self, image_path_list: List[str]):
        return NotImplementedError(
            "This method is not yet implemented. YOLOV8's predictions through Python interface still has bugs that need to be fixed."
        )

    def clear_temp_dir(self):
        """Clears the temporary directories."""
        shutil.rmtree(self.temp_save_dir)

    def scale_mask_coordinates(
        self,
        xy_coordinate_list: List[Tuple[float, float]],
        original_image_size: Tuple[int, int],
        ) -> List[Tuple[float, float]]:
        """
        Scales the mask coordinates to the original image size.
        Args:
            xy_coordinate_list (List[Tuple[float, float]]): The list of x and y coordinates of the mask.
            original_image_size (Tuple[int, int]): The original image size.
        """
        original_image_width, original_image_height = original_image_size
        scaled_xy_coordinate_list = []
        for x, y in xy_coordinate_list:
            scaled_x = round(x * original_image_width)
            scaled_y = round(y * original_image_height)
            scaled_xy_coordinate_list.append((scaled_x, scaled_y))
        return scaled_xy_coordinate_list