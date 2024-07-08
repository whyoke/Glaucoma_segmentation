import os
import os.path as op
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Optional, Tuple
import itertools

DEFAULT_MASK_COLOR_DICT = {
    "DISC": (255, 255, 255),
    "CUP": (255, 255, 255),
}
DEFAULT_MASK_BITNESS = 24
DEFAULT_SCALE_FACTOR = 1.0


def create_mask(
    image_path: str,
    annotation_dictionary: List[dict],
    mask_bitness: int = DEFAULT_MASK_BITNESS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    mask_color: Optional[Tuple[int, int, int]] = None,
    return_points: bool = False,
) -> np.ndarray:
    """A function to create mask from annotation dictionary."""

    current_image = cv2.imread(image_path)
    height, width, _ = current_image.shape
    background = np.zeros((height, width, 3), np.uint8)  # background color

    annotation_type = annotation_dictionary["type"]

    mask = np.full((height, width, mask_bitness // 8), background, dtype=np.uint8)
    if not mask_color:
        mask_color = MASK_COLOR_DICT[annotation_dictionary["label"]]

    if annotation_type == "polyline":
        points = [tuple(map(float, p)) for p in annotation_dictionary["points"]]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points * scale_factor
        points = points.astype(int)
        if return_points:
            points = points.astype(float)
            points[:, 0] = points[:, 0] / annotation_dictionary["width"]
            points[:, 1] = points[:, 1] / annotation_dictionary["height"]
            return points
        else:
            mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
            mask = cv2.fillPoly(mask, [points], color=mask_color)

    elif annotation_type == "ellipse":
        center_coordinates = (eval(annotation_dictionary["cx"]), eval(annotation_dictionary["cy"]))
        axesLength = (eval(annotation_dictionary["rx"]), eval(annotation_dictionary["ry"]))
        ellipse_float = (center_coordinates, axesLength, 0.0)
        if return_points:
            ellipse = (center_coordinates, axesLength)
            points = convert_ellipse_to_polygon_points(ellipse)
            points = points.astype(float)
            points[:, 0] = points[:, 0] / annotation_dictionary["width"]
            points[:, 1] = points[:, 1] / annotation_dictionary["height"]
            return points
        else:
            mask = cv2.ellipse(mask, ellipse_float, color=mask_color, thickness=-1)
    return mask


def generate_masks_from_dataframe(
    dataframe: pd.DataFrame,
    output_mask_dir: str,
    filename_to_annotation_dictionary: dict,
    mask_bitness: int = DEFAULT_MASK_BITNESS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    mask_color: int = None,
    filename_column: str = "filename",
    path_column: str = "path",
):
    os.makedirs(output_mask_dir, exist_ok=True)
    num_without_mask = 0
    for index, row in tqdm(dataframe.iterrows()):
        filename = row[filename_column]
        path = row[path_column]
        segment_annotation_dict = filename_to_annotation_dictionary[filename]

        try:
            mask = create_mask(
                image_path=path,
                annotation_dictionary=segment_annotation_dict,
                mask_bitness=mask_bitness,
                scale_factor=scale_factor,
                mask_color=mask_color,
            )
            if mask is not None:
                # Use absolute path to avoid error when saving.
                cv2.imwrite(op.abspath(op.join(output_mask_dir, filename)), mask)
        except Exception as e:
            print(f"Error in {filename}")
            print(e)
            num_without_mask += 1
            continue

    print(f"{num_without_mask} images without mask")


def convert_ellipse_to_polygon_points(
    ellipse_params: Tuple[Tuple[int, int], Tuple[int, int], float],
    angle_step: int = 10,
) -> np.ndarray:
    """
    Converts ellipse parameters (center coordinates & axes_lengths) to polygon points.
    """
    center_coordinates, axes_lengths = ellipse_params
    x_half_axis, y_half_axis = axes_lengths
    polygon_points = []
    for angle in range(0, 360, angle_step):
        x = x_half_axis * np.cos(np.deg2rad(angle)) + center_coordinates[0]
        y = y_half_axis * np.sin(np.deg2rad(angle)) + center_coordinates[1]
        polygon_points.append((int(x), int(y)))
    return np.array(polygon_points)


def save_polygon_points_to_yolo_label_format(polygon_points: np.ndarray, class_id: int = 0, save_path="temp.txt"):
    """Saves polygon points to YOLO label format as a text file."""
    polygon_point_list = list(itertools.chain(*list(polygon_points)))
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(class_id) + " " + " ".join([str(p) for p in polygon_point_list]))
