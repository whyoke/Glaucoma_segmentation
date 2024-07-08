import os
import sys
import os.path as op
from pathlib import Path
import PIL
from PIL import Image
from typing import Union
from tqdm.auto import tqdm

sys.path.append(os.getcwd())
from modules.segmentor import FastaiCupDiscSegmentor, YoloCupDiscSegmentor
from modules.glaucoma_classifier import GlaucomaClassifier
from modules.quality_classifier import FundusImageQualityClassifier
from modules.dr_classifier import DRClassifier
from modules.utils.get_output_dict import get_dr_output_dict, get_glaucoma_output_dict
from modules.utils.next_visit import suggest_next_glaucoma_visit, suggest_next_dr_visit
from modules.utils.data_fields import MachineType

# import asyncio


class EyeScreener:
    def __init__(
        self,
        image_quality_model_path: str,
        dr_model_path: str,
        cup_model_path: str,
        disc_model_path: str,
        glaucoma_model_path: str,
        device: str = "cuda",
        machine_type: MachineType = MachineType.nidek,
        **kwargs,
    ):
        """
        Args:
            image_quality_model_path (str): Path to the image quality model.
            dr_model_path (str): Path to the DR model.
            cup_model_path (str): Path to the cup model.
            disc_model_path (str): Path to the disc model.
            glaucoma_model_path (str): Path to the glaucoma model.
            device (str): "cuda" or "cpu".
            machine_type (str): "Nidek" or "Eidon".
        """
        self.quality_classifier = FundusImageQualityClassifier(model=image_quality_model_path, device=device)
        self.dr_classifier = DRClassifier(model=dr_model_path, device=device)

        cup_disc_segmentor_kwargs = {
            "measure_mask_length_from_height": kwargs.pop("measure_mask_length_from_height", True),
            "temp_save_dir": kwargs.pop("temp_save_dir", "temp_predicted_masks"),
            "clear_temp_dir_after": kwargs.pop("clear_temp_dir_after", True),
        }

        self.cup_disc_segmentor = YoloCupDiscSegmentor(
            cup_model_path=cup_model_path,
            disc_model_path=disc_model_path,
            device=device,
            **cup_disc_segmentor_kwargs,
        )

        self.glaucoma_classifier = GlaucomaClassifier(glaucoma_model_path, device=device)

        self.output_key_to_classifier_map = {
            "image_quality": self.quality_classifier,
            "dr": self.dr_classifier,
            "glaucoma": self.glaucoma_classifier,
        }

        assert machine_type in [MachineType.nidek, MachineType.eidon], "machine_type must be either 'Nidek' or 'Eidon'."
        self.machine_type = machine_type

    def predict(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_metadata = {"width": image.width, "height": image.height}

        # Classify and store output.
        screener_output_dict = {"image_meta": image_metadata}
        print("Classifying the image...")
        for task, classifier in tqdm(self.output_key_to_classifier_map.items()):
            screener_output_dict[task] = classifier.predict(image)

        # Get suggested next visit for DR.
        screener_output_dict["dr"]["suggested_next_visit"] = suggest_next_dr_visit(
            screener_output_dict["dr"]["probability"]["positive"], self.machine_type
        )

        print("Segmenting the image...")

        # If the segmentor is fastai, then use the get_glaucoma_output_dict function.
        if isinstance(self.cup_disc_segmentor, FastaiCupDiscSegmentor):
            screener_output_dict["glaucoma"].update(
                get_glaucoma_output_dict(
                    image,
                    glaucoma_classifier=self.glaucoma_classifier,
                    cup_disc_segmentor=self.cup_disc_segmentor,
                )
            )

        # If the segmentor is YOLO, then use the YOLOCupDiscSegmentor's predict method.
        elif isinstance(self.cup_disc_segmentor, YoloCupDiscSegmentor):
            prediction_dict = self.cup_disc_segmentor.predict(image_dir=image_path)

            if op.isfile(image_path):
                image_name, image_extension = op.splitext(op.basename(image_path))
                # Update the glaucoma output dict with the cup_disc_segmentor's output.
                screener_output_dict["glaucoma"].update(prediction_dict[image_name])

                # Get suggested next visit for glaucoma.
                screener_output_dict["glaucoma"].update(
                    {
                        "suggested_next_visit": suggest_next_glaucoma_visit(
                            screener_output_dict["glaucoma"]["cup_disc_ratio"]
                        )
                    }
                )

        return screener_output_dict