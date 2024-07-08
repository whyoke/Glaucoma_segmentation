import os
import sys

sys.path.append(os.getcwd())
from modules.utils.next_visit import suggest_next_dr_visit, suggest_next_glaucoma_visit


def get_dr_output_dict(image, dr_classifier, machine_type: str) -> dict:
    """
    Returns a dictionary with the following keys:
    "probability", "prediction", "suggested_next_visit".
    """
    predicted_dict = dr_classifier.predict(image)
    predicted_probabilities = predicted_dict["probability"]
    dr_prediction = predicted_dict["prediction"]
    dr_probability = predicted_probabilities[1]
    suggested_next_visit = suggest_next_dr_visit(dr_probability, machine_type)

    return {
        "probability": predicted_probabilities,
        "prediction": dr_prediction,
        "suggested_next_visit": suggested_next_visit,
    }


def get_glaucoma_output_dict(image, glaucoma_classifier, cup_disc_segmentor) -> dict:
    """
    Returns a dictionary with the following keys:
    "probability", "prediction", "cup_disc_ratio", "suggested_next_visit", "cup", "disc".
    """
    cup, disc = cup_disc_segmentor.segment(image)
    cup_coordinates = cup_disc_segmentor.get_mask_boundary_points(cup)
    disc_coordinates = cup_disc_segmentor.get_mask_boundary_points(disc)

    cup_disc_ratio = cup_disc_segmentor.get_cdr(cup, disc)
    cup_disc_ratio = round(cup_disc_ratio, 3)

    glaucoma_output_dict = glaucoma_classifier.predict(image)
    predicted_probabilities = glaucoma_output_dict["probability"]
    glaucoma_prediction = glaucoma_output_dict["prediction"]
    glaucoma_probability = predicted_probabilities[1]
    suggested_next_visit = suggest_next_glaucoma_visit(cup_disc_ratio, glaucoma_probability)

    return {
        "probability": predicted_probabilities,
        "prediction": glaucoma_prediction,
        "cup_disc_ratio": cup_disc_ratio,
        "suggested_next_visit": suggested_next_visit,
        "cup": cup_coordinates,
        "disc": disc_coordinates,
    }
