import random
import torch

from .utils.data_fields import GlaucomaPredictionType

class GlaucomaClassifier:
    """Dummy Glaucoma Classifier for testing purposes."""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = device

    @torch.no_grad()
    def predict(self, test_file):
        # Dummy probabilities.
        positive_prob = 0.01
        negative_prob = 1 - positive_prob
        probs = [negative_prob, positive_prob]
        # Round probabilities to 2 decimal places.
        probs = [round(p, 2) for p in probs]
        # Dummy prediction from probability of the positive class.
        glaucoma_prediction = self.classify_glaucoma_from_prob(probs[1])
        predicted_probability_dict = {
            "negative": probs[0],
            "positive": probs[1],
        }
        return {"probability": predicted_probability_dict, "prediction": glaucoma_prediction}

    def classify_glaucoma_from_prob(self, prob):
        return GlaucomaPredictionType.suspect.value if prob > 0.5 else GlaucomaPredictionType.non_suspect.value
