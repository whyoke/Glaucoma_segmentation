from enum import Enum

class DRPredictionType(str, Enum):
    no_dr = "no_dr"
    dr = "dr"

class ImageQualityPredictionType(str, Enum):
    good = "good"
    acceptable = "acceptable"
    poor = "poor"

class GlaucomaPredictionType(str, Enum):
    non_suspect = "normal"
    suspect = "glaucoma_suspect"


class DRVisitSuggestionType(str, Enum):
    no_refer = "no_refer"
    very_mild = "6mo"
    mild = "4mo"
    moderate = "2mo"
    severe = "2wk"

class GlaucomaVisitSuggestionType(str, Enum):
    no_refer = "no_refer"
    mild = "1yr"
    moderate = "6mo"
    severe = "2wk"

class MachineType(str, Enum):
    nidek = "Nidek"
    eidon = "Eidon"