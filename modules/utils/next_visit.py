from .data_fields import *

def suggest_next_dr_visit(dr_probability: float, machine_type: str):
    """
    Decision tree for Diabetic Retinopathy referral based on the given dr_probability and machine_type.
    """
    if machine_type == MachineType.nidek:

        if dr_probability < 0.2:
            dr_next_visit_suggestion = DRVisitSuggestionType.no_refer.value
        elif 0.2 <= dr_probability < 0.5:
            dr_next_visit_suggestion = DRVisitSuggestionType.very_mild.value
        elif 0.5 <= dr_probability < 0.8:
            dr_next_visit_suggestion = DRVisitSuggestionType.moderate.value
        elif dr_probability >= 0.8:
            dr_next_visit_suggestion = DRVisitSuggestionType.severe.value

    elif machine_type == MachineType.eidon:

        if dr_probability < 0.1:
            dr_next_visit_suggestion = DRVisitSuggestionType.no_refer.value
        elif 0.1 <= dr_probability < 0.5:
            dr_next_visit_suggestion = DRVisitSuggestionType.very_mild.value
        elif 0.5 <= dr_probability < 0.8:
            dr_next_visit_suggestion = DRVisitSuggestionType.moderate.value
        elif dr_probability >= 0.8:
            dr_next_visit_suggestion = DRVisitSuggestionType.severe.value
    return dr_next_visit_suggestion


def suggest_next_glaucoma_visit(
    cup_disc_ratio: float,
) -> str:
    """
    Decision tree for glaucoma referral based on the given cup-disc ratio.
    Cup-disc ratio >= 0.7: 1mo
    Cup-disc ratio < 0.7: None
    """
    if cup_disc_ratio < 0.6:
        glaucoma_referral_decision = GlaucomaVisitSuggestionType.no_refer.value
    elif 0.6 <= cup_disc_ratio < 0.9:
        glaucoma_referral_decision = GlaucomaVisitSuggestionType.moderate.value
    elif cup_disc_ratio >= 0.9:
        glaucoma_referral_decision = GlaucomaVisitSuggestionType.severe.value

    return glaucoma_referral_decision
