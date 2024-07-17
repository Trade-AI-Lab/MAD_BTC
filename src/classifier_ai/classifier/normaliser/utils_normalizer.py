import optuna
import sys
import os
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).find("BillionaireSoon")+len("BillionaireSoon")])

from sequencer.normaliser.soft_normaliser import *


NORMALISATION_MINMAX_WINDOW = 1
NORMALISATION_LOG = 2
NORMALISATION_Z_SCORE = 3
NORMALISATION_DIVISION = 4
NORMALISATION_Z_SCORE_LOG = 5
NORMALISATION_Z_SCORE_DIVISION = 6
NORMALISATION_LOG_DIVISION = 7



def normalizer_selector(normalizer: complex_fct_normalizer,
                        normalizer_selected: int = NORMALISATION_Z_SCORE_DIVISION):
    if normalizer_selected == NORMALISATION_MINMAX_WINDOW:
        normalizer = normalizer.min_max_normalization
    elif normalizer_selected == NORMALISATION_LOG:
        normalizer = normalizer.division_relative_normalisation
    elif normalizer_selected == NORMALISATION_Z_SCORE:
        normalizer = normalizer.Z_score_normalization
    elif normalizer_selected == NORMALISATION_DIVISION:
        normalizer = normalizer.division_relative_normalisation
    elif normalizer_selected == NORMALISATION_Z_SCORE_LOG:
        normalizer = normalizer.Z_score_log_normalization
    elif normalizer_selected == NORMALISATION_Z_SCORE_DIVISION:
        normalizer = normalizer.Z_score_division_normalization
    elif normalizer_selected == NORMALISATION_LOG_DIVISION:
        normalizer = normalizer.log_division_normalization
    return normalizer

def normalizer_selecter(trial:optuna.Trial,list_normalisation:list[str]=[NORMALISATION_MINMAX_WINDOW,NORMALISATION_LOG,NORMALISATION_Z_SCORE,
                                                         NORMALISATION_DIVISION,NORMALISATION_Z_SCORE_LOG,NORMALISATION_Z_SCORE_DIVISION,
                                                        NORMALISATION_LOG_DIVISION],name_normalizer:str="price_normalizer"):
    """
    Args:
        trial (optuna.Trial):
        list_normalisation (list[str]):
        name_normalizer (str):

    Returns:

    """
    return trial.suggest_categorical(f'{name_normalizer}_normalization', list_normalisation)