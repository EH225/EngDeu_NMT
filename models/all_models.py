# -*- coding: utf-8 -*-
"""
Imports all models into 1 module for quick access and reference.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util

# Whenever we add another model to the library, add it here
from models.Fwd_RNN import Fwd_RNN
from models.LSTM_Att import LSTM_Att
from models.LSTM_AttNN import LSTM_AttNN
from models.Google_API import Google_API
from models.EDTM import EDTM

MODELS = ["Fwd_RNN", "LSTM_Att", "Google_API", "EDTM"] # List the models to be used from this module

def load_model(model_class: str, src_lang: str, tgt_lang: str):
    """
    Helper util function that loads a model from a given model_class for a specified translation language
    pairing i.e. (src_lang, tgt_lang).

    E.g. model = all_models.load_model("LSTM_Att", "deu", "eng")

    Parameters
    ----------
    model_class : str
        The name of the model class to load from e.g. "Fwd_RNN", "LSTM_Att", "Google_API" etc.
    src_lang : str
        The language of the source sentences (e.g. "eng" or "deu").
    tgt_lang : str
        The language of the target sentences (e.g. "eng" or "deu").

    Returns
    -------
    model : Optional[NMT]
        Loads and returns the model instance saved to disk or None if it cannot be located.
    """
    assert src_lang != tgt_lang, f"Source and target language must be different, got {src_lang}, {tgt_lang}"
    translation_name = f"{src_lang.capitalize()}{tgt_lang.capitalize()}"
    model_save_dir = util.get_model_save_dir(model_class, src_lang, tgt_lang, False)

    if os.path.exists(f"{model_save_dir}/model.bin"): # Check if there is a model saved in this dir
        model = globals()[model_class].load(f"{model_save_dir}/model.bin")  # Load the model
    elif model_class == "Google_API": # Handle special case for this model wihch doesn't have saved weights
        model = globals()[model_class](src_lang, tgt_lang)
    else:
        print(f"No {model_class} {translation_name} saved model on disk")
        model = None
    return model
