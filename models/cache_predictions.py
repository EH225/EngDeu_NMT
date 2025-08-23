# -*- coding: utf-8 -*-
"""
This module generates cached machinetranslation dataframes (mt_dfs) for each model on all eval data sets i.e.
train_debug, validation, and test. This makes the process of running the evaluation functions faster be
reducing the computational load of the model_eval module i.e. it will be able to read-in the pre-caced model
predictions instead of having to generate them on-the-fly.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util
from models import all_models
import model_eval

if __name__ == "__main__":

    kwargs = {"beam_size": 5} # Cache beam-search predictions from the models

    ## TODO: Edit the data_set_name list below when we roll out beam search to all models and want to run in prod
    for data_set_name in ["train_tiny"]: # ["train_debug", "validation", "test"]: # Generate predictions for all data sets
        eval_data_dict = model_eval.build_eval_dataset(data_set_name)
        for model_class in all_models.MODELS: # Generate predictions for all models
            for (src_lang, tgt_lang) in [("deu", "eng"), ("eng", "deu")]:
                lang_pair = f"{src_lang.capitalize()}{tgt_lang.capitalize()}"
                model = all_models.load_model(model_class, src_lang, tgt_lang) # Load the model
                mt_df = model_eval.generate_mt_df(model, eval_data_dict[lang_pair], kwargs)
                # Save the cached predictions
                save_dir = f"model_pred/{lang_pair}/{model_class}/"
                os.makedirs(save_dir, exist_ok=True) # Ensure this folder exists, create if needed
                mt_df.to_csv(os.path.join(save_dir, f"{data_set_name}.csv"), index=False)
                print(f"Saved mt_df predictions for {model_class} - {lang_pair} to: {save_dir}")
