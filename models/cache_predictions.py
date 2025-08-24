# -*- coding: utf-8 -*-
"""
This module generates cached machinetranslation dataframes (mt_dfs) for each model on all eval data sets i.e.
train_debug, validation, and test. This makes the process of running the evaluation functions faster be
reducing the computational load of the model_eval module i.e. it will be able to read-in the pre-caced model
predictions instead of having to generate them on-the-fly.
"""
import os, sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util
from models import all_models
import model_eval

if __name__ == "__main__":
    # Example usage:  python models/cache_predictions.py --debug=False --beam-search=True
    parser = argparse.ArgumentParser(description='Run model prediction caching')
    parser.add_argument('--debug', type=str, help='Set to True for running in debug testing mode',
                        default="True")
    parser.add_argument('--beam-search', type=str, help='Set to True to run using beam search',
                        default="True")
    args = parser.parse_args()
    debug = args.debug.lower() == "true"
    beam_search = args.beam_search.lower() == "true"

    msg = "Generating predictions using "
    msg = msg + "beam search" if beam_search else msg + "greedy search"
    print(msg) # Report what kind of search method is being uses

    data_set_names = ["train_tiny"] if debug is True else ["train_debug", "validation", "test"]
    kwargs = {"beam_size": 5} if beam_search is True else {"beam_size": 1, "k_pct": 0.1}
    print("kwargs:", kwargs)

    for data_set_name in data_set_names: # Generate predictions for all data sets
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
