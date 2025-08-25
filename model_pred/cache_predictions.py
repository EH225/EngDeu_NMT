# -*- coding: utf-8 -*-
"""
This module generates cached machine translation dataframes (mt_df) for each model on all eval data sets i.e.
train_debug, validation, and test. This makes the process of running the evaluation functions faster be
reducing the computational load of the model_eval module i.e. it will be able to read-in the pre-cached model
predictions instead of having to generate them on-the-fly for every model.
"""
import os, sys
import argparse, time
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
    print(msg)  # Report what kind of search method is being uses

    data_set_names = ["train_tiny"] if debug is True else ["train_debug", "validation", "test"]
    kwargs = {"beam_size": 5} if beam_search is True else {"beam_size": 1, "k_pct": 0.1}
    print("kwargs:", kwargs)

    device = util.setup_device(try_gpu=True)  # Train on a GPU if one is available
    print(f"Model predictions will be made using the {device}\n")

    for data_set_name in data_set_names:  # Generate predictions for all data sets
        print(f"\nWorking on data set: {data_set_name}")
        eval_data_dict = model_eval.build_eval_dataset(data_set_name)
        for model_class in all_models.MODELS:  # Generate predictions for all models
            for (src_lang, tgt_lang) in [("deu", "eng"), ("eng", "deu")]:
                lang_pair = f"{src_lang.capitalize()}{tgt_lang.capitalize()}"
                model = all_models.load_model(model_class, src_lang, tgt_lang)  # Load the model
                model = model.to(device)  # Move the model to the designated device before predicting
                start_time = time.time()  # Track how long it takes to make predictions
                mt_df = model_eval.generate_mt_df(model, eval_data_dict[lang_pair], kwargs)
                t = f"{time.time() - start_time:.1f}s"  # Record how long it took in seconds
                # Save the cached predictions
                save_dir = f"model_pred/{lang_pair}/{model_class}/"
                os.makedirs(save_dir, exist_ok=True)  # Ensure this folder exists, create if needed
                mt_df.to_csv(os.path.join(save_dir, f"{data_set_name}.csv"), index=False)
                print(f"Saved mt_df predictions for {model_class} ({t}) - {lang_pair} to: {save_dir}")

""" SAMPLE CONSOLE OUTPUT:

Generating predictions using beam search
kwargs: {'beam_size': 5}
Model predictions will be made using the cpu

Working on data set: train_debug
Saved mt_df predictions for Fwd_RNN (258.3s) - DeuEng to: model_pred/DeuEng/Fwd_RNN/
Saved mt_df predictions for Fwd_RNN (343.6s) - EngDeu to: model_pred/EngDeu/Fwd_RNN/
Saved mt_df predictions for LSTM_Att (416.0s) - DeuEng to: model_pred/DeuEng/LSTM_Att/
Saved mt_df predictions for LSTM_Att (367.7s) - EngDeu to: model_pred/EngDeu/LSTM_Att/
Saved mt_df predictions for EDTM (1725.6s) - DeuEng to: model_pred/DeuEng/EDTM/
Saved mt_df predictions for EDTM (1811.7s) - EngDeu to: model_pred/EngDeu/EDTM/
Saved mt_df predictions for Google_API (2.9s) - DeuEng to: model_pred/DeuEng/Google_API/
Saved mt_df predictions for Google_API (2.6s) - EngDeu to: model_pred/EngDeu/Google_API/

Working on data set: validation
Saved mt_df predictions for Fwd_RNN (343.7s) - DeuEng to: model_pred/DeuEng/Fwd_RNN/
Saved mt_df predictions for Fwd_RNN (303.5s) - EngDeu to: model_pred/EngDeu/Fwd_RNN/
Saved mt_df predictions for LSTM_Att (334.5s) - DeuEng to: model_pred/DeuEng/LSTM_Att/
Saved mt_df predictions for LSTM_Att (323.4s) - EngDeu to: model_pred/EngDeu/LSTM_Att/
Saved mt_df predictions for EDTM (1498.3s) - DeuEng to: model_pred/DeuEng/EDTM/
Saved mt_df predictions for EDTM (1514.5s) - EngDeu to: model_pred/EngDeu/EDTM/
Saved mt_df predictions for Google_API (2.8s) - DeuEng to: model_pred/DeuEng/Google_API/
Saved mt_df predictions for Google_API (2.6s) - EngDeu to: model_pred/EngDeu/Google_API/

Working on data set: test
Saved mt_df predictions for Fwd_RNN (359.5s) - DeuEng to: model_pred/DeuEng/Fwd_RNN/
Saved mt_df predictions for Fwd_RNN (334.6s) - EngDeu to: model_pred/EngDeu/Fwd_RNN/
Saved mt_df predictions for LSTM_Att (376.1s) - DeuEng to: model_pred/DeuEng/LSTM_Att/
Saved mt_df predictions for LSTM_Att (386.3s) - EngDeu to: model_pred/EngDeu/LSTM_Att/
Saved mt_df predictions for EDTM (1510.3s) - DeuEng to: model_pred/DeuEng/EDTM/
Saved mt_df predictions for EDTM (1614.7s) - EngDeu to: model_pred/EngDeu/EDTM/
Saved mt_df predictions for Google_API (2.8s) - DeuEng to: model_pred/DeuEng/Google_API/
Saved mt_df predictions for Google_API (2.6s) - EngDeu to: model_pred/EngDeu/Google_API/
"""
