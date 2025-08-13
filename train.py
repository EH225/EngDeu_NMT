#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training - this module contains functions for model training.

Usage:
    train.py --model=<str> [options]

Options:
    -h --help                   Show this screen.
    --model=<str>               The name of the model to be trained
    --embed-size=<int>          The size of the word vec embeddings [default: 256]
    --hidden-size=<int>         The size of the hidden state [default: 256]
    --dropout-rate=<float>      The dropout probability to apply when training [default: 0.3]
    --num-layers=<int>          The number of layers to use in the encoder and decoder [default: 1]
    --n_heads=<int>             The number of attention heads to use in a transformer model [default: 8]
    --src-lang=<str>            The source language to translate from [default: deu]
    --tgt-lang=<str>            The target language to translate into [default: eng]
    --train-set=<int>           Specify which training data set to use 1 to 3 [default: 1]
    --warm-start=<bool>         If set to True, a saved model and optimizer state is used [default: True]
    --pt-embeddings=<bool>      If set to True, cached pre-trained embeddings will be used [default: True]
    --debug=<bool>              If set to True, then run in debug mode [default: False]
"""

import math, time, sys, os
from docopt import docopt
from models.util import NMT
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from vocab.vocab import Vocab
from pathlib import Path
import models.all_models as all_models
import util
import torch
import torch.nn.utils

DEBUG_TRAIN_PARAMS = {"log_niter": 1, "validation_niter": 3}

### NOTE: This function is copied here from model_eval so that we do not need to import that module to run
### this one. Importing model_eval requires importing bert-score which is not part of the default Google Colab
### venv and takes a long time to install. Avoiding that makes it faster to begin a training run.

def compute_perplexity(model: NMT, eval_data: List[Tuple[List[str]]], batch_size: int = 32) -> float:
    """
    Computes a perplexity score of the model over an evaluation data set (eval_data). Note, this is a walk
    forward metric and is therefore a more generous evaluation metric. I.e. for each roll out step, we give
    the model the prior word from the translation provided and ask it to predict the next y_hat. At each step,
    the decoder model is prompted with the prior words from the "true" translation provided, the y_hats of the
    model are never auto-regressively fed into the model.

    Unlike the other evaluation metrics, this one requires the model as an input.

    Parameters
    ----------
    model : NMT
        A NMT model instance to be evaluated.
    eval_data : List[Tuple[List[str]]]
        A list of (src_sentence, tgt_sentence) tuples containing source and target sentences stored
        as lists of word-tokens.
    batch_size : int, optional
        The batch size to use when iterating over eval_data. The default is 32.

    Returns
    -------
    ppl : float
        A perplexity score of the model evaluated over the eval_data.
    """
    was_training = model.training # Check if the model was previously in training mode, save for later
    model.eval() # Switch the model to evaluation mode, do not track gradients

    cuml_loss = 0.0 # Track the cumulative loss over all the eval_data the model is evaluated on
    cuml_tgt_words = 0.0 # Track how many total target language output words were in the eval_data

    with torch.no_grad():  # no_grad() signals backend to throw away all gradients
        for src_sentences, tgt_sentences in util.batch_iter(eval_data, batch_size, shuffle=True):
            loss = model(src_sentences, tgt_sentences, eps=0).sum() # Compute the forward function i.e. the
            # negative log-likelihood of the output target words according to the model without any smoothing
            cuml_loss += loss.item() # Accumulate the loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sentences)  # omitting leading `<s>`
            cuml_tgt_words += tgt_word_num_to_predict # Count tgt words that are in this batch of eval_data

        ppl = np.exp(cuml_loss / cuml_tgt_words) # Compute the preplexity score of the model

    if was_training: # If the model was training, then set it back to that config before exiting
        model.train()

    return ppl


def setup_device(try_gpu: bool = True):
    """
    Setup the device used by PyTorch. If try_gpu is True, then we will attempt to locate GPU hardware.
    """
    device = torch.device("cpu") # Set to the CPU by default

    if try_gpu is True: # Try looking for a GPU if there is one we can connect to
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")

    return device


def train_model(model: NMT, train_data: List[Tuple[List[str]]], dev_data: List[Tuple[List[str]]],
                model_save_dir: str, params: dict = None) -> None:
    """
    General training routine for training a neural machine translation (NMT) model using gradient descent.
    The params dict can be used to specify various training parameters if specified to override the defaults.

    Parameters
    ----------
    model : NMT
        An instantiated NMT model instance to be trained.
    train_data : List[Tuple[List[str]]]
        A list of parallel sentence tuples where each sentence is itself a list of word tokens.
    dev_data : List[Tuple[List[str]]]
        A list of parallel sentence tuples where each sentence is itself a list of word tokens.
    model_save_dir : str
        A directory to save checkpoints of the model into during training.
    params : dict, optional
        An optional dictionary specifying training parameters such as learning rate and batch size etc.
    """

    #### Training Parameters ####
    params = {} if params is None else params
    batch_size_train = params.get("batch_size_train", 32) # Batch size to use during training
    batch_size_val = params.get("batch_size_val", 64) # Batch size to use during validation eval
    lr = params.get("lr", 5e-3) # Specify the learning rate of the model
    grad_clip = params.get("grad_clip", 2) # Gradient clipping threshold
    validation_niter = params.get("validation_niter", 1000) # How often to evaluate on the validation data set
    log_niter = params.get("log_niter", 100) # How often to print training log updates
    patience_lim = params.get("patience_lim", 3) # How many val evals to wait for the model to improve before
    # lowering the learning rate and training again
    max_trial_num = params.get("max_trial_num", 3) # How many times we will lower the learning rate before
    # triggering early stopping i.e. if eval on the validation data and the results aren't better, then the
    # patience counter goes up. trial_num = how many times the patience counter has hit patience_lim which
    # triggers the learning rate to be shrunk
    lr_decay = params.get("lr_decay", 0.5) # Multiplicative factor to use to shrink the learning rate when
    # patience hits the patience limit i.e. when we've evaluated patience limit times without improvement
    max_epochs = params.get("max_epochs", 10) # How many full passes through the training data are allowed
    warm_start = params.get("warm_start", True) # Try to continue off from where we left off last
    epsilon = params.get("eps", 0.0) # The smoothing epsilon parameter for model.forward
    optimizer_kwargs = params.get("optimizer_kwargs", {}) # If there are additional kwargs for the optimizer
    #### Training Parameters ####

    print(f'Starting {model.name} training', file=sys.stderr)
    model_save_path = os.path.join(model_save_dir, "model.bin") # Model save location
    device = setup_device(try_gpu=True) # Train on a GPU if one is available
    print(f'Model training will use the {device}', file=sys.stderr)

    if device == "cuda": # Only try compiling the model iff training on the GPU
        try: # Try generating a compiled version of the model
            model = torch.compile(model)
            print("NMT model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    model.train() # Set the model to train mode, track gradients
    model = model.to(device) # Move the model to the designated device before training

    # Set up variables to track performance for logging and early stopping during model training
    num_trial = 0 # The number of times we've hit patience == patience_lim
    patience = 0 # The number of validation set evals performed without improvement
    epoch = 0 # The number of full training data set cycles
    train_iter = 0 # The number of training batches that have been processed

    # These variables track the performance of the model during each logging printout interval
    logging_loss = 0 # The cumulative loss between log printouts during training
    logging_tgt_words = 0 # How many target words have been processed between log printouts during training
    logging_pairs = 0 # How many sentence pairs have been processed between log printouts during training

    # These variables track the performance of the model between each validation set evaluation
    cuml_loss = 0 # The cumulative loss across all training examples processed so far
    cuml_tgt_words = 0 # The total number of target words processed so far
    cuml_pairs = 0 # How many total sentence pairs have been processed so far
    validation_num = 0 # The number of validation set evals computed

    training_ppl = pd.Series(dtype=float) # Track the training perplexity measures at each log update
    validation_ppl = pd.Series(dtype=float) # Track the validation perplexity measures at each val update

    # Initialize the optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, **optimizer_kwargs)
    # Restore the optimizer's state from the last time we trained if possiable
    if warm_start is True and 'model.bin.optim' in os.listdir(model_save_dir):
        try:
            optimizer.load_state_dict(torch.load(model_save_path + '.optim', weights_only=True))
            training_ppl = pd.read_csv(os.path.join(model_save_dir, "training_ppl.csv"), index_col=0)
            validation_ppl = pd.read_csv(os.path.join(model_save_dir, "validation_ppl.csv"), index_col=0)
            train_iter = training_ppl.index[-1] # Resume where we left off in the model training history
            print("Using saved optimizer state to continue training")
        except Exception as e:
            print("Failed to load pre-trained optimizer and/or training history")
            print(e)
    else:
        print("Using a newly initialized optimizer to train")

    # Compute a validation set perplexity measure immediate before any training to set a baseline for
    # comparison i.e. if we load a model from disk, don't re-save another unless it does better than the
    # original one saved down. This prevents us from automatically saving a new model instance regardless
    # on the first validation iteration
    prior_best_ppl = compute_perplexity(model, dev_data, batch_size=batch_size_val)
    validation_ppl.loc[train_iter] = prior_best_ppl # Record the first validation immediately run

    train_time = start_time = time.time()
    print('Starting maximum likelihood training...')

    while True:
        epoch += 1 # Track how many full passes through the training data are made

        for src_sents, tgt_sents in util.batch_iter(train_data, batch_size=batch_size_train, shuffle=True):
            train_iter += 1 # Track how many batches have been processed in training for logging
            optimizer.zero_grad() # Zero the grad in the optimizer in case any residual
            batch_size = len(src_sents) # The current batch size, could be less if it is the last batch
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading <s>

            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):  # Use BFloat16
                example_losses = model(src_sents, tgt_sents, epsilon) # (batch_size,)
            batch_loss = example_losses.sum() # Compute the sum of loss across all batch examples
            loss = batch_loss / tgt_words_num_to_predict # Normalize by batch size for a stardard loss metric

            loss.backward() # Compute gradients and clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step() # Update model parameters using gradient descent

            batch_losses_val = batch_loss.item() # The total loss across all sentence pairs in this batch

            logging_loss += batch_losses_val # For computing the loss of obs within this log printout interval
            logging_tgt_words += tgt_words_num_to_predict # Tracks the total number of tgt words predicted
            logging_pairs += batch_size # Tracks the total number of sentence pairs processed

            cuml_loss += batch_losses_val # Tracks the cumulative loss between validation set evals
            cuml_tgt_words += tgt_words_num_to_predict # Tracks the total number of tgt words predicted
            cuml_pairs += batch_size # Tracks the total number of sentence pairs processed

            if train_iter % log_niter == 0: # Print log reports periodically
                msg = (f"  Epoch: {epoch}, iter {train_iter}, avg loss: {logging_loss / logging_pairs:.1f}, "
                       f"avg ppl: {math.exp(logging_loss / logging_tgt_words):.1f}, "
                       f"cuml examples: {cuml_pairs}, speed: "
                       f"{logging_tgt_words / (time.time() - train_time):.1f} words/sec, "
                       f"interval duration: {(time.time() - train_time):.1f}, total time elapsed: "
                       f"{time.time() - start_time:.1f}"
                       )
                training_ppl.loc[train_iter] = math.exp(logging_loss / logging_tgt_words) # Log the perf

                print(msg, file=sys.stderr) # Print a logging update during training to report progress
                train_time = time.time() # Update the internal timer for the next iteration
                logging_loss, logging_tgt_words, logging_pairs = 0, 0, 0 # Zero out, reset for next log print

            # Perform validation data set performance testing periodically
            if train_iter % validation_niter == 0:
                val_time = time.time() # Update the internal timer for reporting the validation time
                msg = (f"  Epoch: {epoch}, iter: {train_iter}, cuml avg loss: {cuml_loss / cuml_pairs:.1f}"
                       f" cuml avg ppl: {np.exp(cuml_loss / cuml_tgt_words):.1f}, "
                       f"cuml examples: {cuml_pairs}"
                       )
                print(msg, file=sys.stderr) # Print an update of all the training examples between evals
                cuml_loss, cuml_tgt_words, cuml_pairs = 0, 0, 0 # Zero out and reset for next eval print
                validation_num += 1 # Incriment the validation set eval counter
                print('Beginning validation...', file=sys.stderr)

                # Compute perplexity score on the dev_data (i.e. the evaluation set)
                dev_ppl = compute_perplexity(model, dev_data, batch_size=batch_size_val)
                msg = (f"  Validation iter {validation_num}, validation set ppl {dev_ppl:.3f}, prior best: "
                       f"{prior_best_ppl:.3f}, validation eval duration: {time.time() - val_time:.1f}")
                print(msg, file=sys.stderr)

                validation_ppl.loc[train_iter] = dev_ppl # Log the performance

                if dev_ppl < prior_best_ppl: # Check if the current model is better than the prior, if so save
                    patience = 0 # Reset the early stopping patience counter, we've found a new best model
                    print(f"  Saving the new best model to [{model_save_path}]", file=sys.stderr)
                    model.save(model_save_path)
                    # Also save the optimizer's state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                    # Also log the performance history over time
                    training_ppl.to_csv(os.path.join(model_save_dir, "training_ppl.csv"), index=True)
                    validation_ppl.to_csv(os.path.join(model_save_dir, "validation_ppl.csv"), index=True)
                    prior_best_ppl = dev_ppl # Update if better

                elif patience < patience_lim: # If things haven't improved, but we're still within the limit
                    patience += 1 # Incriment up the patience counter, when it gets too high without a model
                    # improvement on the validation data set, we'll lower the learning rate
                    print(f'  Hit patience {patience}', file=sys.stderr)

                    if patience == patience_lim: # Once we reach the threshold for the patience counter
                        # we will trigger early stopping if the times we've reached it reaches num_trial
                        num_trial += 1
                        print(f'  Hit {num_trial} trial', file=sys.stderr)
                        if num_trial == max_trial_num: # Once we've already lowered the learning rate n times
                            # don't keep doing it, at some point trigger early stopping
                            print('Early stop!', file=sys.stderr)
                            return None

                        else: # Otherwise, lower the leaning rate and try again to make progress
                            # Decay lr, and restore from previously best checkpoint
                            lr = optimizer.param_groups[0]['lr'] * lr_decay
                            print(f'  Loading prior best model, learning rate lowered to {lr}',
                                  file=sys.stderr)
                            params = torch.load(model_save_path, map_location=lambda storage, loc: storage,
                                                weights_only=True) # Load the prior best model
                            model.load_state_dict(params['state_dict'])
                            model = model.to(device)
                            print('  Restoring parameters of the optimizers', file=sys.stderr)
                            optimizer.load_state_dict(torch.load(model_save_path + '.optim',
                                                                 weights_only=True))

                            # Re-instate the training history to continnue onwards from where we left off
                            training_ppl = pd.read_csv(os.path.join(model_save_dir, "training_ppl.csv"),
                                                       index_col=0)
                            validation_ppl = pd.read_csv(os.path.join(model_save_dir, "validation_ppl.csv"),
                                                         index_col=0)
                            train_iter = training_ppl.index[-1] # Roll back to the last train_iter

                            # Set new the new learning rate (lr)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                            patience = 0 # Reset the patience counter now that lr is lower

                # Add to the train time start value the time it took to run the validation loop so that the
                # next training interval doesn't include the time of running the above validation steps
                train_time += (time.time() - val_time)

            if epoch == max_epochs:
                print('Reached maximum number of epochs!', file=sys.stderr)
                return None


def run_model_training(model_params: Dict = None, train_params: Dict = None):
    """
    This function is a more general version of train_model. It handles instantiating a model and/or reading
    in an existing one in from disk for training. It also handles constructing the data set needed for
    training and validation and provides screen updates.

    Parameters
    ----------
    model_params : Dict, optional
        A dictionary of model and data set config arguments, see the module doc string for details.
    train_params : Dict, optional
        An optional dictionary specifying training parameters such as learning rate and batch size etc.

    Returns
    -------
    None. Loads or instantiates a model, loads in the training and validation data set, calls train_model.

    """
    model_params = {} if model_params is None else model_params
    train_params = {} if train_params is None else train_params.copy()

    # 0). Preliminary argument processing
    model_class = str(model_params.get("model", "LSTM_Att")) # Designate which model class to train
    embed_size = int(model_params.get("embed_size", 256)) # Specify the word vec embedding size
    hidden_size = int(model_params.get("hidden_size", 256)) # Specify the hidden state
    num_layers =  int(model_params.get("num_layers", 1)) # Specify how many layers the model has
    n_heads = int(model_params.get("n_heads", 4)) # Specify how many attention heads to use
    block_size = int(model_params.get("block_size", 500)) # Specify the max token input seq length
    dropout =  float(model_params.get("dropout_rate", 0.3)) # Specify the dropout rate for training
    pos_emb = str(model_params.get("pos_emb", "rope")) # Specify what positional embedding type to use
    src_lang = str(model_params.get("src_lang", "deu")) # Specify the source language (from)
    tgt_lang = str(model_params.get("tgt_lang", "eng")) # Specify the target language (to)
    assert src_lang != tgt_lang, "soruce language must differ from target language"
    assert src_lang in ["eng", "deu"], "src_lang must be either eng or deu"
    assert tgt_lang in ["eng", "deu"], "tgt_lang must be either eng or deu"
    warm_start = model_params.get("warm_start", True) # Whether to try continuing with a prior model
    assert isinstance(warm_start, bool), "warm_start must be a bool if provided"
    train_sets = model_params.get("train_sets", [1]) # Specify which training set to use {1, 2, 3}
    train_sets = [train_sets] if isinstance(train_sets, int) else train_sets
    assert isinstance(train_sets, list), "train_sets must be a list of training sets to use or an int"
    use_pretreind_embeddings = model_params.get("pt_embeddings", True)
    assert isinstance(use_pretreind_embeddings, bool), "pt_embeddings must be a bool if provided"
    debug = model_params.get("debug", False) # Specify if the training is to be run in debug mode
    assert isinstance(debug, bool), "debug must be a bool if provided"

    for train_set in train_sets: # Loop over all the training sets specified and train
        # 1). Read in the data sets required to train the model
        print(f"Starting training process for {src_lang} to {tgt_lang}. Data set pre-processing...")
        start_time = time.time()
        # Build the data set for training and validation
        if isinstance(train_set, int): # Interpret as an integer
            data_set_name = f"train_{train_set}" if debug is False else "train_debug"
        else: # Or as a string if not an int
            data_set_name = train_set
        train_data_src = util.read_corpus(src_lang, data_set_name, is_tgt=False)
        train_data_tgt = util.read_corpus(tgt_lang, data_set_name, is_tgt=True)
        train_data = list(zip(train_data_src, train_data_tgt))
        print(f"  Training data ({data_set_name}) processed: {time.time() - start_time:.1f}s")
        start_time = time.time() # Reset the timer start for next step

        val_data_src = util.read_corpus(src_lang, "validation", is_tgt=False)
        val_data_tgt = util.read_corpus(tgt_lang, "validation", is_tgt=True)
        dev_data = list(zip(val_data_src, val_data_tgt))
        print(f"  Validation data processed: {time.time() - start_time:.1f}s")

        # 2). Load in the pre-trained vocab model
        vocab = Vocab.load(f"{src_lang}_to_{tgt_lang}_vocab")

        # 3). Decide whether to use the debug training parameters or not
        train_params = train_params if debug is False else DEBUG_TRAIN_PARAMS.copy()

        # 4) Determine where to save the model during training
        model_save_dir = util.get_model_save_dir(model_class, src_lang, tgt_lang, debug)
        Path(model_save_dir).mkdir(parents=True, exist_ok=True) # Make the save dir if not already there

        # 5). Either load in an existing model from disk or instantiate a new one to train from scratch
        if warm_start and "model.bin" in os.listdir(model_save_dir):
            # Load an existing model and continue training from where we left off if one exists
            model = getattr(all_models, model_class).load(f"{model_save_dir}/model.bin")
            train_params["warm_start"] = True # Use the prior optimizer saved from prev training
        else: # Otherwise, initialize a new model instance to be trained
            model_kwargs = {"embed_size": embed_size, "hidden_size": hidden_size, "num_layers": num_layers,
                            "dropout_rate": dropout, "n_heads": n_heads, "block_size": block_size,
                            "pos_emb": pos_emb, "vocab": vocab}
            model = getattr(all_models, model_class)(**model_kwargs) # Instantiate a new model
            print(f"Instantiating model={model.name} with kwargs:\n", model_kwargs)
            if use_pretreind_embeddings is True: # If creating a new model, we may use pretrained word embeds
                # We will prefer to use the ones specific to this model class if they exist i.e. look for
                # e.g. saved_models/embeddings/LSTM_Att/eng_256 first if the exist, otherwise we can also
                # default to the more general pre-trained word embeddings e.g. saved_models/embeddings/eng_256
                skip_general_emb_wts = False
                try: # Attempt to load the pre-trained embedding weights from this model class subfolder
                    parmas = torch.load(f"saved_models/embeddings/{model_class}/{src_lang}_{embed_size}",
                                        map_location=lambda storage, loc: storage, weights_only=False)
                    model.source_embeddings.weight = torch.nn.Parameter(parmas['state_dict']['weight'])

                    parmas = torch.load(f"saved_models/embeddings/{model_class}/{tgt_lang}_{embed_size}",
                                        map_location=lambda storage, loc: storage, weights_only=False)
                    model.target_embeddings.weight = torch.nn.Parameter(parmas['state_dict']['weight'])

                    print(f"Using {model_class} pre-trained word embeddings of size: {model.embed_size}")
                    skip_general_emb_wts = True # If successful, then skip the next section
                except Exception as e: # If not able to load in pre-trained word-embeddings, then report it
                    print(f"Could not load {model_class} pre-trained word embedding weights")
                    print(e)

                if skip_general_emb_wts is False: # If not found above, then try the general folder
                    try: # Attempt to load the pre-trained embedding weights if possible
                        parmas = torch.load(f"saved_models/embeddings/{src_lang}_{embed_size}",
                                            map_location=lambda storage, loc: storage, weights_only=False)
                        model.source_embeddings.weight = torch.nn.Parameter(parmas['state_dict']['weight'])

                        parmas = torch.load(f"saved_models/embeddings/{tgt_lang}_{embed_size}",
                                            map_location=lambda storage, loc: storage, weights_only=False)
                        model.target_embeddings.weight = torch.nn.Parameter(parmas['state_dict']['weight'])

                        print(f"Using general pre-trained word embeddings of size: {model.embed_size}")

                    except Exception as e: # If not able to load in pre-trained word-embeddings, then report
                        print("Could not load general pre-trained word embedding weights")
                        print(e)

        # Run a full training iteration for this model
        print("train_params:\n", train_params)  # Report the training parameters that will be used in training
        train_model(model=model, train_data=train_data, dev_data=dev_data, model_save_dir=model_save_dir,
                    params=train_params)


if __name__ == "__main__":
    msg = ("Please update your installation of PyTorch. You have {torch.__version__}, but you should have "
           "version 1.0.0")
    assert(torch.__version__ >= "1.0.0"), msg

    # Ingest input kwargs for training
    args = docopt(__doc__)
    model_params = {}
    for key, val in args.items():
        key = key.replace("--", "").replace("-", "_")
        model_params[key] = val
        if key in ["warm_start", "debug", "pt_embeddings"]:
            model_params[key] = (val == "True") # Convert from str to bool

    run_model_training(model_params) # Run model training using the args passed
