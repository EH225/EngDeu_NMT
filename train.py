#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training - this module contains functions for model training.

Usage:
    train.py --model=<str> [options]

Options:
    -h --help                   Show this screen.
    --model=<str>               The name of the model to be trained
    --embed-size=<int>          The size of the word vec embeddings [default: 512]
    --hidden-size=<int>         The size of the hidden state [default: 512]
    --dropout-rate=<float>      The dropout probability to apply when training [default: 0.3]
    --num-layers=<int>          The number of layers to use in the encoder and decoder [default: 1]
    --src-lang=<str>            The source language to translate from [default: deu]
    --tgt-lang=<str>            The target language to translate into [default: eng]
    --train-set=<int>           Specify which training data set to use 1 to 3 [default: 1]
    --warm-start=<bool>         If set to True, a saved model and optimizer state is used [default: True]
    --pt-embeddings=<bool>      If set to True, cached pre-trained embeddings will be used [default: True]
    --debug=<bool>              If set to True, then run in debug mode [default: False]
"""

import math, time, sys, os
from docopt import docopt
from models.util import Hypothesis, NMT
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from vocab.vocab import Vocab, VocabEntry
from pathlib import Path
import sentencepiece as spm
import models.all_models as all_models
import model_eval
import util
import torch
import torch.nn.utils

DEBUG_TRAIN_PARAMS = {"log_niter": 1, "validation_niter": 3}


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
    grad_clip = params.get("grad_clip", 5) # Gradient clipping threshold
    validation_niter = params.get("validation_niter", 1000) # How often to evaluate on the validation data set
    log_niter = params.get("log_niter", 100) # How often to print training log updates
    patience_lim = params.get("patience_lim", 3) # How many val evals to wait for the model to improve before
    # lowering the learning rate and training again
    max_trial_num = params.get("max_trial_num", 2) # How many times we will lower the learning rate before
    # triggering early stopping i.e. if eval on the validation data and the results aren't better, then the
    # patience counter goes up. trial_num = how many times the patience counter has hit patience_lim which
    # triggers the learning rate to be shrunk
    lr_decay = params.get("lr_decay", 0.5) # Multiplicative factor to use to shrink the learning rate when
    # patience hits the patience limit i.e. when we've evaluated patience limit times without improvement
    max_epochs = params.get("max_epochs", 10) # How many full passes through the training data are allowed
    warm_start = params.get("warm_start", True) # Try to continue off from where we left off last
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Initialize the optimizer for training
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
    prior_best_ppl = model_eval.compute_perplexity(model, dev_data, batch_size=batch_size_val)
    validation_ppl.loc[train_iter] = prior_best_ppl # Record the first validation immediately run

    train_time = start_time = time.time()
    print('Starting maximum likelihood training...')

    while True:
        epoch += 1 # Track how many full passes through the training data are made

        for src_sents, tgt_sents in util.batch_iter(train_data, batch_size=batch_size_train, shuffle=True):
            train_iter += 1 # Track how many batches have been processed in training for logging
            optimizer.zero_grad() # Zero the grad in the optimizer in case any residual
            batch_size = len(src_sents) # The current batch size, could be less if it is the last batch

            example_losses = -model(src_sents, tgt_sents) # (batch_size,) # Compute the loss for each example
            batch_loss = example_losses.sum() # Compute the sum of loss across all batch examples
            loss = batch_loss / batch_size # Normalize by batch size for a standardized loss metric

            loss.backward() # Compute gradients and clip
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step() # Update model parameters using gradient descent

            batch_losses_val = batch_loss.item() # The total loss across all sentence pairs in this batch
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`

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
                msg = (f"  Epoch: {epoch}, iter: {train_iter}, cuml avg loss: {cuml_loss / cuml_pairs:.1f}"
                       f" cuml avg ppl: {np.exp(cuml_loss / cuml_tgt_words):.1f}, "
                       f"cuml examples: {cuml_pairs}"
                       )
                print(msg, file=sys.stderr) # Print an update of all the training examples between evals
                cuml_loss, cuml_tgt_words, cuml_pairs = 0, 0, 0 # Zero out and reset for next eval print
                validation_num += 1 # Incriment the validation set eval counter
                print('Beginning validation...', file=sys.stderr)

                # Compute perplexity score on the dev_data (i.e. the evaluation set)
                dev_ppl = model_eval.compute_perplexity(model, dev_data, batch_size=batch_size_val)
                msg = (f"  Validation iter {validation_num}, validation set ppl {dev_ppl:.3f}, prior best: "
                       f"{prior_best_ppl:.3f}")
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
                            return

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

            if epoch == max_epochs:
                print('Reached maximum number of epochs!', file=sys.stderr)
                return


## TODO: Make this easier to do, create some helper functions for reading in the data set


##############################################################################################################
##############################################################################################################
##############################################################################################################

if __name__ == "__main__":
    msg = ("Please update your installation of PyTorch. You have {torch.__version__}, but you should have "
           "version 1.0.0")
    assert(torch.__version__ >= "1.0.0"), msg

    ### Ingest input kwargs for training
    args = docopt(__doc__)
    model_class = str(args.get("--model", "LSTM_Att")) # Designate which model class to train
    embed_size = int(args.get("--embed-size", 512)) # Specify the word vec embedding size
    hidden_size = int(args.get("--hidden-size", 512)) # Specify the hidden state
    num_layers =  int(args.get("--num-layers", 1)) # Specify how many layers the model has
    dropout =  float(args.get("--dropout-rate", 0.3)) # Specify the dropout rate for training
    src_lang = args.get("--src-lang", "deu") # Specify the source language (from)
    tgt_lang = args.get("--tgt-lang", "eng") # Specify the target language (to)
    warm_start = args.get("--warm-start", "True") == "True" # Whether to try continuing with a prior model
    train_set = int(args.get("--train-set", 1)) # Specify which training set to use {1, 2, 3}
    use_pretreind_embeddings = args.get("--pt-embeddings", "True") == "True"
    assert src_lang != tgt_lang, "soruce language must differ from target language"
    debug = args.get("--debug", "False") == "True" # Specify if the training is to be run in debug mode

    ### Read in the data sets required to train the model
    print(f"Starting training process for {src_lang} to {tgt_lang}. Data set pre-processing...")
    start_time = time.time()
    # Build the data set for training and validation
    data_set_name = f"train_{train_set}" if debug is False else "train_debug"
    train_data_src = util.read_corpus(src_lang, data_set_name, is_tgt=False)
    train_data_tgt = util.read_corpus(tgt_lang, data_set_name, is_tgt=True)
    train_data = list(zip(train_data_src, train_data_tgt))
    print(f"  Training data processed: {time.time() - start_time:.1f}s")
    start_time = time.time() # Reset the timer start for next step

    val_data_src = util.read_corpus(src_lang, "validation", is_tgt=False)
    val_data_tgt = util.read_corpus(tgt_lang, "validation", is_tgt=True)
    dev_data = list(zip(val_data_src, val_data_tgt))
    print(f"  Validation data processed: {time.time() - start_time:.1f}s")

    ### Load in the pre-trained vocav model
    vocab = Vocab.load(f"vocab/{src_lang}_to_{tgt_lang}_vocab")

    train_params = {} if debug is False else DEBUG_TRAIN_PARAMS.copy()

    ### Determine where to save the model during training
    model_save_dir = util.get_model_save_dir(model_class, src_lang, tgt_lang, debug)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True) # Make the save dir if not already there

    if warm_start and "model.bin" in os.listdir(model_save_dir):
        # Load an existing model and continue training from where we left off
        model = getattr(all_models, model_class).load(f"{model_save_dir}/model.bin")
        train_params["warm_start"] = True # Use the prior optimizer saved from training
    else: # Initialize a new model instance to be trained
        model_kwargs = {"embed_size": embed_size, "hidden_size": hidden_size, "num_layers": num_layers,
                        "dropout_rate": dropout, "vocab": vocab}
        model = getattr(all_models, model_class)(**model_kwargs)
        print(f"Instantiating model={model.name} with kwargs:\n", model_kwargs)
        if use_pretreind_embeddings is True:
            try: # Attempt to load the pre-trained embedding weights if possible
                parmas = torch.load(f"saved_models/embeddings/{src_lang}_{embed_size}",
                                    map_location=lambda storage, loc: storage, weights_only=False)
                model.source_embeddings.weight = torch.nn.Parameter(parmas['state_dict']['weight'])

                parmas = torch.load(f"saved_models/embeddings/{tgt_lang}_{embed_size}",
                                    map_location=lambda storage, loc: storage, weights_only=False)
                model.target_embeddings.weight = torch.nn.Parameter(parmas['state_dict']['weight'])
                print("Using pre-trained word embeddings of size: {model.embed_size}")
            except Exception as e:
                print("Could not load pre-trained word embedding weights")
                print(e)

    # Run a training iter for the model
    print("train_params:\n", train_params)
    train_model(model, train_data, dev_data, model_save_dir, params=train_params)
