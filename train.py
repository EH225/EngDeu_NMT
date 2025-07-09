#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training module

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
    --warm-start=<bool>         If set to True, a saved model and optimizer state is used [default: True]
    --debug=<bool>              If set to True, then run in debug mode [default: False]
"""

import math, time, sys, os
from docopt import docopt
import sacrebleu
from models.util import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from vocab.vocab import Vocab, VocabEntry
from pathlib import Path
import sentencepiece as spm
import pandas as pd
import models.all_models as all_models

import torch
import torch.nn.utils


def eval_perplexity(model: NMT, dev_data: List[Tuple[List[str]]], batch_size: int = 32) -> float:
    """
    Computes a perplexity score of the model over an evaluation data set (dev_data).

    Parameters
    ----------
    model : TYPE
        A NMT model instance.
    dev_data : List[Tuple[List[str]]]
        A list of (src_sentence, tgt_sentence) list of tuples containing source and target sentences.
    batch_size : int, optional
        The batch size to use when iterating over dev_data. The default is 32.

    Returns
    -------
    ppl : float
        A perplexity score of the model evaluated over the dev_data.
    """
    was_training = model.training # Check if the model was previously in training mode, save for later
    model.eval() # Switch the model to evaluation mode, do not track gradients

    cuml_loss = 0.0 # Track the cumulative loss over all the dev_data the model is evaluated on
    cuml_tgt_words = 0.0 # Track how many total target language output words were in the dev_data

    with torch.no_grad():  # no_grad() signals backend to throw away all gradients
        for src_sentences, tgt_sentences in batch_iter(dev_data, batch_size):
            loss = -model(src_sentences, tgt_sentences).sum() # Compute the forward function i.e. the
            # negative log-likelihood of the output target words according to the model
            cuml_loss += loss.item() # Accumulate the loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sentences)  # omitting leading `<s>`
            cuml_tgt_words += tgt_word_num_to_predict # Count how many tgt words are in this batch of dev_data

        ppl = np.exp(cuml_loss / cuml_tgt_words) # Compute the preplexity score of the model

    if was_training: # If the model was training, then set it back to that config before exiting
        model.train()

    return ppl


def compute_corpus_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Computes a corpus-level BLEU score given a set of references (gold-standard translations) and hypotheses
    i.e. translation outputs from the model. Compares how similar they are to one another as a measure of
    model performance.

    The number of references and hypotheses should be equal. The references are translations of input source
    sentences that are considered to be as good as one can get in terms of translation quality. This function
    compares them vs a paired output translation from a model for the same input source sentence.

    Parameters
    ----------
    references : List[List[str]]
        A list of gold-standard translations i.e. a list sentences (which are lists of words).
    hypotheses : List[Hypothesis]
        A list of hypotheses, one for each reference translation.

    Returns
    -------
    float
        A corpus-level BLEU score.
    """
    if references[0][0] == '<s>': # Remove the start and end tokens from the references if present
        references = [ref[1:-1] for ref in references]

    # Detokenize the subword pieces to get full sentences
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])
    return bleu.score


def read_corpus(lang: str, subset: str, is_tgt: bool) -> List[List[str]]:
    """
    Reads in a text corpus file from disk specified by file_path. This function is primarily used for
    creating training, validation, and testing data sets. The contents of the given file are read in and
    tokenized by a pre-trained tokenizer model. This function returns a list of lists containing word tokens.

    Set is_tgt = True if the data set being read in is to be used as a target data set. If so, then all the
    sentences have a <s> start sentence token appeneded to the front and a </s> end sentence token appended
    to the end.

    Parameters
    ----------
    lang : str
        The language to read in data for i.e. either "eng" or "deu".
    subset : str
        The data subset to read in i.e. one of the following ["train", "validation", "test"]
    is_tgt : bool
        A bool flag indicating if the data set is to be used as a target data set.
    vocab_size : int, optional
        The . The default is 2500.

    Returns
    -------
    List[List[str]]
        List of lists where each list is a collection of word tokens.
    """
    tokenized_sentences = []
    sp = spm.SentencePieceProcessor() # Instantiate the tokenizer model
    sp.load(f"vocab/{lang}/{lang}.model") # Load in the pre-trained weights
    file_path = f"dataset/{lang}/{subset}.csv"
    sentences = pd.read_csv(file_path) # Read in the entire data set using pandas
    tokenized_sentences = sentences[sentences.columns[0]].astype(str).apply(lambda x: sp.encode_as_pieces(x))
    tokenized_sentences = list(tokenized_sentences.values) # Convert to a list of lists

    if is_tgt is True: # Only append <s> and </s> tokens if this is a target data set
        for s in tokenized_sentences:
            s.insert(0, "<s>")
            s.append('</s>')

    return tokenized_sentences


def batch_iter(data: List[Tuple[List[str]]], batch_size: int, shuffle: bool = False):
    """
    Generator that yields batches of (source, target) sentences in sorted order by length (largest to
    smallest) until the entire data set is returned in batches.

    Parameters
    ----------
    data : List[Tuple[List[str]]]
        A list of paired (source, target) sentence tuples.
    batch_size : int
        The number of paired sentences per batch.
    shuffle : bool, optional
        Whether to randomly shuffle the dataset. The default is False.

    Yields
    ------
    src_sentences : List[List[str]]
        A list of source language sentences.
    tgt_sentences : List[List[str]]
        A list of target language sentences.

    """
    n_batches = math.ceil(len(data) / batch_size) # How many total batches to iter over the whole data set
    index_array = list(range(len(data)))

    if shuffle is True: # Shuffle the ordering of the sentence pairs before batching
        np.random.shuffle(index_array)

    for i in range(n_batches): # Iterate over how many batches are required to cover the whole data set
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sentences = [e[0] for e in examples]
        tgt_sentences = [e[1] for e in examples]

        yield (src_sentences, tgt_sentences)


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

DEBUG_TRAIN_PARAMS = {"log_niter": 1, "validation_niter": 10, "uniform_init": 0.1}

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
    validation_niter = params.get("validation_niter", 300) # How often to evaluate on the validation data set
    log_niter = params.get("log_niter", 50) # How often to print training log updates
    patience_lim = params.get("patience_lim", 3) # How many val evals to wait for the model to improve before
    # lowering the learning rate and training again
    max_trial_num = params.get("max_trial_num", 3) # How many times we will lower the learning rate before
    # triggering early stopping i.e. if eval on the validation data and the results aren't better, then the
    # patience counter goes up. trial_num = how many times the patience counter has hit patience_lim which
    # triggers the learning rate to be shrunk
    lr_decay = params.get("lr_decay", 0.5) # Multiplicative factor to use to shrink the learning rate when
    # patience hits the patience limit i.e. when we've evaluated patience limit times without improvement
    max_epochs = params.get("max_epochs", 10) # How many full passes through the training data are allowed
    uniform_init = params.get("uniform_init", 0) # How to initialize the parameters if specified, default to
    # which means no initialization
    load_optimizer = params.get("load_optimizer", True) # Try to continue off from where we left off last
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

    if uniform_init > 0.0: # Initialize the model parameters uniformly
        print('Uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    model = model.to(device) # Move the model to the designated device before training

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Initialize the optimizer for training
    # Restore the optimizer's state from the last time we trained if possiable
    if load_optimizer is True and 'model.bin.optim' in os.listdir(model_save_dir):
        print("Using saved optimizer state to continue training")
        optimizer.load_state_dict(torch.load(model_save_path + '.optim', weights_only=True))

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

    train_time = start_time = time.time()
    print('Starting maximum likelihood training...')

    # Compute a validation set perplexity measure immediate before any training to set a baseline for
    # comparison i.e. if we load a model from disk, don't re-save another unless it does better than the
    # original one saved down. This prevents us from automatically saving a new model instance regardless
    # on the first validation iteration
    prior_best_ppl = eval_perplexity(model, dev_data, batch_size=batch_size_val)
    training_ppl = pd.Series(dtype=float) # Track the training perplexity measures at each log update
    validation_ppl = pd.Series(dtype=float) # Track the validation perplexity measures at each val update

    while True:
        epoch += 1 # Track how many full passes through the training data are made

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=batch_size_train, shuffle=True):
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
                dev_ppl = eval_perplexity(model, dev_data, batch_size=batch_size_val)
                msg = (f"  Validation iter {validation_num}, validation set ppl {dev_ppl:.1f}, prior best: "
                       f"{prior_best_ppl:.1f}")
                print(msg, file=sys.stderr)

                is_better = prior_best_ppl is None or dev_ppl < prior_best_ppl
                validation_ppl.loc[train_iter] = dev_ppl # Log the performance

                if is_better: # Check if the current model is better than the prior ones, if so save
                    patience = 0 # Reset the early stopping patience counter, we've found a new best model
                    print(f"  Saving the new best model to [{model_save_path}]", file=sys.stderr)
                    model.save(model_save_path)
                    # Also save the optimizer's state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                    # Also log the performance history over time
                    training_ppl.to_csv(os.path.join(model_save_dir, "training_ppl.csv"))
                    validation_ppl.to_csv(os.path.join(model_save_dir, "validation_ppl.csv"))

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

                            # Set new the new learning rate (lr)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr

                            patience = 0 # Reset the patience counter now that lr is lower

            if epoch == max_epochs:
                print('Reached maximum number of epochs!', file=sys.stderr)
                return



##############################################################################################################
### UNIT TESTING of the train function

if __name__ == "__main__":
    args = docopt(__doc__)
    # TODO: Probably don't need these items here given the defaults above
    model_class = str(args.get("--model", "LSTM_Att")) # Designate which model class to train
    embed_size = int(args.get("--embed-size", 512)) # Specify the word vec embedding size
    hidden_size = int(args.get("--hidden-size", 512)) # Specify the hidden state
    num_layers =  int(args.get("--num-layers", 1)) # Specify how many layers the model has
    dropout =  float(args.get("--dropout-rate", 0.3)) # Specify the dropout rate for training
    src_lang = args.get("--src-lang", "deu") # Specify the source language (from)
    tgt_lang = args.get("--tgt-lang", "eng") # Specify the target language (to)
    assert src_lang != tgt_lang, "soruce language must differ from target language"
    debug = args.get("--debug", False) # Specify if the training is to be run in debug mode

    print(f"Starting training process for {src_lang} to {tgt_lang}. Data set pre-processing...")
    start_time = time.time()
    # Build the data set for training and validation
    data_set_name = "train_1" if debug is False else "train_debug"
    train_data_src = read_corpus(src_lang, data_set_name, is_tgt=False)
    train_data_tgt = read_corpus(tgt_lang, data_set_name, is_tgt=True)
    train_data = list(zip(train_data_src, train_data_tgt))
    print(f"  Training data processed: {time.time() - start_time:.1f}s")
    start_time = time.time() # Reset the timer start for next step

    val_data_src = read_corpus(src_lang, "validation", is_tgt=False)
    val_data_tgt = read_corpus(tgt_lang, "validation", is_tgt=True)
    dev_data = list(zip(val_data_src, val_data_tgt))
    print(f"  Validation data processed: {time.time() - start_time:.1f}s")

    vocab = Vocab.load(f"vocab/{src_lang}_to_{tgt_lang}_vocab")

    train_params = {} if debug is False else DEBUG_TRAIN_PARAMS.copy()

    if args["--warm-start"] == "True": # Load an existing model and continue training from where we left off
        model = getattr(all_models, model_class).load(f"saved_models/{model_class}/model.bin")
        train_params["uniform_init"] = 0 # Do NOT randomly initiate the model parameters
        train_params["load_optimizer"] = True # Use the prior optimizer saved from training
    else: # Initialize a new model instance to be trained
        model_kwargs = {"embed_size": embed_size, "hidden_size": hidden_size, "num_layers": num_layers,
                        "dropout_rate": dropout, "vocab": vocab}
        model = getattr(all_models, model_class)(**model_kwargs)
        train_params["uniform_init"] = 0.1 # Initialize weights randomly

    model_save_dir = f"saved_models/{model.name}/"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True) # Make the save dir if not already there

    # Run a training pass for the model
    train_model(model, train_data, dev_data, model_save_dir, params=train_params)


##############################################################################################################
## TODO: Clean up the below functions and other stuff
## TODO: Need to test running this



# def decode(args: Dict[str, str]):
#     """ Performs decoding on a test set, and save the best-scoring decoding results.
#     If the target gold-standard sentences are given, the function also computes
#     corpus-level BLEU score.
#     @param args (Dict): args from cmd line
#     """

#     print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
#     test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src', vocab_size=3000)
#     if args['TEST_TARGET_FILE']:
#         print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
#         test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt', vocab_size=2000)

#     print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
#     model = NMT.load(args['MODEL_PATH'])
#     model = model.to(setup_device(args['--gpu']))

#     hypotheses = beam_search(model, test_data_src,
#                             #  beam_size=int(args['--beam-size']),                      EDIT: BEAM SIZE USED TO BE 5
#                              beam_size=10,
#                              max_decoding_time_step=int(args['--max-decoding-time-step']))

#     if args['TEST_TARGET_FILE']:
#         top_hypotheses = [hyps[0] for hyps in hypotheses]
#         bleu_score = compute_corpus_bleu_score(test_data_tgt, top_hypotheses)
#         print('Corpus BLEU: {}'.format(bleu_score), file=sys.stderr)

#     with open(args['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
#         for src_sent, hyps in zip(test_data_src, hypotheses):
#             top_hyp = hyps[0]
#             hyp_sent = ''.join(top_hyp.value).replace('▁', ' ')
#             f.write(hyp_sent + '\n')


# def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
#     """ Run beam search to construct hypotheses for a list of src-language sentences.
#     @param model (NMT): NMT Model
#     @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
#     @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
#     @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
#     @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
#     """
#     was_training = model.training
#     model.eval()

#     hypotheses = []
#     with torch.no_grad():
#         for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
#             example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

#             hypotheses.append(example_hyps)

#     if was_training: model.train(was_training)

#     return hypotheses


# def main():
#     """ Main func.
#     """
#     args = docopt(__doc__)
#     print(args)

#     # Check pytorch version
#     assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

#     # seed the random number generators
#     seed = int(args['--seed'])
#     torch.manual_seed(seed)
#     np.random.seed(seed * 13 // 7)

#     if args['train']:
#         train(args)
#     elif args['decode']:
#         decode(args)
#     else:
#         raise RuntimeError('invalid run mode')


# if __name__ == '__main__':
#     main()
