# -*- coding: utf-8 -*-
"""
This module contains helper functions for model evaluation e.g. functions to compute model perplexity and
BLEU scores etc. and also model summary comparison tables.
"""

import models.all_models as all_models
import sentencepiece as spm
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import sacrebleu
import matplotlib.pyplot as plt
from models.util import Hypothesis, NMT
import torch, os
import util

#####################################################
### Model Evaluation Metric Calculation Functions ###
#####################################################

def compute_perplexity(model: NMT, eval_data: List[Tuple[List[str]]], batch_size: int = 32) -> float:
    """
    Computes a perplexity score of the model over an evaluation data set (eval_data).

    Parameters
    ----------
    model : NMT
        A NMT model instance to be evaluated.
    eval_data : List[Tuple[List[str]]]
        A list of (src_sentence, tgt_sentence) list of tuples containing source and target sentences stored
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
        for src_sentences, tgt_sentences in util.batch_iter(eval_data, batch_size):
            loss = -model(src_sentences, tgt_sentences).sum() # Compute the forward function i.e. the
            # negative log-likelihood of the output target words according to the model
            cuml_loss += loss.item() # Accumulate the loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sentences)  # omitting leading `<s>`
            cuml_tgt_words += tgt_word_num_to_predict # Count tgt words that are in this batch of eval_data

        ppl = np.exp(cuml_loss / cuml_tgt_words) # Compute the preplexity score of the model

    if was_training: # If the model was training, then set it back to that config before exiting
        model.train()

    return ppl


def compute_corpus_bleu_score(model: NMT, eval_data: List[Tuple[List[str]]]) -> float:
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
        A list of gold-standard translations i.e. a list sentences which are lists of sub-word tokens.
    hypotheses : List[Hypothesis]
        A list of hypotheses, one for each reference translation, which are lists of sub-word token outputs
        from a translation model.

    Returns
    -------
    float
        A corpus-level BLEU score.
    """
    references, hypotheses = [], []
    for src_sentence, tgt_sentence in eval_data: # Iterate through the data set provided
        # Compute what a model-generated translation of each input sentence
        hypotheses.append(model.greedy_search(src_sentence, max_decode_length=len(src_sentence) * 2)[0])
        references.append(tgt_sentence) # Record the "gold-standard" translation of this sentence

    # Detokenize the subword pieces to get full sentences
    detokenized_refs = [util.tokens_to_str(s) for s in references]
    detokenized_hyps = [util.tokens_to_str(hyp) for hyp in hypotheses]
    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one per output
    bleu = sacrebleu.corpus_bleu(detokenized_hyps, [detokenized_refs])
    return bleu.score


# TODO: Add more automatic evaluation metrics here as well, make sure they all follow the same input conventions


######################################################
### Model Performance Summary Generation Functions ###
######################################################

def build_eval_dataset(data_set_name: str) -> Dict[str, List[Tuple[List[str]]]]:
    """
    Builds an evaluation data set i.e. a dictionary containing the dataset to be used for automatic model
    evaluation and expected by generate_model_summary_table. The keys of the dict are"EngDeu" and "DeuEng"
    for the 2 directions of translation and the values are a list of parallel sentence tuples
    (src_sentence, tgt_sentence) where each sentence is recorded as a list of sub-word tokens.

    Parameters
    ----------
    data_set_name : str
        The name of the data set to read from disk and generate an eval dataset for e.g. "train_1" or "test".

    Returns
    -------
    Dict[str, List[Tuple[List[str]]]]
       A dictionary containing the dataset to be used for automatic model evaluation and expected by
       generate_model_summary_table. The keys of the dict are"EngDeu" and "DeuEng" for the 2 directions of
       translation and the values are a list of parallel sentence tuples (src_sentence, tgt_sentence) where
       each sentence is recorded as a list of sub-word tokens.
    """
    eval_data_dict = {}

    # EngDeu evaluation data set
    src_data = util.read_corpus("eng", data_set_name, is_tgt=False)[:25] # TODO: TEMP
    tgt_data = util.read_corpus("deu", data_set_name, is_tgt=True)[:25] # TODO: TEMP
    eval_data_dict["EngDeu"] = list(zip(src_data, tgt_data))

    # DeuEng evaluation data set
    src_data = util.read_corpus("deu", data_set_name, is_tgt=False)[:25] # TODO: TEMP
    tgt_data = util.read_corpus("eng", data_set_name, is_tgt=True)[:25] # TODO: TEMP
    eval_data_dict["DeuEng"] = list(zip(src_data, tgt_data))

    return eval_data_dict


def generate_eval_summary(model: NMT, eval_data: List[Tuple[List[str]]]) -> pd.Series:
    # TODO: Add a doc string - Compute evaluation metrics for this model
    eval_summary = pd.Series(dtype=float)
    eval_summary.loc["Perplexity"] = compute_perplexity(model, eval_data)
    eval_summary.loc["BLEU"] = compute_corpus_bleu_score(model, eval_data)
    # TODO: Add other evaluation metrics here
    return eval_summary


def generate_model_summary_table(model_classes: List[str],
                                 eval_data_dict: Dict[str, List[Tuple[List[str]]]]) -> pd.DataFrame:
    """
    This function generates a comparative performance summary table across all models in model_classes using
    the same evaluation data for each model contained in eval_data_dict. The output summary table has the
    model class name as the index (rows) and for each it records A). model size in terms of Embed Size,
    Hidden Size and Total (Trainable) Params B). automatic evaluation metrics for DeuEng and C). automatic
    evaluation metrics for EngDeu.

    Parameters
    ----------
    model_classes : List[str]
        A list of model class names e.g. ['Fwd_RNN_3', 'LSTM_Att', 'LSTM_AttNN'].
    eval_data_dict : Dict[str, List[str]]
        A dictionary containing the dataset to be used for automatic model evaluation. The keys of the dict
        should be "EngDeu" and "DeuEng" for the 2 directions of translation and the values should be a list
        of parallel sentence tuples (src_sentence, tgt_sentence) where each sentence is recorded as a list of
        sub-word tokens.

    Returns
    -------
    summary_table : pd.DataFrame
        A comparative summary across models.
    """
    metrics = ["Perplexity", "BLEU"] # Enumerate all the evaluation metrics used
    cols = pd.MultiIndex.from_tuples([("Model", x) for x in ["Embed Size", "Hidden Size", "Total Params"]] +
                                     [("DeuEng", x) for x in metrics] + [("EngDeu", x) for x in metrics])
    summary_table = pd.DataFrame(index=model_classes, columns=cols)

    for model_class in model_classes: # Try to generate data for this model if possible
        for (src_lang, tgt_lang) in [("deu", "eng"), ("eng", "deu")]:
            translation_name = f"{src_lang.capitalize()}{tgt_lang.capitalize()}"

            model_save_dir = util.get_model_save_dir(model_class, src_lang, tgt_lang, False)
            if os.path.exists(f"{model_save_dir}/model.bin"): # Check if there is a model saved in this dir
                model = getattr(all_models, model_class).load(f"{model_save_dir}/model.bin")
                summary_table.loc[model_class, ("Model", "Embed Size")] = model.embed_size
                summary_table.loc[model_class, ("Model", "Hidden Size")] = model.hidden_size
                col =  ("Model", "Total Params")
                summary_table.loc[model_class, col] = util.count_trainable_parameters(model)
                model_smry = generate_eval_summary(model, eval_data_dict[translation_name])
                for eval_metric, metric_score in model_smry.items():
                    summary_table.loc[model_class, (translation_name, eval_metric)] = metric_score
            else:
                print(f"No {translation_name} found for {model_class}")

    return summary_table


if __name__ == "__main__":
    model_classes = ['Fwd_RNN', 'LSTM_Att', 'LSTM_AttNN'] # All the models to evaluate

    # Generate evaluation tables for each data set
    for data_set_name in ["train_debug"]: # ["train_1", "validation", "test"]
        print(f"Running model evaluation for {model_classes} using dataset={data_set_name}")
        summary_table = generate_model_summary_table(model_classes, build_eval_dataset(data_set_name))
        summary_table.to_csv(f"eval_tables/{data_set_name}_eval.csv")
        print(f"\nSummary Table for Dataset={data_set_name}")
        print(summary_table)


#### TODO: Also do some qualitative analysis, compare the translations EngDeu and DeuEng vs the ""ground truth"
# in 3 differenet categoreis according to sentence length

### TODO: ADD QUALITATIVE ANALYSIS HERE



## TODO: Look at whether I should do anything with the below or not

# def decode(args: Dict[str, str]):
#     """
#     Performs decoding on a test set, and save the best-scoring decoding results.
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
#                               beam_size=10,
#                               max_decoding_time_step=int(args['--max-decoding-time-step']))

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



