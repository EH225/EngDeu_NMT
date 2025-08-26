# -*- coding: utf-8 -*-
"""
This module contains general utility functionalities shared among multiple modules e.g. train, model_eval etc.
"""
from typing import List, Tuple, Union
import sentencepiece as spm
import pandas as pd
import numpy as np
import math, os, sys, importlib
import torch.nn as nn
import torch


def setup_device(try_gpu: bool = True):
    """
    Setup the device used by PyTorch. If try_gpu is True, then we will attempt to locate GPU hardware.
    """
    device = torch.device("cpu")  # Set to the CPU by default

    if try_gpu is True:  # Try looking for a GPU if there is one we can connect to
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")

    return device


def get_model_save_dir(model_class: str, src_lang: str, tgt_lang: str, debug: bool = False) -> str:
    """
    Returns a string of the model save directory based on the inputs.

    Parameters
    ----------
    model_class : str
        The name of the model class e.g. LSTM_Att.
    src_lang : str
        The input language of the translation model e.g. "eng".
    tgt_lang : str
        The output language of the translation model e.g. "deu".
    debug : bool
        Whether to use the debug location instead of the production one.

    Returns
    -------
    str
        The model's save directory.
    """
    if debug is False:
        return f"saved_models/{model_class}/{src_lang.capitalize()}{tgt_lang.capitalize()}/"
    else:  # Differentiate so that we do not accidentally save over a trained model when debug testing
        return f"saved_models/debug/{model_class}/{src_lang.capitalize()}{tgt_lang.capitalize()}/"


def read_corpus(lang: str, subset: str, is_tgt: bool, tokenize: bool = True) -> List[List[str]]:
    """
    Reads in a text corpus file from disk specified by file_path. This function is primarily used for
    creating training, validation, and testing data sets. The contents of the given file are read in and
    tokenized by a pre-trained tokenizer model. This function returns a list of lists containing sub-word
    tokens according to a pre-trained tokenizer model saved to disk.

    Set is_tgt = True if the data set being read in is to be used as a target data set. If so, then all the
    sentences will have a <s> start sentence token appended to the front and a </s> end sentence token
    appended to the end.

    Parameters
    ----------
    lang : str
        The language to read in data for i.e. either "eng" or "deu".
    subset : str
        The data subset to read in e.g. "train_1", "validation", "test"
    is_tgt : bool
        A bool flag indicating if the data set is to be used as a target data set.
    tokenize : bool
        If True, then the sentences read in are tokenized before being returned. Otherwise, this function
        returns a list of sentences stored as strings.

    Returns
    -------
    List[List[str]]
        List of lists where each list is a collection of sub-word tokens.
    """
    file_path = f"dataset/{lang}/{subset}.csv"
    sentences = pd.read_csv(file_path)  # Read in the entire specified data set using pandas
    if tokenize is False:
        return sentences.iloc[:, 0].to_list()

    tokenized_sentences = []  # Collect the tokenized sentences
    sp = spm.SentencePieceProcessor()  # Instantiate the tokenizer model
    sp.load(f"vocab/{lang}/{lang}.model")  # Load in the pre-trained tokenizer model
    tokenized_sentences = sentences[sentences.columns[0]].astype(str).apply(lambda x: sp.encode_as_pieces(x))
    tokenized_sentences = list(tokenized_sentences.values)  # Convert to a list of lists

    if is_tgt is True:  # Only append <s> and </s> tokens if this is a target data set
        for s in tokenized_sentences:
            s.insert(0, "<s>")
            s.append('</s>')

    return tokenized_sentences


def batch_iter(data: List[Tuple[List[str]]], batch_size: int, shuffle: bool = False):
    """
    Generator that yields batches of (source, target) sentences. If shuffle is True, then the sentence pairs
    are sorted in descending order of source sentence length before being batched so that within each batch
    they are sorted by length and also so that sentences of similar length are batched together which reduces
    the amount of padding used in the batch overall. The order of the batches are then also shuffled so that
    the longest sentences do not dominate all of the initial batches.

    Parameters
    ----------
    data : List[Tuple[List[str]]]
        A list of paired (source, target) sentence tuples.
    batch_size : int
        The number of paired sentences per batch.
    shuffle : bool, optional
        If True, then the data set is sorted by source sentence length, batched into equal sized blocks, and
        then randomly shuffled. This should be used during training. The default is False.

    Yields
    ------
    src_sentences : List[List[str]]
        A list of source language sentences.
    tgt_sentences : List[List[str]]
        A list of target language sentences.

    """
    n_batches = math.ceil(len(data) / batch_size)  # How many total batches to iter over the whole data set
    batch_idx = list(range(n_batches))  # Number the batches

    if shuffle is True:  # Sort by source sentence length before batching, then randomly shuffle the batches
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)  # Sort by source sentence, longest first
        np.random.shuffle(batch_idx)  # Randomly shuffle the ordering of the batches to use

    for i in batch_idx:  # Iterate over how many batches are required to cover the whole data set
        examples = data[(i * batch_size):(i + 1) * batch_size]
        src_sentences = [e[0] for e in examples]
        tgt_sentences = [e[1] for e in examples]
        yield (src_sentences, tgt_sentences)


def tokenize_sentences(sentences: List[str], lang: str, is_tgt: bool = False) -> List[List[str]]:
    """
    Applies the pre-trained sub-word tokenizer to a collection of input sentences. If is_tgt is True, then
    <s> start and </s> end of sentence tokens are append to the front and back of each sentence.

    Parameters
    ----------
    sentences : List[str]
        A list of plain text sentences (strings) to be tokenized. Note, the sentences are full strings and
        not lists of words.
    lang : str
        The language of the input sentences i.e. either "eng" or "deu".
    is_tgt : bool, optional
        If set to True, then <s> and </s> tokens are added to each sentence. The default is False.

    Returns
    -------
    List[List[str]]
        Outputs a list of sentences that are each a list of sub-word tokens.
    """
    sp = spm.SentencePieceProcessor()  # Instantiate the tokenizer model
    sp.load(f"vocab/{lang}/{lang}.model")  # Load in the pre-trained weights
    tokenized_sentences = [sp.encode_as_pieces(sentence) for sentence in sentences]
    if is_tgt is True:  # Only append <s> and </s> tokens if this is a target data set
        for s in tokenized_sentences:
            s.insert(0, "<s>")
            s.append('</s>')

    return tokenized_sentences


def tokens_to_str(tokenized_sentence: List[str]) -> str:
    """
    Converts a list of sub-word tokens (a tokenized sentence) into a string representation. In essence, this
    function reverses the tokenization yielding a plain test sentence.

    Parameters
    ----------
    tokenized_sentence : List[str]
        A list of sub-word tokens.

    Returns
    -------
    str
        A plain text sentence derived from the sub-word tokens.
    """
    tokenized_sentence = tokenized_sentence.copy()  # Copy to avoid mutation
    if len(tokenized_sentence) > 0 and tokenized_sentence[-1] == "</s>":  # Remove the start token if present
        tokenized_sentence.pop()
    if len(tokenized_sentence) > 0 and tokenized_sentence[0] == "<s>":  # Remove the end token if present
        tokenized_sentence.pop(0)
    return ''.join(tokenized_sentence).replace('▁', ' ').strip()


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are to be counted.

    Returns
    -------
    int
        The total number of trainable parameters.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_heatmap(df: pd.DataFrame, max_val: float = None, min_val: float = None, cmap: str = "RdBu",
                 fmt: str = None, show_cbar: bool = False, ax=None) -> None:
    """
    Generates a heatmap summary table to display a DataFrame. The default color scheme is Red (low) to Blue
    (high). The color scheme can be changed by altering the cmap parameter. If neither min_val nor max_val
    is provided, then the color scheme will auto-detect the max absolute value present in the input df and
    use that value for the min and max color scheme values. If only one provided (either max_val or min_val)
    then the other will be inferred by multiplying the one provided by -1 to create a symmetric coloring about
    0. Note, max_val must be > min_cal otherwise the coloring will default to [-2, +2].

    E.g. max_val=2.5 means that any value in summary_df with an absolute value of 2.5 or greater will have
    the maximally dark color saturation applied (either red or blue). This parameter is used to keep the
    shading convention consistent across various tables with differing min and max values so that they are
    visually comparable (i.e. the same color hue has the same meaning across tables).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing float values to be displayed as a heatmap.
    max_val : float, optional
        Sets the upper threshold for max color saturation. The default is None.
    min_val : float, optional
        Sets the lower threshold for max color saturation. The default is None.
    cmap : str, optional
        A color palette (cmap) compatible with the seaborn package. "RdBu" is the default value. Use "Reds"
        or "Blues" or some other mino-chromatic cmap and set a min_val for a 1-color heatmap.
        The default is "RdBu".
    fmt : str, optional
        Specifies the format of the annotations e.g. '1f' for 1 decimal float or ".2%" for percent.
    show_cbar : bool, optional
        A bool for if the color bar should be displayed on the right. The default is False.
    ax : TYPE, optional
        A matplot.pyplot plotting axis to use. One will be created if none provided. The default is None.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(df, pd.Series):  # Convert to ad pd DF if given a series
        df = df.to_frame().T

    if ax is None:  # If no ax provided, create a new plotting axis, auto-set figsize
        nrow, ncol = df.shape  # Use the input summary_df shape for the figsize
        if show_cbar is True:
            fig, ax = plt.subplots(1, 1, figsize=(12 / 18 * ncol, 4 / 8 * nrow))
        else:  # Slightly smaller because there is no color bar shown on the right
            fig, ax = plt.subplots(1, 1, figsize=(11 / 18 * ncol, 4 / 8 * nrow))

    # Set default formatting parameters if not specified by user
    fmt = ".1f" if fmt is None else fmt

    if max_val is None:  # If not provided, infer from the data in the summary_df
        if min_val is None:  # No value provided for either, infer based on max(|data|)
            vmax = abs(df).max().max()
            vmin = -vmax
        else:  # If a min_val provided but no max_val, create a symmetric color scheme about 0
            vmax, vmin = -min_val, min_val
    else:  # max_val is not None, then use what was provided
        if min_val is None:  # Create a color scheme symmetric about 0 using [-max_val, max_val]
            vmax, vmin = max_val, -max_val
        else:  # When both are provided by the user or preset default
            vmax, vmin = max_val, min_val
    if vmax <= vmin:  # For cases when max_val doesn't exceed min_val, set to a default
        vmin, vmax = -2, 2

    sns.heatmap(df.astype(float), ax=ax, cmap=cmap, fmt=fmt, cbar=show_cbar, vmin=vmin, vmax=vmax,
                annot=df.values, annot_kws={'size': 9})

    # Display y-ticks horizontally for ease of reading
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
