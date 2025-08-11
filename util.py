# -*- coding: utf-8 -*-
"""
This module contains functionalities shared among multiple modules e.g. train, model_eval etc.
"""
from typing import List, Tuple, Union
import sentencepiece as spm
import pandas as pd
import numpy as np
import math, os, sys, importlib
import torch.nn as nn

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
    else: # Differentiate so that we do not accidently save over a trained model when debug testing
        return f"saved_models/debug/{model_class}/{src_lang.capitalize()}{tgt_lang.capitalize()}/"


def read_corpus(lang: str, subset: str, is_tgt: bool, tokenize: bool = True) -> List[List[str]]:
    """
    Reads in a text corpus file from disk specified by file_path. This function is primarily used for
    creating training, validation, and testing data sets. The contents of the given file are read in and
    tokenized by a pre-trained tokenizer model. This function returns a list of lists containing sub-word
    tokens according to a pre-trained tokenizer model saved to disk.

    Set is_tgt = True if the data set being read in is to be used as a target data set. If so, then all the
    sentences will have a <s> start sentence token appeneded to the front and a </s> end sentence token
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
    sentences = pd.read_csv(file_path) # Read in the entire specified data set using pandas
    if tokenize is False:
        return sentences.iloc[:, 0].to_list()

    tokenized_sentences = [] # Collect the tokenized sentences
    sp = spm.SentencePieceProcessor() # Instantiate the tokenizer model
    sp.load(f"vocab/{lang}/{lang}.model") # Load in the pre-trained tokenizer model
    tokenized_sentences = sentences[sentences.columns[0]].astype(str).apply(lambda x: sp.encode_as_pieces(x))
    tokenized_sentences = list(tokenized_sentences.values) # Convert to a list of lists

    if is_tgt is True: # Only append <s> and </s> tokens if this is a target data set
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
    n_batches = math.ceil(len(data) / batch_size) # How many total batches to iter over the whole data set
    batch_idx = list(range(n_batches)) # Number the batches

    if shuffle is True: # Sort by source sentence length before batching, then randomly shuffle the batches
        data = sorted(data, key=lambda x: len(x[0]), reverse=True) # Sort by source sentence, longest first
        np.random.shuffle(batch_idx) # Randomly shuffle the ordering of the batchs to use

    for i in batch_idx: # Iterate over how many batches are required to cover the whole data set
        examples = data[(i * batch_size):(i + 1) * batch_size]
        src_sentences = [e[0] for e in examples]
        tgt_sentences = [e[1] for e in examples]
        yield (src_sentences, tgt_sentences)


def tokenize_sentences(sentences: List[str], lang: str, is_tgt: bool = False) -> List[List[str]]:
    """
    Applies the pre-trained sub-word tokenizer to a collection of input sentences. If is_tgt is True, then
    <s> start and </s> end of sentence tokens are appened to the front and back of each sentence.

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
    sp = spm.SentencePieceProcessor() # Instantiate the tokenizer model
    sp.load(f"vocab/{lang}/{lang}.model") # Load in the pre-trained weights
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
    tokenized_sentence = tokenized_sentence.copy() # Copy to avoid mutation
    if tokenized_sentence[-1] == "</s>": # Remove the start token if present
        tokenized_sentence.pop()
    if tokenized_sentence[0] == "<s>": # Remove the end token if present
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
