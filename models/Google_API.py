#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.util import NMT
from vocab.vocab import Vocab
import util


class Google_API(NMT):
    """
    This model class returns translations made using the Google Translate API. No translations are actually
    done on-the-fly, rather this class reads in a set of pre-cached data that was translated using the Google
    Translate API and returns translations it recognizes as pre-computed results. If the input sentence
    is unrecognized, it will return a blank string.
    """
    def __init__(self, src_lang: str, tgt_lang: str, *args, **kwargs):
        """
        Stand-in model for retrieving pre-cached translations of the train_debug, validation, and test data.

        Parameters
        ----------
        vocab : Vocab
            A Vocabulary object containing source (src) and target (tgt) language vocabularies.
        """
        super().__init__()
        self.embed_size = np.nan  # Record the word vector embedding dimensionality
        self.hidden_size = np.nan # Record the size of the hidden states used by the LSTMs
        self.vocab = Vocab.load(f"{src_lang}_to_{tgt_lang}_vocab")
        self.name = "Google_API"
        self.lang_pair = (self.vocab.src_lang, self.vocab.tgt_lang) # Record the language pair

        # Load in the pre-translated sentence translations from cache
        cached_data = []
        for file_name in os.listdir("google_api/"):
            if file_name.endswith(".csv"):
                cached_data.append(pd.read_csv("google_api/" + file_name, encoding="utf-8"))

        cached_data = pd.concat(cached_data) # Combine into 1
        if self.vocab.src_lang == "eng": # Configure the model to do eng -> deu translations
            self.cache_dict = {A: B for A, B in zip(cached_data["eng"], cached_data["deu_google"])}
        elif self.vocab.src_lang == "deu": # Configure the model to do deu -> eng translations
            self.cache_dict = {A: B for A, B in zip(cached_data["deu"], cached_data["eng_google"])}
        else:
            raise AttributeError("vocab src_lang must be either 'eng' or 'deu'")

        self.dummy_layer = nn.Linear(10, 10)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """
        Takes a mini-batch of source and target sentences, compute the negative log-likelihood of the target
        sentences under the language models learned by the NMT system.
        Return a tensor of size (batch_size) of all 0s.
        """
        return torch.zeros(len(source))

    def translate(self, src_sentences: Union[List[str], List[List[str]]], tokenized: bool = True,
                      *args, **kwargs) -> List[List[Union[Union[str, List[str]], float]]]:
        """
        Given a list of source sentences (src_sentences) this method returns the pre-cached translations
        from Google if they exist, otherwise a blank string is returned if the input sentence is not cached.

        Set tokenized = False if src_sentences is passed as a list of sentence strings or True if they have
        already been tokenized into list of sub-word tokens. The returned output will match the input i.e.
        lists of sub-word tokens will be returned if tokenized = True.

        Parameters
        ----------
        src_sentences : Union[List[str], List[List[str]]]
            A list of input source sentences stored as strings (if tokenize is True)
            e.g. ["Wo ist due bank?", ...]
            Or a list of input source sentences where each is a list of sub-word tokens if tokenize is False
            e.g. [['▁Wo', '▁ist', '▁die', '▁Bank', '?'], ...]
        tokenized : bool, optional
            Denotes whether src_sentences has already been tokenized.

            If False, then src_sentences is assumed to be a list of sentences stored as strings which will be
            tokenized internally before being fed into the model. If False, then the output list of machine
            translations for each input sentence will also be sentences stored as strings.
            E.g. [['Where is the Bank?', 0.9648], ...]

            If True, the src_sentences is assumed to be a list of sub-word token lists which can be fed into
            the model directly. If True, then the output list of machien translations for each input sentence
            will be a list of sub-word tokens similar to the way src_sentences was input.
            E.g. [[['<s>', '▁Where', '▁is', '▁the', '▁Bank', '?', '</s>'], 0.9648], ...]

        Returns
        -------
        List[List[Union[Union[str, List[str]], float]]]
            Returns a list of hypotheses i.e. length 2 lists each containing:
                - The predicted translation from the model as either a string (if tokenize is True) or a
                  list of sub-word tokens (if tokenize is False).
                - A negative log-likelihood score of the decoding as a float
        """
        b = len(src_sentences) # Record how many input sentences there are i.e. the batch size
        assert b > 0, "len(src_sentences) must be >= 1"
        if tokenized is True and isinstance(src_sentences[0], str):
            # If 1 sentence of word-tokens is passed in, then add an outer list wrapper
            src_sentences = [src_sentences] # Make src_sentences a list of lists
            b = len(src_sentences) # Redefine to be 1

        if tokenized is True: # Tokenization not needed, src_sentences are already tokenized
            # src_sentences comes in pre-tokenized, remove the tokenization and attempt to locate in the
            # lookup cache of pre-translated sentences, return a list for each which is length 1
            mt = [self.cache_dict.get(util.tokens_to_str(s), "") for s in src_sentences] # Look up
            mt = util.tokenize_sentences(mt, self.lang_pair[0], is_tgt=False) # Tokenize
            mt = [[token_list, 0] for token_list in mt] # Re-format as a list of lists
        else: # If the sentences are not tokenized, then use them directly in the look up
            mt = [[self.cache_dict.get(s, ""), 0] for s in src_sentences]

        return mt

    def save(self, model_path: str):
        """
        Placeholder method, no action for this model.
        """
        pass
