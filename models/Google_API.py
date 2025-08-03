#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from collections import namedtuple
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from models.util import NMT, Hypothesis
from vocab.vocab import Vocab
import util

class Google_API(nn.Module):
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
        self.vocab = Vocab.load(f"vocab/{src_lang}_to_{tgt_lang}_vocab")
        self.name = "Google_API"

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
        Takes a mini-batch of source and target sentences, compute the log-likelihood of the target sentences
        under the language models learned by the NMT system. Return a tensor of size (batch_size) of all 0s.
        """
        return torch.zeros(len(source))


    def greedy_search(self, src_sentences: List[List[str]], *args,
                      **kwargs) -> List[List[Union[List[str], int]]]:
        """
        Given a list of source sentences (src_sentences) this method returns the pre-cached translations
        from Google if they exist, otherwise a blank string is returned if the input sentence is not cached.

        Parameters
        ----------
        src_sentences : List[List[str]]
            A list of input source sentences where each is a list of sub-word tokens.
            e.g. ['▁Wo', '▁ist', '▁die', '▁Bank', '?']

        Returns
        -------
        List[List[Union[List[str], int]]]
            Returns a list of hypotheses i.e. a length 2 lists each containing:
                - A list of sub-word tokens predicted as the translation of the ith input sentence
                - A negative log-likelihood score of the decoding
        """
        b = len(src_sentences) # Record how many input sentences there are i.e. the batch size
        assert b > 0, "len(src_sentences) must be >= 1"
        if isinstance(src_sentences[0], str): # If 1 sentence is passed in, then add an outer list wrapper
            src_sentences = [src_sentences] # Make src_sentences a list of lists
            b = len(src_sentences) # Redefine to be 1

        # src_sentences comes in pre-tokenized, remove the tokenization and attempt to locate in the lookup
        # cache of pre-translated sentences, return a list for each which is length 1
        return [[[self.cache_dict.get(util.tokens_to_str(s), "")], 0] for s in src_sentences]

