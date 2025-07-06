#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import namedtuple
import sys, os
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        # TODO: Add descriptions here of what is expected
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def beam_search(self, *args, **kwargs):
        pass

    @property
    def device(self) -> torch.device:
        """
        Method for determining which device to place the Tensors upon, CPU or GPU.
        """
        return self.source_embeddings.weight.device

    @abstractmethod
    @classmethod
    def load(cls, model_path: str):
        """
        Method for loading in model weights saved locally to disk.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = cls(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        return model

    @abstractmethod
    def save(self, model_path: str):
        pass


# TODO: move common stuff here that we don't need to define in each model
# Define what methods need to be implemented by all models so that the evaluation tools are consistent etc.



