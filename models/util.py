#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import namedtuple
import sys, os
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


### TOODO: Make sure this abstract class has all the things that we need for training and eval etc.
# e.g. should have self.source_embeddings and self.target_embeddings, greedy_search etc. but not others
# that are not strictly required e.g. encode and decode, or maybe we do. Ideally the Google API model would
# Fit this mold. Think about what things are assumed to be common and called during training and eval that
# all models absolutely must have

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
    def greedy_search(self, *args, **kwargs):
        pass

    @property
    def device(self) -> torch.device:
        """
        Method for determining which device to place the Tensors upon, CPU or GPU.
        """
        return self.source_embeddings.weight.device

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

    def size(self):
        """
        Prints out key info about the model's size.
        """
        print(f"model: {self.name}")
        print(f"embed_size: {self.embed_size}")
        print(f"hidden_size: {self.hidden_size}")
        if hasattr(self, "num_layers"):
            print(f"num_layers: {self.num_layers}")

# TODO: move common stuff here that we don't need to define in each model
# Define what methods need to be implemented by all models so that the evaluation tools are consistent etc.
