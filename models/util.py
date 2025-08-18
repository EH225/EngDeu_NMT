#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import namedtuple
import sys, os
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils


class NMT(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        This method should return the negative log-likelihood of a batch of input sentences using the teacher
        forcing method for computing the loss and back propagating to train the model.
        """
        pass

    @abstractmethod
    def translate(self, *args, **kwargs):
        """
        This method should take a batch of input source sentences and translate them to the target language.
        """
        pass

    @property
    def device(self) -> torch.device:
        """
        This is a method for determining which device to place the Tensors upon, CPU or GPU.
        """
        return self.source_embeddings.weight.device

    @classmethod
    def load(cls, model_path: str):
        """
        Method for loading in a model saved to disk.

        Parameters
        ----------
        model_path : str
            A file path detailing where the model should be saved e.g. saved_models/{model}/DeuEng/model.bin

        Returns
        -------
        model : {self.name}
            Returns an object instance of this model class with the weights saved to disk.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = cls(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        return model

    @abstractmethod
    def save(self, model_path: str):
        """
        Method for saving the model's weights and init parameters to disk.
        """
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
        if hasattr(self, "n_heads"):
            print(f"n_heads: {self.n_heads}")
