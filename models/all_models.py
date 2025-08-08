# -*- coding: utf-8 -*-
"""
Imports all models into 1 module for quick access and reference.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..')))

# Whenever we add another model to the library, add it here
from models.Fwd_RNN import Fwd_RNN
from models.LSTM_Att import LSTM_Att
from models.LSTM_AttNN import LSTM_AttNN
from models.Google_API import Google_API
from models.EDTM import EDTM
