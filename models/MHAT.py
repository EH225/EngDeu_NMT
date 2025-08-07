#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Originally forked from Andrej Karpathy's minGPT, modified from the Stanford XCS224N Assignment 5 code
"""

from collections import namedtuple
import sys, os
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .util import NMT, Hypothesis
from vocab.vocab import Vocab

import math
import logging # TODO: Consider adding logging in more places?

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


### TODO: Should attn_pdrop be different than resid_pdrop, we can probably make it the same for now


def get_rope_cache(hs: int, block_size: int) -> torch.Tensor:
    """
    Rotary positional embeddings (RoPE) uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(2(i-1)/hs) for i in [1, hs/2]

    Since the maximum length of any input token sequence is known, we can pre-compute these positional
    embedding rotation matrix values to increase the speed of training. This function will return a tensor
    of size (block_size, hs / 2, 2) where block_size = the max number of input tokens, hs = head size i.e.
    the size of the vectors that each attention head uses to compute dot products in the attention
    calculation, and that last dimension contains the cosine and sine values for each position of each
    dimension of the embedding

    Parameters
    ----------
    hs : int
        The dimension of the vectors used to compute attention in each head.
    block_size : int
        The max token length of any input sequence.

    Returns
    -------
    rope_cache : torch.Tensor
        A tensor of size (block_size, nh / 2, 2) that are pre-cached rotation matrix values used to apply
        rotary positional embedding to key and query vectors within each attention head.
    """
    # Create a [1, 2, ... block_size] tensor of size(block_size, hs/2)
    t_vals = torch.Tensor(range(1, block_size + 1))
    t_vals = t_vals.expand(hs // 2, block_size ).transpose(0, 1)
    i_vals = torch.Tensor(range(1, hs // 2 + 1))
    theta_i = (1 / 10000) ** (2 * (i_vals - 1) / hs)
    theta_i = theta_i.expand(block_size , hs // 2)
    rope_cache = (t_vals * theta_i) # Compute t * theta for each (block_size, hs/2)
    rope_cache = torch.dstack([rope_cache, rope_cache]) # d-Stack to get 2 last dims
    rope_cache[:, :, 0] = torch.cos(rope_cache[:, :, 0]) # Compute cos(t theta) in idx 0 of the last dim
    rope_cache[:, :, 1] = torch.sin(rope_cache[:, :, 1]) # Compute sin(t theta) in idx 1 of the last dim
    return rope_cache # (block_size, nh / 2, 2)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    Apply the RoPE to the input tensor x of size (batch_size, n_heads, time_steps, embed_size).

    rope_cache comes in with size (block_size, hs / 2, 2). We apply these vector space rotations to the key
    and query vectors before computing their dot products in the attention heads to encode positional info.

    block_size = max input sequence token length
    hs = head-size i.e. the length of the key and query vectors used by each head of the attention mechanism.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of key or query vectors of size (batch_size, nheads, T, hs).
    rope_cache : torch.Tensor
        Pre-cached RoPE vectors used to rotate x. The size of this is (block_size, hs / 2, 2)
        where T <= block_size since block_size is the max that T can ever be.

    Returns
    -------
    rotated_x : torch.Tensor
        Returns x but with it's last dimension rotated according to RoPE. Same dimensions as x i.e.
        (batch_size, nheads, T, hs).
    """
    b, nh, T, hs = x.size() # Get the dimensions of the input x tensor

    # rope_cache comes in as (block_size, hs/2, 2), truncate the end to match the length of x i.e. T
    rope_cache = rope_cache[:T, :, :] # T <= block_size so this always works
    rope_cache = rope_cache.expand((b, nh, T, hs // 2, 2)) # Extend the values as needed to fit the dimensions

    # The cosine values are in the 0 idx of the last dimension and the sine values are in idx 1 of the last
    # dimension, treat the cos values as real and the sin values as imaginary, create a vector of length hs/2
    # with each entry being cos(t theta_i) + i sin(t theta_i)

    # Compute cos(t theta_i) x_t^(i) - sin(t theta_i) x_t^(i+1)
    real_components = rope_cache[..., 0] * x[..., ::2] - rope_cache[..., 1] * x[..., 1::2]

    # Compute sin(t theta_i) x_t^(i) + cos(t theta_i) x_t^(i+1)
    img_components = rope_cache[..., 1] * x[..., ::2] + rope_cache[..., 0] * x[..., 1::2]

    rotated_x = torch.cat((real_components.unsqueeze(-1), 0 * img_components.unsqueeze(-1)), dim=-1)
    rotated_x = rotated_x.view(x.size())
    return rotated_x # Same shape as the original x input tensor (batch_size, nh, T, hs)


class SelfAttentionLayer(nn.Module):
    """
    A multi-head self-attention layer using rotary positional embeddings (RoPE) with a linear projection at
    the end. This attention sub-layer can be bi-directional or causal using making. Input tokens attend to
    one another, this attention mechanism is used in both the encoder (bi-directional, no masking) and also
    in the decoder (causal with masking).
    """

    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: int, causal: bool):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size # The size of each latent vector represenation of each token
        self.n_heads = n_heads # The number of attention heads
        self.block_size = block_size # The max length input token sequence (the max possiable tgt_len)
        self.dropout_rate = dropout_rate # Dropout probability during training
        self.causal = causal # Whether to use causal masking in the attention mechanism

        assert hidden_size % n_heads == 0, "hidden_size must be evenly divisible by n_heads"

        ######################################################################################################
        ### Define the model architecture

        # Set up the key, query, value projection matricies for all attention heads. These are the matricies
        # that transform and input tensor x into key, query, and value vectors, all of the same size
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        assert (hidden_size // n_heads) % 2 == 0, "d = hidden_size / n_heads must be even for RoPE"
        # Store the pre-computed rope_cache values for use later, block_size = max sequence input length
        self.register_buffer("rope_cache", get_rope_cache(hidden_size // n_heads, block_size))

        # Add dropout regularization layers
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

        # Add a causal mask to ensure that attention is only applied to the left in the input sequence
        # i.e. no lookahead bias in generating attention values, can only look at prior tokens at time t.
        # Create a lower left triangular matrix of 1s to designate masking i.e. do not use anything in the
        # upper right. Each row corresponds to a given time step t. The 1s in that row indicate what words
        # are available. Only have 1s at or prior to column t for each row t. Thus lower left triangular of 1s
        bs = block_size
        self.register_buffer("mask", torch.tril(torch.ones(bs, bs)).view(1, 1, bs, bs))

        # Add an output projection later
        self.final_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forwards pass evaluation through this self-attention layer which alters the input vectors
        based on the attention scores of each token to every other token if bi-directional or to ever prior
        token if causal.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of size (batch_size, seq_len, hidden_size) containing the toekn-vectors for
            each input token for the batch of text inputs.

        Returns
        -------
        y : torch.Tensor
            An output tensor of the same size as the original input tensor (batch_size, seq_len, hidden_size)
            where each token vector has been altered by the self-attention mechanism.
        """
        B, T, H = x.size() # Get the B = batch_size, T = input max seq length, H = hidden_size

        # Calculate query, key, and value vectors for all heads in this batch. Split the vectors along the
        # last dimension into head_size (hs) = hidden_size / n_heads equal sized segements, Re-roder the dims.
        hs = H // self.n_head  # Let hs be the size of each head i.e. self.n_head, E // self.n_head
        K = self.W_k(x).view(B, T, self.n_heads, hs).transpose(1, 2) # (B, nh, T, hs)
        Q = self.W_q(x).view(B, T, self.n_heads, hs).transpose(1, 2) # (B, nh, T, hs)
        V = self.W_v(x).view(B, T, self.n_heads, hs).transpose(1, 2) # (B, nh, T, hs)

        # Apply the rotary positional embeddings to the key and value vectors before computing the dot prod
        Q = apply_rope(Q, self.rope_cache)
        K = apply_rope(K, self.rope_cache)

        # Compute the self-attention scores by taking the dot product of all key and value vectors
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(hs)) # (B, nh, T, T)
        # Now we have a matrix for each sentence and each head that is (T x T) which are the attention scores
        # between all pairs of words in the sequence provided (x)

        if self.causal: # If causal, then zero out the attention scores for all words after each token
            att_scores = att_scores.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)

        att_scores = F.softmax(att_scores, dim=-1) # Apply softmax normalization along the last dimension
        # so that we have scores that sum to 1 to allow for a weighted avg of V according to the att_scores
        att_scores = self.attn_dropout(att_scores) # Apply dropout regularization

        # Compute the weighted avg value vector for each seq element using the attention scores
        y = att_scores @ V # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, H) # Re-assemble all head outputs side by side

        # Apply a final linear projection and dropout before returning
        return self.resid_dropout(self.final_proj(y))


class EncoderBlock(nn.Module):
    """
    Encoder Transformer Attention Block that computes:
        x = LayerNorm(SelfAttention(LayerNorm(x)) + x)
        x = LayerNorm(MLP(x) + x)
        return x
    """
    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size # The size of each latent vector represenation of each token
        self.n_heads = n_heads # The number of attention heads
        self.block_size = block_size # The max length input token sequence (the max possiable tgt_len)
        self.dropout_rate = dropout_rate # Dropout probability during training

        ######################################################################################################
        ### Define the model architecture

        self.ln1 = nn.LayerNorm(hidden_size) # Normalization of x going into the self-attention mechanism
        self.attn = SelfAttentionLayer(hidden_size, n_heads, block_size, dropout_rate, causal=False)

        self.ln2 = nn.LayerNorm(hidden_size) # Normalization of (att(x) + x) going into the MLP FFNN
        self.mlp = nn.Sequential(
            # Each attention adjusted word vector comes in with size hidden_size, then we apply a FFNN to it
            # with a hidden size of 4 x hidden_size
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(), # Gaussian Error Linear Unit activation function, non-linearity
            nn.Linear(4 * hidden_size, hidden_size), # Apply a linear layer to project down to hidden_size
            nn.Dropout(dropout_rate), # Apply dropout for regularization
        )
        self.ln3 = nn.LayerNorm(hidden_size) # Normalize the (mlp(x) + x) combined outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forwards pass evaluation through this transformer attention block which involves:
            x = x + SelfAttention(LayerNorm(x))
            x = x + MLP(LayerNorm(x))
            return x

        Residual connections are used in this model.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of size (batch_size, max_word_len, embed_size) containing the word-vectors for
            each input text for a batch of text inputs.

        Returns
        -------
        x : torch.Tensor
            An output tensor of the same size as the input after being passed through this attention block
            i.e. after we have adjusted the word-vector values to be more context-rich based on the attention
            scores to word vectors around it.
        """
        # Normalize the x input vector and then pass it through the self-attention block, then add x to that
        # output to form a residual connection and then norm the combined output
        x = self.ln2(self.attn(self.ln1(x)) + x)
        # Pass the updated x into the multi-layer-perceptron (MLP) FFNN, then add x to that output to form
        # a residual connection and then norm the combined output one more time
        x = self.ln3(self.mlp(x) + x)
        return x # (batch_size, seq_len, hidden_size)


class CrossAttentionLayer(nn.Module):
    """
    A multi-head cross-attention layer using rotary positional embeddings (RoPE) with a linear projection at
    the end. Performs an attention operation where we use inputs from the prior decoder sub-layer to create
    the query vectors and the outputs from the encoder to create the key and value vectors.

    This attention block does not use masking because we are attending to input sequence tokens which are
    available at all time steps in the decoder.
    """

    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size # The size of each latent vector represenation of each token
        self.n_heads = n_heads # The number of attention heads
        self.block_size = block_size # The max length input token sequence (the max possiable tgt_len)
        self.dropout_rate = dropout_rate # Dropout probability during training

        assert hidden_size % n_heads == 0, "hidden_size must be evenly divisible by n_heads"

        ######################################################################################################
        ### Define the model architecture

        # Set up the key, query, value projection matricies for all attention heads. These are the matricies
        # that transform and input tensor x into key, query, and value vectors, all of the same size
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        assert (hidden_size // n_heads) % 2 == 0, "d = hidden_size / n_heads must be even for RoPE"
        # Store the pre-computed rope_cache values for use later, block_size = max sequence input length
        self.register_buffer("rope_cache", get_rope_cache(hidden_size // n_heads, block_size))

        # Add dropout regularization layers
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

        # Add an output projection later
        self.final_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_kv: torch.Tensor, x_kv_masks: torch.Tensor, x_q: torch.Tensor):
        """
        Computes the forward pass through the cross-attention layer.

        x_kv is (batch_size, src_len, hidden_size) and are the encoder hiddens which are to be used to
        generate the key and value vectors (i.e. K = x_kv @ W_k) hence the underscore kv.

        x_q is (batch_size, tgt_len, hidden_size) and are the decoder hiddens which are to be used to
        generate the query vectors (i.e. Q = x_q @ W_q), hence the underscore q.

        The purpose of this cross-attention block is to alter the decoder hiddens (i.e. the decoded seq toekn
        vectors) by attending to the hiddens from the full encoder output.

        Parameters
        ----------
        x_kv : torch.Tensor
            An input tensor of size (batch_size, src_len, hidden_size) from the EncoderBlock that are used to
            generate key and value vectors.
        x_kv_masks : torch.Tensor
            ## TODO: Add more here
        x_q : torch_tensor
            An input tensor of size (batch_size, tgt_len, hidden_size) from the prior sub-layer of the
            masked multi-head attention decoder block i.e. the decoder hiddens to be altered by the info
            contained in the input seq processed by the encoder.

        Returns
        -------
        y : torch.Tensor
            An output tensor of the same size as the original input x_q (batch_size, tgt_len, hidden_size)
            tensor where each token vector now be altered by cross-attention with the encoder hiddens.
        """
        B_kv, T_kv, H_kv = x_kv.size() # Get the shape of the x_kv input from the encoder
        B_q, T_q, H_q = x_q.size() # Get the shape of the x_q input from the decoder prior layer
        assert B_kv == B_q, "Batch sizes do not match"
        assert H_kv == H_q, "x_kv and x_q have different hidden sizes"
        B, H = B_kv, H_kv # Short-hand notation, should be the same for both

        # Calculate query, key, and value vectors for all heads in this batch. Split the vectors along the
        # last dimension into head_size (hs) = hidden_size / n_heads equal sized segements, Re-roder the dims.
        hs = H // self.n_heads  # Let hs be the size of vectors used by each head
        K = self.W_k(x_kv).view(B, T_kv, self.n_heads, hs).transpose(1, 2) # (B, nh, T_kv, hs)
        Q = self.W_q(x_q).view(B, T_q, self.n_heads, hs).transpose(1, 2) # (B, nh, T_q, hs)
        V = self.W_v(x_kv).view(B, T_kv, self.n_heads, hs).transpose(1, 2) # (B, nh, T_kv, hs)

        # Apply the rotary positional embeddings to the key and value vectors before computing the dot prod
        Q = apply_rope(Q, self.rope_cache)
        K = apply_rope(K, self.rope_cache)

        # Compute cross-attention scores by taking the dot product of all key and value vectors
        # Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv) do matrix mult on the last
        # 2 dimensions to figure out the output size (_, _, T_q, hs) x (_, _, hs, T_kv) -> (_, _, T_q, T_kv)
        att_scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(hs)) # (B, nh, T_q, T_kv)

        ## TODO: Apply the x_kv_masks filtering so that we do not attend to the padding tokens of the input
        ## sequence



        # Now we have a matrix for each sentence and each head that is (T_q x T_kv) which are the attention
        # scores of each query token (decoder seq tokens) to each of the key tokens (encoder seq tokens)
        # Next we will apply a softmax normalization to the last layer i.e. for each query vector, normalize
        # to a prob dist the attention scores from each of the encoder tokens
        att_scores = F.softmax(att_scores, dim=-1) # Apply softmax normalization along the last dimension
        att_scores = self.attn_dropout(att_scores) # Apply dropout regularization
        # Compute the weighted avg value vector for each decoder seq element using the attention scores
        y = att_scores @ V # (B, nh, T_q, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T_q, hs)

        y = y.transpose(1, 2).contiguous().view(B, T_q, H) # Re-assemble all head outputs side by side

        # Apply a final linear projection and dropout before returning
        y = self.resid_dropout(self.final_proj(y))
        return y # (batch_size, tgt_len, hidden_size)


class DecoderBlock(nn.Module):
    """
    Decoder Transformer Attention Block that computes:
        x = LayerNorm(SelfAttention(LayerNorm(x)) + x)
        x = LayerNorm(CrossAttention(enc_hiddens, x) + x)
        x = LayerNorm(MLP(x) + x)
        return x
    """
    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size # The size of each latent vector represenation of each token
        self.n_heads = n_heads # The number of attention heads
        self.block_size = block_size # The max length input token sequence (the max possiable tgt_len)
        self.dropout_rate = dropout_rate # Dropout probability during training

        ######################################################################################################
        ### Define the model architecture

        self.ln1 = nn.LayerNorm(hidden_size) # Normalization of x going into the self-attention mechanism
        self.self_attn = SelfAttentionLayer(hidden_size, n_heads, block_size, dropout_rate, causal=True)
        self.ln2 = nn.LayerNorm(hidden_size) # Normalization of x coming out of the self-attention mechanism

        self.cross_attn = CrossAttentionLayer(hidden_size, n_heads, block_size, dropout_rate)
        self.ln3 = nn.LayerNorm(hidden_size) # Normalization of x coming out of the cross-attention mechanism

        self.mlp = nn.Sequential(
            # Each attention-adjusted token vector comes in with size hidden_size, then we apply a FFNN to it
            # with a hidden size of 4 x hidden_size
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(), # Gaussian Error Linear Unit activation function, non-linearity
            nn.Linear(4 * hidden_size, hidden_size), # Apply linear layer to project back down to hidden_size
            nn.Dropout(dropout_rate), # Apply dropout for regularization
        )
        self.ln4 = nn.LayerNorm(hidden_size) # Normalization of x coming out of the MLP

    def forward(self, x: torch.Tensor, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor) -> torch.Tensor:
        """
        Defines the forwards pass evaluation through this decoder transformer attention block which involves:
            x = LayerNorm(SelfAttention(LayerNorm(x)) + x)
            x = LayerNorm(CrossAttention(enc_hiddens, x) + x)
            x = LayerNorm(MLP(x) + x)
            return x

        Residual connections are used in this model.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of size (batch_size, tgt_len, hidden_size) containing the token vectors for each
            token in each sentence for all sentences in the batch.
        enc_hiddens : torch.Tensor
            ## TODO: ADD MORE HERE
        enc_masks : torch.Tensor
            ## TODO: ADD MORE HERE

        Returns
        -------
        x : torch.Tensor
            Returns a tensor of the same size as the input x tensor i.e. (batch_size, tgt_len, hidden_size)
            which contains the context-rich latent representations of each word in the decoded sequence. Each
            target sequence token attends to all of the encoder tokens and also the ones at or before the
            current token in the decode sequence.
        """
        # Pass x from the decoder through masked multi-headed self-attention, then add to self and norm
        x = self.ln2(self.self_attn(self.ln1(x)) + x)
        # Pass x then through the multi-headed cross-attention block with the encoder outputs, then add to
        # self and norm for residual connections
        x = self.ln3(self.cross_attn(enc_hiddens, enc_masks, x) + x)
        # Pass x through the multi-layer perceptron (NLP) FFNN later, then add to self and norm again
        x = self.ln4(self.mlp(x) + x)
        return x # (batch_size, tgt_len, hidden_size)








# TODO: At some point we should be able to do some key-query caching to make things faster, but we'll see I guess
# I suppose with the transformer, does it need to be sequentially evaluated? Yeah I think it does, especially for
# prediction. Although I guess for training we could make things a bit more parallelized, but there would be a lot
# of redundant key-query computations?



class MHTM(NMT):
    """
    Multi-Headed Transformer Model (MHTM) comprised of:
        - A bi-directional, multi-headed self-attention encoder
        - A multi-headed attention decoder with cross-attention to the encoder attention scores
        - Uses rotary positional embeddings (RoPE)
        - Model archtecture modeled off the encoder-decoder transformer model described in the famous paper
          "Attention is All You Need"
    """

    def __init__(self, embed_size: int, hidden_size: int, num_layers: int, n_heads: int,
                 dropout_rate: float, block_size: int, vocab: Vocab, *args, **kwargs):
        """
        Bi-directional, multi-headed self-attention encoder + multi-headed attention decoder with
        cross-attention.

        Parameters
        ----------
        embed_size : int
            The size of the word vector embeddings (dimensionality).
        hidden_size : int
            The size of the hidden states (dimensionality) used by the encoder and decoder LSTM.
        num_layers : int
            The number of transformer attention blocks to use in the encoder and decoder.
        n_heads : int
            The number of attention heads to use in both the encoder and decoder. Note, that for this model,
            embed_size == hidden_size and n_heads should evenly divide into hidden_size and also
            hidden_size // n_heads should be even (for RoPE).
        dropout_rate : float
            The dropout rate used in the attention mechanism, specifies the probability of a node being
            switched off during training, something around 0.2 is typical.
        block_size : int
            The max number of input tokens to be used as context in the transformer attention blocks.
        vocab : Vocab
            A Vocabulary object containing source (src) and target (tgt) language vocabularies.
        """
        super(MHTM, self).__init__()
        self.embed_size = embed_size  # Record the word vector embedding dimensionality
        self.hidden_size = hidden_size # Record the size of the hidden states i.e. key, value, query vectors
        self.block_size = block_size # The max number of input tokens into the attention blocks
        assert self.embed_size == self.hidden_size, "embed_size must be equal to hidden_size"
        assert isinstance(num_layers, int) and (1 <= num_layers <= 5), "num_layers must be an int [1, 5]"
        self.num_layers = num_layers # Record how many attention block layers to use
        assert self.hidden_size % n_heads == 0, "hidden_size must be evenly divisible by n_heads"
        self.d = self.hidden_size // n_heads # Record the dimensionality used per head i.e. block_size
        assert self.d % 2 == 0, "hidden_size // n_heads must result in an even integer"
        self.dropout_rate = dropout_rate # Record the dropout rate parameter
        self.vocab = vocab # Use self.vocab.src_lang and self.vocab.tgt_lang to access the language labels
        self.name = "MHTM"
        # self.lang_pair = (vocab.src_lang, vocab.tgt_lang) # Record the language pair of the translation


        ######################################################################################################
        ### Define the model architecture

        # Create a word-embedding mapping for the source language vocab
        self.source_embeddings = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=self.embed_size,
                                              padding_idx=vocab.src['<pad>'])

        # Create a word-embedding mapping for the target language vocab
        self.target_embeddings = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=self.embed_size,
                                              padding_idx=vocab.tgt['<pad>'])

        # Transformer blocks for the encoder, we will pass the input sequence through these layers to obtain
        # a deep, context-rich embedding representation of each word in the sequence. These are bi-directional
        # i.e. non-causal, multi-headed self-attention blocks
        self.encoder = nn.Sequential(*[EncoderBlock(hidden_size, n_heads, block_size, dropout_rate)
                                       for _ in range(self.num_layers)])

        # Transformer blocks for the decoder, we will pass the decoded values available so far through these
        # blocks and combine them with the encoder representations to create vectors that can be used to
        # generate Y_hat distributions across the vocab at each time step using casual cross-attention
        self.decoder = nn.Sequential(*[DecoderBlock(hidden_size, n_heads, block_size, dropout_rate)
                                       for _ in range(self.num_layers)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.ln_final = nn.LayerNorm(self.hidden_size) # The final layer normalization before prediction
        # One final linear layer used to compute the final y-hat distribution of probabilities over the
        # entire vocab for what word token should come next
        self.target_vocab_proj = nn.Linear(self.hidden_size, len(vocab.tgt), bias=False)





        ## TODO: Clean up below
        ### Need to do some zeroing out for padding in the transformer model forward pass, right? We don't
        ### want any attention scores attending to padding vectors so we should set all of those dot products
        ### to 0
        self.decoder

        # What needs to happen? We get some input sequence in x (batch_size, src_len) in the source language.
        # We pass that into the embedding layer to convert to word vectors and get (batch_size, src_len, embed_size)
        # Then we pass that x into the encoder transformer layers. The self-attention calcs are done and we get
        # out a tensor of context-rich latent embeddings of size (batch_size, src_len, embed_size)
        # This pre-processing should be done at the very start before any decoding. Now we have those deep
        # latent encoder representations called x_embed

        # Okay then when it comes predict, the first token we always give the network is <s> to start a new
        # sentence. That token and all prior decoded tokens go into the network in a different entry point
        # we feed in (batch_size, tgt_len) -> pass it through the embedding layer and get a tensor of size
        # (batch_size, tgt_len, embed_size) where tgt_len is the length of the prior tokens so far

        # Okay Then this tensor of prior decoded tokens is passed into a SelfAttention, then those outputs are
        # combined with the encoder outputs (which are the same each time) and fed into another attention
        # mechanism, this one being Casual Cross Attention. TBD how exactly this works under the hood, I guess
        # we use only the input x from the decoder model thus far as the query vectors and the key and value
        # matrices are larger and contain all the vectors from the encoder + decoder so far so we can attend
        # each decoder token to everything currently decoded so far + the stuff in the encoder blocks

        # Then that stuff combined goes into the feed forward NN and we repeat this operation a few times
        # potentially and then at the very end we use the last token attention value through a linear layer
        # and softmax to generate our yhat distribution.

        # Decoder head - this part will be more complicated and is something we'll have to think about more
        # carefully. Weill need a step function and such

        ## TODO: This needs work as well, we need to attend to the exisitng output seq generated so far and
        ## combine that with the attention outputs of the encoder blocks, this decoder below is too somple
        ## and just predicts 1 word ahead given the context input. Our network here has to be seq2seq so we
        ## need to attend to both the input seq and also what we've generated so far





        self.apply(self._init_weights)

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")





        ### TODO: STOPPED HERE - NEED TO UPDATE BELOW
        ## Need to add the correct layers, look at A5 for some inspiration, might need to define some other
        ## classes as well, that could be helpful. Add in the RoPE embeddings and pre-cached values etc.





        # This is the bi-directional LSTM encoder that takes in the word embedding for each input word of the
        # source language (each of size embed_size) and outputs a hidden state vector of size hidden_size and
        # a cell memory vector (also of size hidden_size)
        # self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bias=True,
        #                       batch_first=True, bidirectional=True)

        # # This is the LSTM decoder section of the model that is one-directional since it is making the y_hats
        # # Takes in the word embedding of the prior predicted output word and rolls the prediction forward to
        # # produce the predicted translation in the output language. This layer cannot be bi-directional since
        # # we make y-hat predictions sequentially from left-to-right. The inputs are a concatenation of the
        # # word embedding of the prior predicted word and the final context vector from the encoder
        # # The inputs are a concatenated vector of the input word y_t and the prior hidden state
        # self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size, hidden_size=hidden_size, bias=True)

        # # Takes in the concatenated input of:
        # #   [last hidden_state of the forward LSTM] + [first hidden state of the backward LSTM]
        # # which is of size hidden_size * 2 and outputs h_0 for the decoder to initialize it
        # self.h_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)

        # # Takes in the concatenated input of:
        # #   [last cell state of the forward LSTM] + [first cell state of the backward LSTM]
        # # which is of size hidden_size * 2 and outputs c_0 for the decoder to initialize it
        # self.c_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)

        # # This is used to compute e_{t,i} = h_{t}^{dec}^T @ W_{attProj} @ h_{i}^{enc} where h_{t}^{dec} is
        # # the current hidden state of the decoder LSTM at time step t which changes each step and h_{i}^{enc}
        # # is a concatenation of the forward and reverse hidden states of the bi-directional LSTM of the ith
        # # input word from the original text. These 2 together are used to compute the attention scores
        # self.att_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)

        # # Used to compute v_{t} = W_{u} @ u_{t} where u_{t} is the concatenation of h_{t}^{dec} i.e. the
        # # hidden state of the decoder LSTM at time stamp t and also a_{t} which is the attention output
        # # which was the weighted sum of the encoder hidden states for each input word weighted by their
        # # softmax attention probability scores. So we're making y_hat predictions based on the attention
        # # score outputs (which are 2h in size because they're based on bi-directional encoder hiddens) and
        # # the current hidden state of the LSTM decoder (which is also of size h)
        # self.combined_output_projection = nn.Linear(in_features=hidden_size * 3, out_features=hidden_size,
        #                                             bias=False)

        # # This is used to compute the final y-hat distribution of probabilities over the entire vocab for what
        # # word token should come next. I.e. y_hat = softmax(W_{vocab} @ o_{t}) where y_hat is a length |V|
        # # vector and o_{t} = dropout(tanh(v_{t})) using the v_t from above
        # self.target_vocab_projection = nn.Linear(in_features=hidden_size, out_features=len(vocab.tgt),
        #                                          bias=False)
        # # Create a dropout layer for the attention with a probability of dropout_rate of an element being
        # # zeroed during training, this helps with regularization in the network training
        # self.dropout = nn.Dropout(p=dropout_rate)


    def generate_sentence_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Generates sentence masks identifying which are pad tokens so that the attention scores computed from
        the encoder hidden states that are not real input words.

        Parameters
        ----------
        enc_hiddens : torch.Tensor
            A tensor of encoder hidden states of size (b, src_len, 2*h) where b=batch_size, src_len=max source
            sentence length within this batch and h=hidden size. We have 2*h since the encoder is
            bi-directional.
        source_lengths : List[int]
            A list of ints denoting how long each source input sentence is i.e. all tokens beyond are padding.

        Returns
        -------
        torch.Tensor
            A tensor of sentence masks of size (b, src_len).
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1 # Set the padding word tokens to have 1s rather thans 0s
        return enc_masks.to(self.device)

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        This method passes a tensor of input source language sentences (that have already been padded to all
        the same length) into the encoder part of the transformer model to obtain deep latent representations
        of each input sub-word token. These encoder token representations will be supplied to the decoder for
        producing a roll out sequence.

        Parameters
        ----------
        source_padded : torch.Tensor
            A tensor of padded source sentences of size (b, src_len) encoded as word id integer values
            where b = batch_size and src_len = the max sentence length in the batch of source sentences.
            These have been pre-sorted in order of longest to shortest sentence.
        source_lengths : List[int]
            A list containing the length of each input sentence without padding in the batch. This list is of
            length b with max(source_lengths) == src_len.

        Returns
        -------
        enc_hiddens : torch.Tensor
            A tensor of hidden states from the encoder of shape (b, src_len, h) where h = hidden_size.
        """
        # Convert input sentences (already padded to all be the same length src_len) stored as a tensor of
        # word_ids of size  (batch_size, src_len) into a tensor of word embeddings (b, src_len, embed_dim)
        X = self.source_embeddings(source_padded) ## TODO: Figure out if this is what we want to do here

        # pack_padded_sequence takes these padded sequences and their corresponding lengths as input. It then
        # transforms them into a "packed" format that only includes the non-padded portions of the sequences.
        # By removing unnecessary computations on padded portions of the sequences, the computations are
        # faster to perform. The size of X is (src_len, batch_size, embed_dim) so the batch_size is the first
        # so we set batch_first=True.
        X_packed = pack_padded_sequence(X, lengths=source_lengths, batch_first=True, enforce_sorted=True)

        # Pass in the packed sentense of sentences of size (batch_size, src_len, embed_dim). This returns
        # a tensor of size (batch_size, src_len, hidden_size) containing the context-rich vector
        # representations of each input sequence sub-word token i.e. the deep latent representation of each
        enc_hiddens = self.encoder(X_packed) # (batch_size, src_len, hidden_size)

        # Apply the pad_packed_sequence function to the outputs which is the inverse operation of
        # pack_padded_sequence. This function returns a tuple of a Tensor containing the padded sequence, and
        # a Tensor containing the list of lengths of each sequence in the batch. We only care about the first
        # element, not the lengths of each sequence.
        enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first=True)[0] # (b, src_len, h)
        return enc_hiddens # Return the latent representations of the input sorce text tokens (b, src_len, h)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """
        Takes a mini-batch of source and target sentences, compute the log-likelihood of the target sentences
        under the language models learned by the NMT system. Essentially, pass the soruce words into the
        encoder, then make the first prediction using the decoder. Compare that prediction to the actual
        first word of the target language true translation and compute a log-likelihood loss. Feed the true y
        of the target language into the decoder (instead of the y-hat predicted at this time step) for the
        next time-step.

        The length of source and target must be equal.

        Parameters
        ----------
        source : List[List[str]]
            A list of input source language sentences i.e. a list of sentences where each sentence is a list
            of sub-word tokens.
        target : List[List[str]]
            A list of target source language sentences i.e. a list of sentences where each sentence is a list
            of sub-word tokens wrapped by <s> and </s>.

        Returns
        -------
        scores : torch.Tensor
            A Tensor of size (batch_size, ) representing the log-likelihood of generating the target
            sentence for each example in the input batch. This is used for back-prop in gradient descent.
            This computes the loss of the model over the input batch.
        """
        assert len(source) == len(target), "The number of source and target sentences must be equal"
        source_lengths = [len(s) for s in source]  # Compute the length of each input source sentence

        # Convert from a list of lists into tensors of word_ids where src_len is the max length of sentences
        # among the input source sentences and tgt_len is the max length of sentences among the outpu
        # sentences and b = batch_size i.e. how many sentences in total (which should be equal in both)
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor (b, src_len)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor (b, tgt_len)

        # Call the encoder on the padded source sentences which will be re-used for each step of the decoder
        enc_hiddens = self.encode(source_padded, source_lengths)

        # Generate a set of token masks for each source sentence so that we don't attend to padding tokens
        # in the decoder when computing attention scores
        enc_masks = self.generate_sentence_masks(enc_hiddens, source_lengths) # (b, src_len)

        # Call decode to run the full set of decoder operations which passes in the true tgt word tokens to
        # the decoder at each timestep regardless of what the model would predict. This call to decode returns
        # the final sub-word token hidden state at each time-step of the decoder i.e. what we would use to
        # make next-word predictions at each time step
        decoder_outputs = self.decode(enc_hiddens, enc_masks, target_padded) # (b, tgt_len, h)

        # Compute the prob distribution over the vocabulary for each prediction timestep from the decoder,
        # decoder_outputs is what would be used at each timestep, we can process them all at once since we
        # have them all here at once as one big tensor of size (b, tgt_len, V)
        prob = F.log_softmax(self.dropout(self.ln_final((self.target_vocab_proj(decoder_outputs)))), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text i.e. the padding, create a bool
        # mask of 0s and 1s by checking that each entry is not equal to the <pad> token
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating the true target words provided in this example i.e. compute
        # the cross-entropy loss by pulling out the model's y-hat values for the true target words. For each
        # word in each sentence, pull out the y_hat prob associated with the true target word at time t.
        # prob is (b, tgt_len, V) and describes the probability distribution over the next word after the
        # current time step t. I.e. the first Y_t token is <s> and the first y_hat is the distribution of
        # what the model thinks should come afterwards. Hence prob[:, :-1, :] aligns with the true Y_t words
        # target_padded[:, 1:]. tgt_len includes <s> at the start and </s> at the end. We don't want to
        # include the prob of <s> but we do want to include the prob of predicting </s> to end the sentence.
        target_words_log_prob = torch.gather(prob[:, :-1, :], index=target_padded[:, 1:].unsqueeze(-1),
                                             dim=-1).squeeze(-1) # (b, tgt_len - 1) result
        # Zero out the y_hat values for the padding tokens so that they don't contribute to the sum
        target_words_log_prob = target_words_log_prob * target_masks[:, 1:] # (b, tgt_len - 1)
        return target_words_log_prob.sum(dim=1) # Return the log prob per sentence

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               target_padded: torch.Tensor) -> torch.Tensor:
        """
        Computes decoder output hidden-state vectors for each word in each batch of target sentences i.e. runs
        the decoder to generate the output sequence of hidden states for each word of each sentence while
        using the true Y_t words provided in the target translation as inputs at each time step instead of the
        prior Y_hat_values provided from the prior step. This method is used for training only. In the end,
        we return a (batch_size, tgt_len, hidden_size) vector which can be used with target_vocab_projection
        and softmax to generate the distribution of next time step predicted tokens according to the model.

        Parameters
        ----------
        enc_hiddens : torch.Tensor
            A tensor of size (b, src_len, h) of hidden state from the encoder.
        enc_masks : torch.Tensor
            A tensor of sentence masks (0s and 1s) for masking out the padding tokens of size (b, src_len),
            where a value of 1 denotes the presence of a padding token.
        target_padded : torch.Tensor
            Gold-standard padded target sentences of size (b, tgt_len) i.e. good translations of the inputs.

        Returns
        -------
        combined_outputs : torch.Tensor
            Returns a tensor of combined outputs that are used to make y_hat predictions of size
            (b, tgt_len, h) which incorporates info from the encoder hiddens and previous decoded tokens
            using multi-headed casual cross-attention.
        """
        # target_padded = target_padded[:, :-1] # Remove the <END> token for max length sentences

        # Construct a tensor Y of observed translated sentences with a shape of (b, tgt_len, e) using the
        # target model embeddings where tgt_len = maximum target sentence length and e = embedding size.
        # We use these actual translated words in our training set to score our model's predictions
        # Convert from word_ids (b, tgt_len) to word vectors ->  b, tgt_len, e)
        Y = self.target_embeddings(target_padded) # (b, tgt_len, e)

        # Pass the full Y sequence of words (gold-standard translations) into the decoder model for processing
        # in parallel. Also provide the enc_hiddens that will be used for cross-attention calculations. We
        # can compute this all at once by using masking to make sure that decoder tokens at time step t
        # attend only to those decoder tokens prior.
        decoder_outputs = self.decoder(Y, enc_hiddens, enc_masks) # (b, tgt_len, h)

        # This produces an output vector of values at each time step that can then be used to compute logits
        # for each possiable output word in the vocab after passing them through a linear projection layer
        return decoder_outputs # (batch_size, tgt_len, hidden_size)

    ### TODO: Need to go in an make sure that this works self.decoder(Y, enc_hiddens, enc_masks)
    # i.e. the forward pass accepts these 3 arguments and uses them correctly. We do not want to attend to
    # words that are not present i.e. padding tokens so use enc_masks as well






### TODO: Everything below needs updating






    def step(self, Ybar_t: torch.Tensor, dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor, enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """
        TODO: This needs work
        Computes one forward step of the LSTM decoder, returns the updated decoder state (hidden, cell),
        a combined output tensor used to make y_hat predictions and e_t attention scores as a distribution.

        Parameters
        ----------
        Ybar_t : torch.Tensor
            A concatenated tensor of [Y_t, o_prev], with shape (b, e + h). This is the input for the decoder,
            where b = batch size, e = embedding size, h = hidden size.
        dec_state : Tuple[torch.Tensor, torch.Tensor]
            A tuple of tensors both with shape (b, h), where b = batch size, h = hidden size. The first tensor
            is the decoder's prev hidden state and the second tensor is the decoder's prev cell.
        enc_hiddens : torch.Tensor
            Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
            src_len = maximum source length, h = hidden size.
        enc_hiddens_proj : torch.Tensor
            Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is shape (b, src_len, h), where
            b = batch size, src_len = maximum source length, h = hidden size.
        enc_masks : torch.Tensor
            Tensor of sentence masks shape (b, src_len), where b = batch size, src_len is maximum source
            length.

        Returns
        -------
        dec_state : Tuple[torch.Tensor, torch.Tensor]
            Tuple of tensors representing the decoder's new hidden state and cell state, each of size (b, h).
        O_t : torch.Tensor
            Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size. This
            incorporates all the new info (Y_t, O_(t-1), attention scores, prior hidden state etc.)
        e_t : torch.Tensor
            A tensor of shape (b, src_len) containing the computed attention score distribution.
        """
        # Apply the decoder to Ybar_t and dec_state to obtain the new dec_state = decoder hidden state and
        # decoder cell outputs
        dec_state = self.decoder(Ybar_t, dec_state) # Update the decoder state (hidden_t, cell_t)
        dec_hidden, dec_cell = dec_state # Unpack into components
        # Compute the attention scores e_t, a tensor of size (b, src_len). Here we are computing:
        # e_ti = (h_{t}^{dec}.T) @ W_{attProj} @ h_{i}^{enc} with the last 2 terms already stored in
        # enc_hiddens_proj. This computation is to be done in batches. dec_hidden is size (b, h) and
        # enc_hiddens_proj is size (b, src_len, h) and we want an output of size (b, src_len) i.e. for each
        # batch (sentence), a probability distribution over all the words in each sentence (src_len).
        # We want to take dec_hidden for each sentence and multiply it with enc_hiddens_proj for each
        # sentence which would result in a (1 x src_len) tensor per sentence. Use torch bmm to perform this
        # batch matrix multiplication: (b x src_len x h) @ (b x h x 1) = (b x src_len x 1)
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2) # Squeeze to remove last dim

        if enc_masks is not None: # Set e_t to -inf where enc_masks has a 1, since 1 indicators padding
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=-1) # Apply softmax to e_t within each sentence to yield alpha_t
        # Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the attention output
        # vector, a_t.
        # - alpha_t is shape (b, src_len)
        # - enc_hiddens is shape (b, src_len, 2h)
        # - a_t should be shape (b, 2h)
        # (b x 1 x src_len) @ (b x src_len x 2h) = (b x 1 x 2h)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(dim=1)
        # Concatenate dec_hidden with a_t to compute tensor U_t
        U_t = torch.cat(tensors=(dec_hidden, a_t), dim=1) # dec_hidden is (b x h), a_t is (b x 2h) = (b x 3h)
        # Apply the combined output projection layer to U_t to compute tensor V_t
        V_t = self.combined_output_projection(U_t)
        # Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        O_t = self.dropout(torch.tanh(V_t))

        # Returns the updated decoder state, the O_t combined outputs and the attention scores e_t
        return dec_state, O_t, e_t


    def greedy_search(self, src_sentences: List[List[str]], k_pct: float = 0.1,
                      max_decode_lengths: Union[List[int], int] = None) -> List[List[Union[List[str], int]]]:
        """
        TODO: This needs work
        Given a list of source sentences (where each is a list of sub-word tokens), this method performs
        greedy search yielding a translation in the target langauge by sequentially predicting the next token
        by randomly sampling among the sub-words that make up the top k% of the probability distribution
        among all possiable output sub-word tokens, according to their relative probabilities. src_sentences
        is processed in batches to speed up calculations, but this computation can be slow for large sets of
        input source sentences.

        If k_pct is let as None, then the most probable sub-word token is always choosen and the output has no
        variation from one call to another. By default, k_pct is set to 10% which means that the model will
        sample from the sub-words that make up the top 10% of the probability distribution at each prediction
        step. k_pct must be a float value (0, 1]. E.g. if the most likely work token has a prob of 50% and
        k_pct = 10%, then it will be selected with probability 100%. If instead the top 2 most probably tokens
        have probs of 7% and 5% respectively, then the next token will be sampled from just those 2 with more
        of a chance given the the first due to it's higher relative probability.

        max_decode_lengths specifies the max length of the translation output for each input sentence. If an
        integer is provided, then that value is applied to all sentences. If not specified, then the default
        value will be len(src_sentence) * 1.2 for each src_sentence in src_sentences. The values of
        max_decode_lengths are capped at 200 globally.

        Parameters
        ----------
        src_sentences : List[List[str]]
            A list of input source sentences where each is a list of sub-word tokens.
            e.g. ['▁Wo', '▁ist', '▁die', '▁Bank', '?']
        k_pct: float
            This method builds an output transltion by sampling among the eligible candidate sub-word tokens
            according to their relative model-assigned probabilities at each time step. If k_pct is set to
            None, then the most likely word is always choosen (100% greedy). Otherwise, the most probably
            words making up k_pct of the overall probability distribution are used. As k_pct is lowered, the
            variance of the model's outputs increases.
        max_decode_lengths : Union[List[int], int], optional
            The max number of time steps to run the decoder unroll sequence for each input sentence. The
            output machine translation produced for each sentence will be capped in length to a certain
            amount of sub-word tokens specified here. The default is 1.2 * len(src_sentence) and all values
            must be <= 250.

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
        if k_pct is not None:  # If not None, then perform data-validation
            assert 0 < k_pct <= 1.0, "k_pct must be in (0, 1] if not None"
        if max_decode_lengths is None: # Default to allow for 20% more words per sentence if not specified
            max_decode_lengths = [int(len(s) * 1.2) for s in src_sentences]
        if isinstance(max_decode_lengths, int): # Convert to a list if provided as an int
            max_decode_lengths = [max_decode_lengths for i in range(b)]
        max_decode_lengths = max_decode_lengths.copy() # Copy to avoid mutation
        for i, n in enumerate(max_decode_lengths): # Check all are integer valued and capped at 250
            assert isinstance(n, int) and n > 0, "All max_decode_lengths must be integers > 0"
            max_decode_lengths[i] = min(n, 250)

        msg = "src_sentences and max_decode_lengths must be the same length"
        assert len(max_decode_lengths) == len(src_sentences), msg

        # Figure out the sort order to arrange the sentences in decreasing length order
        argsort_idx = np.argsort([len(s) for i, s in enumerate(src_sentences)])[::-1]
        new_to_orig_idx = {int(x): i for i, x in enumerate(argsort_idx)} # Reverse the mapping backwards
        src_sentences = [src_sentences[idx] for idx in argsort_idx] # Re-order by sentence length (desc)

        with torch.no_grad():  # no_grad() signals backend to throw away all gradients

            # Convert the input source sentence into a tensor object of size (b, src_len) of word indices
            src_sentence_tensor = self.vocab.src.to_input_tensor(src_sentences, self.device) # (b, src_len)

            # Pass it through the encoder to generate the encoder hidden states for each word of each input
            # sentence and also the  the decoder initial hidden state (h of t minus 1) for each sentence
            enc_hiddens, dec_init_state = self.encode(src_sentence_tensor, [len(s) for s in src_sentences])
            # enc_hiddens (b, src_len, h*2), dec_init_state is a tuple of 2 vectors each of size (b, h)

            dec_state = dec_init_state # Tuple((b, h), (b, h)) = (hidden, cell)
            o_prev = torch.zeros(b, self.hidden_size, device=self.device)  # Initialize as all zeros (b, h)
            enc_hiddens_proj = self.att_projection(enc_hiddens) # Outputs a tensor that is (b, src_len, h)
            # (b, src_len) encode where the padding tokens are i.e. use 1s to denote right-padding
            enc_masks = self.generate_sentence_masks(enc_hiddens, [len(s) for s in src_sentences])

            # Create output translations for each input sentence, begin with the start-of-sentence begin
            # token and also record the negative log likelihood of the sentence
            mt = [[['<s>'], 0] for _ in range(b)] # Machine translations

            # Use the last output word Y_hat_(t-1) as the next input word (Y_t) going into the decoder, we
            # always start with the <s> sentence start token for each output translation
            Y_t = torch.tensor([self.vocab.tgt[mt[i][0][-1]] for i in range(b)],
                               dtype=torch.long, device=self.device) # (b, )

            # Iterate until we've a complete output translations or we reach the max output len
            finished = 0 # Track how many output translation sentences are finished
            finished_flags = [0 for i in range(b)] # Mark which sentences have been completed

            while finished < b: # Iterate until all output translations are finished generating
                Y_t_embed = self.target_embeddings(Y_t) # (b, embed_size) convert to a word vector

                # Compute an updated hidden state using the last y_hat and the prior hidden state
                Ybar_t = torch.cat(tensors=(Y_t_embed, o_prev), dim=1) # (b, e + h)
                dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
                # dec_state is a length 2 tuple with (b, h) for the hidden state and cell state of the decoder
                # at the current time-step for each sentence, o_t is (b, h), e_t is (b, src_len)

                # Compute the log probabilities over all possiable next target words using the last hidden
                # layer i.e. the one that is to be fed to self.target_vocab_projection, gives us (b, |V|)
                log_p_t = F.log_softmax(self.target_vocab_projection(o_t), dim=-1) # (b, |V|)

                if k_pct is None: # Select the word with the highest modeled probability always
                    # Find which word has the highest log prob for each sentence, idx = word_id in the vocab
                    Y_hat_t = torch.argmax(log_p_t, dim=1) # (b, ) the most probably next word_id for each
                else: # Randomly sample from the sub-words at or above the kth most probably percentile
                    prob_t = torch.exp(log_p_t) # Exponentiate to convert to a prob dist (b, |V|)
                    # Find what cutoff is required to make it into the words that collectively sum to form
                    # the top k percent of the probability distribution i.e. for a flat distribution there
                    # will be more words, for a more concentrated distribution, there will be fewer words that
                    # make the cut
                    Y_hat_t = torch.zeros(b, dtype=int) # Start off with all zeros
                    for i in range(b):
                        if finished_flags[i] == 0: # Compute if this sentence is not already finished
                            sorted_probs = prob_t[i, :].sort(descending=True) # Sort the probs of this dist
                            bool_vec = sorted_probs.values.cumsum(0) <= k_pct # The entries in the top k %
                            bool_vec[0] = True # Always have at least 1 entry set to true i.e. this happens if
                            # the most likely word has a higher prob than k
                            idx, prob = sorted_probs.indices[bool_vec], sorted_probs.values[bool_vec]
                            prob /= prob.sum() # Re-normalize to 1 and then sample to get the next prediction
                            Y_hat_t[i] = idx[prob.multinomial(num_samples=1, replacement=True).item()]
                        # Else leave the word_id as 0 which defaults to the padding token

                for i in range(b): # Record the next predicted word for each output translation
                    if finished_flags[i] == 0: # Record if this sentence is not already finished
                        mt[i][0].append(self.vocab.tgt.id2word[Y_hat_t[i].item()])
                        mt[i][1] += -log_p_t[i, int(Y_hat_t[i].item())] # Sum the log prob of y-hats
                        # Check if the translation has been complete i.e. we got a sentence stop token or the
                        # max decode length was reached for this sentence
                        if mt[i][0][-1] == "</s>" or len(mt[i][0]) - 1 == max_decode_lengths[i]:
                            # mt[i][0] is the list of output sub-word tokens, which beings with </s> for all
                            # so it is already length 1, so we subtract 1 to trigger when the output tokens
                            # added after </s> are max_decode_lengths[i]
                            finished += 1 # Record that 1 more sentence was finished
                            finished_flags[i] = 1 # Mark this sentence off as finished

                # Update relevant state variables for next iteration
                Y_t = Y_hat_t # For next iter, set the current y_hat output as the next y (b, )
                o_prev = o_t # Update the combined outputs
                # dec_state was already updated in the step above so we do not need to do anything further

        # Re-order before returning to re-instate the original sentence ordering
        return [mt[new_to_orig_idx[idx]] for idx in range(len(mt))]


    def beam_search():
        # TODO: Finish building out a beam-search method here
        pass

    @classmethod
    def load(cls, model_path: str):
        """
        TODO: This needs work
        Method for loading in model weights saved locally to disk.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = cls(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, model_path: str):
        """
        TODO: This needs work
        Method for saving the model to a file.
        """
        # print(f"Saving model parameters to {model_path}", file=sys.stderr)
        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)
