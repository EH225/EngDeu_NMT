#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the source code of the encoder-decoder transformer model (EDTM).

The basis of this code comes from Stanford XCS224N Assignment 5 code, which was originally forked Andrej
Karpathy's minGPT project and has been heavily modified to fit the needs of this project.
"""
from __future__ import annotations
import sys, os
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import math, logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.util import NMT
from vocab.vocab import Vocab
import util

logger = logging.getLogger(__name__)


############################
### RoPE Cache Functions ###
############################
# TODO: Section marker

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
    t_vals = t_vals.expand(hs // 2, block_size).transpose(0, 1)
    i_vals = torch.Tensor(range(1, hs // 2 + 1))
    theta_i = (1 / 10000) ** (2 * (i_vals - 1) / hs)
    theta_i = theta_i.expand(block_size, hs // 2)
    rope_cache = (t_vals * theta_i)  # Compute t * theta for each (block_size, hs/2)
    rope_cache = torch.dstack([rope_cache, rope_cache])  # d-Stack to get 2 last dims
    rope_cache[:, :, 0] = torch.cos(rope_cache[:, :, 0])  # Compute cos(t theta) in idx 0 of the last dim
    rope_cache[:, :, 1] = torch.sin(rope_cache[:, :, 1])  # Compute sin(t theta) in idx 1 of the last dim
    return rope_cache  # (block_size, nh / 2, 2)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor, pos_idx: int = None) -> torch.Tensor:
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
    pos_idx : int
        If specified, then x should be (batch_size, nheads, T=1, hs) and have only 1 timestep length, then
        this function will apply RoPE to it according to the pos_idx provided i.e. so instead of always
        treating it as being at positional index 0, we can specify what positional index instead.

    Returns
    -------
    x_rotated : torch.Tensor
        Returns x but with its last dimension rotated according to RoPE. Same dimensions as x i.e.
        (batch_size, nheads, T, hs).
    """
    b, nh, T, hs = x.size()  # Get the dimensions of the input x tensor
    if pos_idx is not None:  # A particular positional index has been specified
        assert isinstance(pos_idx, int) and pos_idx >= 0, "pos_idx must be an int >= 0 if not None"
        rope_cache = rope_cache[pos_idx, :, :].unsqueeze(0)  # (seq_len, hs/2, 2)
    else:
        # rope_cache comes in as (block_size, hs/2, 2), truncate the end to match the length of x i.e. T
        rope_cache = rope_cache[:T, :, :]  # T <= block_size so this always works
    rope_cache = rope_cache.expand(
        (b, nh, T, hs // 2, 2))  # Extend the values as needed to fit the dimensions

    # The cosine values are in the 0 idx of the last dimension and the sine values are in idx 1 of the last
    # dimension, treat the cos values as real and the sin values as imaginary, create a vector of length hs/2
    # with each entry being cos(t theta_i) + i sin(t theta_i)

    # Compute cos(t theta_i) x_t^(i) - sin(t theta_i) x_t^(i+1)
    real_components = rope_cache[..., 0] * x[..., ::2] - rope_cache[..., 1] * x[..., 1::2]

    # Compute sin(t theta_i) x_t^(i) + cos(t theta_i) x_t^(i+1)
    img_components = rope_cache[..., 1] * x[..., ::2] + rope_cache[..., 0] * x[..., 1::2]

    x_rotated = torch.stack([real_components, img_components], dim=-1)  # (..., hs//2, 2)
    x_rotated = x_rotated.view(x.size())  # (b, nh, T, hs)

    # rotated_x = torch.cat((real_components.unsqueeze(-1), img_components.unsqueeze(-1)), dim=-1)
    # rotated_x = rotated_x.view(x.size())
    return x_rotated  # Same shape as the original x input tensor (batch_size, nh, T, hs)


########################
### Attention Layers ###
########################
# TODO: Section marker

class SelfAttentionLayer(nn.Module):
    """
    A multi-head self-attention layer using rotary positional embeddings (RoPE) with a linear projection at
    the end. This attention sub-layer can be bidirectional or causal using making. Input tokens attend to
    one another, this attention mechanism is used in both the encoder (bidirectional, no masking) and also
    in the decoder (causal with masking).
    """

    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float, pos_emb: str,
                 causal: bool):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size  # The size of each latent vector representation of each token
        self.n_heads = n_heads  # The number of attention heads
        self.block_size = block_size  # The max length input token sequence (the max possible tgt_len)
        self.dropout_rate = dropout_rate  # Dropout probability during training
        self.pos_emb = pos_emb  # Designate which positional embedding type to use i.e. learned or rope
        self.causal = causal  # Whether to use causal masking in the attention mechanism

        assert hidden_size % n_heads == 0, "hidden_size must be evenly divisible by n_heads"

        ######################################################################################################
        ### Define the model architecture

        # Set up the key, query, value projection matrices for all attention heads. These are the matrices
        # that transform and input tensor x into key, query, and value vectors, all having the same size
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        assert (hidden_size // n_heads) % 2 == 0, "d = hidden_size / n_heads must be even for RoPE"
        if self.pos_emb == "rope":
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

        # Add an output projection layer
        self.final_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Create a key-value cache for use in forward step-wise decoding. If we auto-regressively decode 1
        # token at a time and feed the model's y_hat from the prior step in for the next step, then we will
        # not be able to do all the self-attention calcs at once, we will have only 1 new query, key, and
        # value vector at a time. All the prior key and value vectors will remain unchanged so it would be
        # repetitive to re-compute them every step, thus we cache them here as a tuple of tensors each of
        # size (B, nh, T_kv, hs). We do not cache the query vectors since they are not needed, we are only
        # concerned with computing the self-attention updated representation of the last token which requires
        # only the query vector of the last token + the keys and values of all prior including the last
        self.KV_cache = None

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None, step: bool = False) -> torch.Tensor:
        """
        Defines the forwards pass evaluation through this self-attention layer which alters the input vectors
        based on the attention scores of each token to every other token if bidirectional or to ever prior
        token if causal.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of size (batch_size, seq_len, hidden_size) containing the token-vectors for
            each input token for the batch of text inputs.
        masks : torch.Tensor
            An input tensor of size (batch_size, seq_len) denoting the location of padding tokens which is
            used to prevent any token from attending to any of them by setting their attention score logit to
            -inf before computing the softmax operation across attention scores.
        step : torch_tensor
            If True, then the decoder is set for step-wise decoding and expects an input tensor for x
            of size (batch_size, 1, hidden_size) representing the 1 input decoder token being processed. If
            available, the key-value cache will be utilized, otherwise it will be computed and cached.

        Returns
        -------
        y : torch.Tensor
            An output tensor of the same size as the original input tensor (batch_size, seq_len, hidden_size)
            where each token vector has been altered by the self-attention mechanism.
        """
        B, T, H = x.size()  # Get the B = batch_size, T = input max seq length, H = hidden_size
        if step is True:
            assert T == 1, "If step is set to True, then T should be 1 i.e. only 1 token at a time"

        # Calculate query, key, and value vectors for all heads in this batch. Split the vectors along the
        # last dimension into head_size (hs) = hidden_size / n_heads equal sized segments, Re-order the dims.
        hs = H // self.n_heads  # Let hs be the size of each head i.e. self.n_head, E // self.n_head
        K = self.W_k(x).view(B, T, self.n_heads, hs).transpose(1, 2)  # (B, nh, T, hs)
        Q = self.W_q(x).view(B, T, self.n_heads, hs).transpose(1, 2)  # (B, nh, T, hs)
        V = self.W_v(x).view(B, T, self.n_heads, hs).transpose(1, 2)  # (B, nh, T, hs)

        if step is True:  # Utilize the key-value cache if available if performing step-wise decoding
            if self.KV_cache is not None:
                # Count how many prior tokens, that will be the positional index of the next since we index
                # starting at 0 e.g. K.shape = (B, nh, T, hs) if T = 3 then there were 3 prior tokens so the
                # correct index for the next is also 3 starting at 0 for our indexing
                pos_idx = self.KV_cache[0].shape[2]
            else:  # Otherwise if there is no KV_cache, then this is the first token and is at position 0
                pos_idx = 0

            if self.pos_emb == "rope":
                # Apply the appropriate positional embeddings using RoPE
                K = apply_rope(K, self.rope_cache, pos_idx)
                Q = apply_rope(Q, self.rope_cache, pos_idx)

            if self.KV_cache is not None:  # If there are prior tokens, bring in their data
                # Append the new key and value vectors computed for the next input token to the cache
                K = torch.concat((self.KV_cache[0], K), dim=2)  # Concat along the T dimension
                V = torch.concat((self.KV_cache[1], V), dim=2)  # Concat along the T dimension
            self.KV_cache = (K, V)  # Update the key-value cache after appending the new token vectors

        else:  # Apply RoPE to the entirety of Q and K as-is, this is not step-wise processing
            # Apply the rotary pos embeddings to the query and key vectors before computing the dot prod
            if self.pos_emb == "rope":
                Q = apply_rope(Q, self.rope_cache)
                K = apply_rope(K, self.rope_cache)

        # Compute the self-attention scores by taking the dot product of all key and value vectors
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)
        # Now we have a matrix for each sentence and each head that is (T x T) which are the attention scores
        # between all pairs of words in the sequence provided (x)

        if self.causal:  # If causal, then zero out the attention scores for all words after each token
            att_scores = att_scores.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10)

        if masks is not None:  # Apply masking to the attention scores so that we do not attend to any
            # padding tokens. Masking is done in a similar way to the causal mask, set the logits to -inf
            # for the interactions we wish to eliminate so that the attention scores after the softmax are 0
            # masks comes in with size (B, T), unsqueeze to get (B, 1, 1, T) and apply masking to
            # att_scores which is size (B, nh, T, T), i.e. anywhere there is a 1 in the mask, set to -inf
            att_scores = att_scores.masked_fill(masks.unsqueeze(1).unsqueeze(1) == 1, -1e10)

        att_scores = F.softmax(att_scores, dim=-1)  # Apply softmax normalization along the last dimension
        # so that we have scores that sum to 1 to allow for a weighted avg of V according to the att_scores
        att_scores = self.attn_dropout(att_scores)  # Apply dropout regularization

        # Compute the weighted avg value vector for each seq element using the attention scores
        y = att_scores @ V  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, H)  # Re-assemble all head outputs side by side

        # Apply a final linear projection and dropout before returning i.e. before this sub-layer output is
        # added to the sub-layer input again via a residual connection and normalized
        return self.resid_dropout(self.final_proj(y))


class CrossAttentionLayer(nn.Module):
    """
    A multi-head cross-attention layer using rotary positional embeddings (RoPE) with a linear projection at
    the end. Performs an attention operation where we use inputs from the prior decoder sub-layer to create
    the query vectors and the outputs from the encoder to create the key and value vectors.

    This attention block does not use masking because we are attending to input sequence tokens which are
    available at all time steps in the decoder.
    """

    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float, pos_emb: str):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size  # The size of each latent vector representation of each token
        self.n_heads = n_heads  # The number of attention heads
        self.block_size = block_size  # The max length input token sequence (the max possible tgt_len)
        self.dropout_rate = dropout_rate  # Dropout probability during training
        self.pos_emb = pos_emb  # Designate which positional embedding type to use i.e. learned or rope

        assert hidden_size % n_heads == 0, "hidden_size must be evenly divisible by n_heads"

        ######################################################################################################
        ### Define the model architecture

        # Set up the key, query, value projection matrices for all attention heads. These are the matrices
        # that transform and input tensor x into key, query, and value vectors, all having the same size
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        assert (hidden_size // n_heads) % 2 == 0, "d = hidden_size / n_heads must be even for RoPE"
        if self.pos_emb == "rope":
            # Store the pre-computed rope_cache values for use later, block_size = max sequence input length
            self.register_buffer("rope_cache", get_rope_cache(hidden_size // n_heads, block_size))

        # Add dropout regularization layers
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

        # Add an output projection layer
        self.final_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Create a key-value cache for use in forward step-wise decoding. If we auto-regressively decode 1
        # token at a time and feed the model's y_hat from the prior step in for the next step, then we will
        # not be able to do all the cross-attention calcs at once, we will have only 1 query vector at a time.
        # The key and value vectors remain unchanged, so it would be repetitive to re-compute them every step,
        # thus we cache them here as a list of tensors each of size (B, nh, T_kv, hs) plus a pos_idx int
        # to record what position index in the decoder we're at, the keys and values are based on the encoder
        # and never change, for the step-wise query vector, we need to apply RoPE to it which requires the
        # positional index of this ith input decoder token
        self.KV_cache = None

    def forward(self, x_kv: torch.Tensor, x_q: torch.Tensor, masks_kv: torch.Tensor = None,
                step: bool = False) -> torch.Tensor:
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
        x_q : torch_tensor
            An input tensor of size (batch_size, tgt_len, hidden_size) from the prior sub-layer of the
            masked multi-head attention decoder block i.e. the decoder hiddens to be altered by the info
            contained in the input seq processed by the encoder.
        masks_kv : torch.Tensor
            An input tensor of size (batch_size, src_len) that denotes where padding tokens are in the input
            batch with 1s, which is used here to set the attention scores of any token to a padding token to
            zero so that no attending is done to them.
        step : torch_tensor
            If True, then the decoder is set for step-wise decoding and expects an input tensor for x_q
            of size (batch_size, 1, hidden_size) representing the 1 input decoder token being processed. If
            available, the key-value cache will be utilized, otherwise it will be computed and cached.

        Returns
        -------
        y : torch.Tensor
            An output tensor of the same size as the original input x_q (batch_size, tgt_len, hidden_size)
            tensor where each token vector now be altered by cross-attention with the encoder hiddens.
        """
        B_kv, T_kv, H_kv = x_kv.size()  # Get the shape of the x_kv input from the encoder
        B_q, T_q, H_q = x_q.size()  # Get the shape of the x_q input from the decoder prior layer
        if step is True:  # If performing step-wise decoding, we expect the input to be of size 1 token
            assert T_q == 1, "If step is set to True, then T_q should be 1 i.e. only 1 token at a time"
        assert B_kv == B_q, "Batch sizes do not match"
        assert H_kv == H_q, "x_kv and x_q have different hidden sizes"
        B, H = B_kv, H_kv  # Short-hand notation, should be the same for both

        # Calculate query, key, and value vectors for all heads in this batch. Split the vectors along the
        # last dimension into head_size (hs) = hidden_size / n_heads equal sized segments, Re-order the dims.
        hs = H // self.n_heads  # Let hs be the size of vectors used by each head
        if step is True and self.KV_cache is not None:
            K, V, pos_idx = self.KV_cache  # Retrieve from the key-value cache if possible
        else:  # If step is False or self.KV_cache is None
            K = self.W_k(x_kv).view(B, T_kv, self.n_heads, hs).transpose(1, 2)  # (B, nh, T_kv, hs)
            V = self.W_v(x_kv).view(B, T_kv, self.n_heads, hs).transpose(1, 2)  # (B, nh, T_kv, hs)
            if self.pos_emb == "rope":
                # Apply the rotary positional embeddings to the key vectors before saving
                K = apply_rope(K, self.rope_cache)
            if step is True:  # Cache for later use if step-wise decoding
                pos_idx = 0  # Start a counter for what positional index this decoder token is at for RoPE
                self.KV_cache = [K, V, pos_idx]  # Store the keys, values and pos_idx for future use

        Q = self.W_q(x_q).view(B, T_q, self.n_heads, hs).transpose(1, 2)  # (B, nh, T_q, hs)
        # Apply the rotary positional embeddings to the query vector(s) before computing the dot prod
        if step is True:  # If step-wise decoding
            if self.pos_emb == "rope":
                Q = apply_rope(Q, self.rope_cache, pos_idx)
            self.KV_cache[-1] += 1  # Increment for the next iter, when retrieved we can use it as-is
        else:  # If not step-wise decoding, then we have all the query vectors at once to work with
            if self.pos_emb == "rope":
                Q = apply_rope(Q, self.rope_cache)

        # Compute cross-attention scores by taking the dot product of all key and value vectors
        # Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv) do matrix mult on the last
        # 2 dimensions to figure out the output size (_, _, T_q, hs) x (_, _, hs, T_kv) -> (_, _, T_q, T_kv)
        att_scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T_q, T_kv)

        if masks_kv is not None:  # Apply masking to the attention scores so that we do not attend to any
            # padding tokens. Masking is done in a similar way to the causal mask, set the logits to -inf
            # for the interactions we wish to eliminate so that the attention scores after the softmax are 0
            # x_kv_masks comes in with size (B, T_kv), unsqueeze to get (B, 1, 1, T_kv) and apply masking to
            # att_scores which is size (B, nh, T_q, T_kv), anywhere there is a 1 in the mask, set to -inf
            att_scores = att_scores.masked_fill(masks_kv.unsqueeze(1).unsqueeze(1) == 1, -1e10)

        # Now we have a matrix for each sentence and each head that is (T_q x T_kv) which are the attention
        # scores of each query token (decoder seq tokens) to each of the key tokens (encoder seq tokens)
        # Next we will apply a softmax normalization to the last layer i.e. for each query vector, normalize
        # to a prob dist the attention scores from each of the encoder tokens
        att_scores = F.softmax(att_scores, dim=-1)  # Apply softmax normalization along the last dimension
        att_scores = self.attn_dropout(att_scores)  # Apply dropout regularization
        # Compute the weighted avg value vector for each decoder seq element using the attention scores
        y = att_scores @ V  # (B, nh, T_q, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T_q, hs)

        y = y.transpose(1, 2).contiguous().view(B, T_q, H)  # Re-assemble all head outputs side by side

        # Apply a final linear projection and dropout before returning i.e. before this sub-layer output is
        # added to the sub-layer input again via a residual connection and normalized
        y = self.resid_dropout(self.final_proj(y))
        return y  # (batch_size, tgt_len, hidden_size)


#######################
### Encoder Objects ###
#######################
# TODO: Section marker

class EncoderBlock(nn.Module):
    """
    Encoder Transformer Attention Block that computes:
        x = LayerNorm(SelfAttention(x) + x)
        x = LayerNorm(MLP(x) + x)
        return x
    """

    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float, pos_emb: str):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size  # The size of each latent vector representation of each token
        self.n_heads = n_heads  # The number of attention heads
        self.block_size = block_size  # The max length input token sequence (the max possible tgt_len)
        self.dropout_rate = dropout_rate  # Dropout probability during training
        self.pos_emb = pos_emb  # Designate which positional embedding type to use i.e. learned or rope

        ######################################################################################################
        ### Define the model architecture

        self.attn = SelfAttentionLayer(hidden_size, n_heads, block_size, dropout_rate, pos_emb, causal=False)

        self.ln1 = nn.LayerNorm(hidden_size)  # Normalization of (att(x) + x) going into the MLP FFNN
        self.mlp = nn.Sequential(
            # Each attention adjusted word vector comes in with size hidden_size, then we apply a FFNN to it
            # with a hidden size of 4 x hidden_size, we apply the same FFNN at each position separately and
            # identically
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),  # Gaussian Error Linear Unit activation function, non-linearity
            nn.Linear(4 * hidden_size, hidden_size),  # Apply a linear layer to project down to hidden_size
            nn.Dropout(dropout_rate),  # Apply dropout for regularization
        )
        self.ln2 = nn.LayerNorm(hidden_size)  # Normalize the (mlp(x) + x) combined outputs

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forwards pass evaluation through this transformer attention block which involves:
            x = LayerNorm(SelfAttention(x) + x)
            x = LayerNorm(MLP(x) + x)
            return x

        Residual connections are used in this model.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of size (batch_size, max_word_len, embed_size) containing the word-vectors for
            each input text for a batch of text inputs.
        masks : torch.Tensor
            An input tensor of size (batch_size, seq_len) denoting the location of padding tokens which is
            used to prevent any token from attending to any of them by setting their attention score logit to
            -inf before computing the softmax operation across attention scores.

        Returns
        -------
        x : torch.Tensor
            An output tensor of the same size as the input after being passed through this attention block
            i.e. after we have adjusted the word-vector values to be more context-rich based on the attention
            scores to word vectors around it.
        """
        # Normalize the x input vector and then pass it through the self-attention block, then add x to that
        # output to form a residual connection and then norm the combined output
        x = self.ln1(self.attn(x, masks) + x)
        # Pass the updated x into the multi-layer-perceptron (MLP) FFNN, then add x to that output to form
        # a residual connection and then norm the combined output one more time
        x = self.ln2(self.mlp(x) + x)
        return x  # (batch_size, seq_len, hidden_size)


class Encoder(nn.Sequential):
    """
    This class is used to instantiate the Encoder of the transformer model and is defined to overwrite the
    default forward class definition so that it expects the same input args as EncoderBlock.forward.
    """

    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        for block in self:  # Iterate overall the encoder blocks, pass the x tensor from one to the next
            x = block(x, masks)
        return x


#######################
### Decoder Objects ###
#######################

class DecoderBlock(nn.Module):
    """
    Decoder Transformer Attention Block that computes:
        x = LayerNorm(SelfAttention(x) + x)
        x = LayerNorm(CrossAttention(enc_hiddens, enc_masks, x) + x)
        x = LayerNorm(MLP(x) + x)
        return x
    """

    def __init__(self, hidden_size: int, n_heads: int, block_size: int, dropout_rate: float, pos_emb: str):
        super().__init__()
        # Record the config parameters provided for quick user reference
        self.hidden_size = hidden_size  # The size of each latent vector representation of each token
        self.n_heads = n_heads  # The number of attention heads
        self.block_size = block_size  # The max length input token sequence (the max possible tgt_len)
        self.dropout_rate = dropout_rate  # Dropout probability during training
        self.pos_emb = pos_emb  # Designate which positional embedding type to use i.e. learned or rope

        ######################################################################################################
        ### Define the model architecture

        self.self_attn = SelfAttentionLayer(hidden_size, n_heads, block_size, dropout_rate, pos_emb,
                                            causal=True)
        self.ln1 = nn.LayerNorm(hidden_size)  # Normalization of x coming out of the self-attention mechanism

        self.cross_attn = CrossAttentionLayer(hidden_size, n_heads, block_size, dropout_rate, pos_emb)
        self.ln2 = nn.LayerNorm(hidden_size)  # Normalization of x coming out of the cross-attention mechanism

        self.mlp = nn.Sequential(
            # Each attention adjusted word vector comes in with size hidden_size, then we apply a FFNN to it
            # with a hidden size of 4 x hidden_size, we apply the same FFNN at each position separately and
            # identically
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),  # Gaussian Error Linear Unit activation function, non-linearity
            nn.Linear(4 * hidden_size, hidden_size),  # Apply linear layer to project back down to hidden_size
            nn.Dropout(dropout_rate),  # Apply dropout for regularization
        )
        self.ln3 = nn.LayerNorm(hidden_size)  # Normalization of x coming out of the MLP

    def clear_KV_cache(self) -> None:
        """
        This method is used to clear the key-value cache of the attention layers within this decoder block.
        """
        self.self_attn.KV_cache = None
        self.cross_attn.KV_cache = None

    def forward(self, x: torch.Tensor, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                step: bool = False) -> torch.Tensor:
        """
        Defines the forwards pass evaluation through this decoder transformer attention block which involves:
            x = LayerNorm(SelfAttention(x) + x)
            x = LayerNorm(CrossAttention(enc_hiddens, enc_masks, x) + x)
            x = LayerNorm(MLP(x) + x)
            return x

        Residual connections are used in this model.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of size (batch_size, tgt_len, hidden_size) containing the token vectors for each
            token in each sentence for all sentences in the batch.
        enc_hiddens : torch.Tensor
            An input tensor of size (batch_size, src_len, hidden_size) containing the token vectors for each
            token in each sentence for all sentences in the batch of the input source sequence i.e. this is
            the tensor that is output by the encoder.
        enc_masks : torch.Tensor
            An input tensor of size (batch_size, src_len) that contains 0/1 bool flags for whether each
            element is a padding token (1 means it is a padding token). This is used to prevent attending to
            the padding tokens.
        step : bool
            If set to True, then x is expected to be (batch_size, 1, hidden_size) and this forward pass will
            return the decoder outputs for 1 token sequentially by building off any that have been previously
            passed. This is used for greedy search roll-out decoding where the full target sequence is not
            known ahead of time.

        Returns
        -------
        x : torch.Tensor
            Returns a tensor of the same size as the input x tensor i.e. (batch_size, tgt_len, hidden_size)
            which contains the context-rich latent representations of each word in the decoded sequence. Each
            target sequence token attends to all of the encoder tokens and also the ones at or before the
            current token in the decode sequence.
        """
        # Pass x from the decoder through masked multi-headed self-attention, then add to self and norm
        # We don't need to pass in padding masks for the decoder self-attention layer because causal=True
        # so attention will only be applied to tokens to the left of each and padding is always on the right
        x = self.ln1(self.self_attn(x, step=step) + x)
        # Pass x then through the multi-headed cross-attention block with the encoder outputs, then add to
        # self and norm for residual connections
        x = self.ln2(self.cross_attn(enc_hiddens, x, enc_masks, step=step) + x)
        # Pass x through the multi-layer perceptron (NLP) FFNN later, then add to self and norm again
        x = self.ln3(self.mlp(x) + x)
        return x  # (batch_size, tgt_len, hidden_size)


class Decoder(nn.Sequential):
    """
    This class is used to instantiate the Decoder of the transformer model and is defined to overwrite the
    default forward class definition so that it expects the same input args as DecoderBlock.forward.
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                step: bool = False) -> torch.Tensor:
        for block in self:  # Iterate overall the decoder blocks, pass the x tensor from one to the next
            x = block(x, enc_hiddens, enc_masks, step)
        return x


#############################
### Main Model Definition ###
#############################
# TODO: Section marker

class EDTM(NMT):
    """
    Encoder-Decoder Transformer Model (EDTM) comprised of:
        - A bidirectional, multi-headed self-attention encoder
        - A multi-headed attention decoder with cross-attention to the encoder attention scores
        - Uses rotary positional embeddings (RoPE)
        - Model architecture modeled off the encoder-decoder transformer model described in the famous paper
          "Attention is All You Need"
    """

    def __init__(self, embed_size: int, hidden_size: int, num_layers: int, n_heads: int,
                 dropout_rate: float, block_size: int, pos_emb: str, vocab: Vocab, *args, **kwargs):
        """
        Bidirectional, multi-headed self-attention encoder + multi-headed attention decoder with
        cross-attention.

        Parameters
        ----------
        embed_size : int
            The size of the word vector embeddings (dimensionality).
        hidden_size : int
            The size of the hidden states (dimensionality) used by the encoder and decoder LSTM.
        num_layers : int
            The number of transformer attention blocks to use in the encoder and decoder. Must be an int value
            between 1 and 6.
        n_heads : int
            The number of attention heads to use in both the encoder and decoder. Note, that for this model,
            embed_size == hidden_size and n_heads should evenly divide into hidden_size and also
            hidden_size // n_heads should be even (for RoPE).
        dropout_rate : float
            The dropout rate used in the attention mechanism, specifies the probability of a node being
            switched off during training, something around 0.2 is typical.
        block_size : int
            The max number of input tokens to be used as context in the transformer attention blocks.
        pos_emb : str
            Designates whether to use "learned" or "rope" positional embeddings for this model.
        vocab : Vocab
            A Vocabulary object containing source (src) and target (tgt) language vocabularies.
        """
        super(EDTM, self).__init__()
        self.embed_size = embed_size  # Record the word vector embedding dimensionality
        self.hidden_size = hidden_size  # Record the size of the hidden states i.e. key, value, query vectors
        assert self.embed_size == self.hidden_size, "embed_size must be equal to hidden_size"
        assert isinstance(num_layers, int) and (1 <= num_layers <= 6), "num_layers must be an int [1, 6]"
        self.num_layers = num_layers  # Record how many attention block layers to use
        self.n_heads = n_heads  # Record the number of attention heads applied in each block
        assert self.hidden_size % n_heads == 0, "hidden_size must be evenly divisible by n_heads"
        self.hs = self.hidden_size // n_heads  # Record the dimensionality used per head i.e. block_size
        assert self.hs % 2 == 0, "hidden_size // n_heads must result in an even integer"
        self.dropout_rate = dropout_rate  # Record the dropout rate parameter
        self.block_size = block_size  # The max number of input tokens into the attention blocks
        pos_emb_types = ["learned", "rope"]
        assert pos_emb in pos_emb_types, f"pos_emb must be one of: {pos_emb_types}"
        self.pos_emb = pos_emb  # Designate which positional embedding type to use
        if pos_emb == "learned":  # Create learnable parameters for the positional embeddings
            self.pos_embeddings = nn.Parameter(torch.zeros(1, block_size, embed_size))
        self.vocab = vocab  # Use self.vocab.src_lang and self.vocab.tgt_lang to access the language labels
        self.name = "EDTM"
        self.lang_pair = (vocab.src_lang, vocab.tgt_lang)  # Record the language pair of the translation

        ######################################################################################################
        ### Define the model architecture

        # Create a word-embedding mapping for the source language vocab
        self.source_embeddings = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=self.embed_size,
                                              padding_idx=vocab.src['<pad>'])

        # Create a word-embedding mapping for the target language vocab
        self.target_embeddings = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=self.embed_size,
                                              padding_idx=vocab.tgt['<pad>'])

        self.ln_enc = nn.LayerNorm(hidden_size)  # Apply layer norm before the encoder blocks
        self.ln_dec = nn.LayerNorm(hidden_size)  # Apply layer norm before the decoder blocks

        # Transformer blocks for the encoder, we will pass the input sequence through these layers to obtain
        # a deep, context-rich embedding representation of each word in the sequence. These are bi-directional
        # i.e. non-causal, multi-headed self-attention blocks
        self.encoder = Encoder(*[EncoderBlock(hidden_size, n_heads, block_size, dropout_rate, pos_emb)
                                 for _ in range(num_layers)])

        # Transformer blocks for the decoder, we will pass the decoded values available so far through these
        # blocks and combine them with the encoder representations to create vectors that can be used to
        # generate Y_hat distributions across the vocab at each time step using casual cross-attention
        self.decoder = Decoder(*[DecoderBlock(hidden_size, n_heads, block_size, dropout_rate, pos_emb)
                                 for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_rate)
        # One final linear layer used to compute the final y-hat distribution of probabilities over the
        # entire vocab for what word token should come next according to the model
        self.target_vocab_proj = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        # print(f"Total model parameters: {sum(p.numel() for p in self.parameters())}")

    def generate_sentence_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Generates sentence masks identifying which are pad tokens so that the attention scores computed from
        the encoder hidden states that are not real input words.

        Parameters
        ----------
        enc_hiddens : torch.Tensor
            A tensor of encoder hidden states of size (batch_size, src_len, hidden_size) where src_len is the
            max length sentence within this batch.
        source_lengths : List[int]
            A list of ints denoting how long each source input sentence is i.e. all tokens beyond are padding.

        Returns
        -------
        torch.Tensor
            A tensor of sentence masks of size (b, src_len) with 1s denoting the locations of padding tokens.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for idx, src_len in enumerate(source_lengths):
            enc_masks[idx, src_len:] = 1  # Set the padding word tokens to have 1s rather than 0s
        return enc_masks.to(self.device)

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        This method passes a tensor of input source language sentences (that have already been padded to all
        the same length) into the encoder part of the transformer model to obtain deep latent representations
        of each input sub-word token. These encoder token representations will be supplied to the decoder for
        producing a roll-out decoded sequence.

        Parameters
        ----------
        source_padded : torch.Tensor
            A tensor of padded source sentences of size (batch_size, src_len) encoded as word id integer
            values with src_len = the max sentence length in the batch of source sentences.
        source_lengths : List[int]
            A list containing the length of each input sentence without padding in the batch. This list is of
            length b with max(source_lengths) == src_len.

        Returns
        -------
        enc_hiddens : torch.Tensor
            A tensor of hidden states from the encoder of shape (batch_size, src_len, hidden_size).
        """
        # Convert input sentences (already padded to all be the same length src_len) stored as a tensor of
        # word_ids of size  (batch_size, src_len) into a tensor of word embeddings (b, src_len, embed_dim)
        x = self.source_embeddings(source_padded)
        if self.pos_emb == "learned":  # Add the positional embeddings to the token embeddings
            x += self.pos_embeddings[:, :x.shape[1], :]
        x = self.ln_enc(x)  # Apply layer normalization before being fed into the encoder blocks

        # Generate a set of token masks for each source sentence so that we don't attend to padding tokens
        # in the decoder when computing attention scores
        enc_masks = self.generate_sentence_masks(x, source_lengths)  # (batch_size, src_len)
        # Pass the input seq word vectors with info on which tokens are padding tokens into the encoder block
        # of the transformer model to get context-rich latent representations of each token
        enc_hiddens = self.encoder(x, enc_masks)  # (batch_size, src_len, hidden_size)
        return enc_hiddens  # Return the latent representations of the input text tokens (b, src_len, h)

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
            A tensor of size (b, src_len, h) of hidden states i.e. the output of the encoder.
        enc_masks : torch.Tensor
            A tensor of sentence masks (0s and 1s) for masking out the padding tokens of size (b, src_len),
            where a value of 1 denotes the presence of a padding token.
        target_padded : torch.Tensor
            Gold-standard padded target sentences of size (b, tgt_len) i.e. good translations of the inputs.

        Returns
        -------
        combined_outputs : torch.Tensor
            Returns a tensor of combined outputs that are used to make y_hat predictions of size
            (batch_size, tgt_len, hidden_size) which incorporates info from the encoder hiddens and previous
            decoded tokens using multi-headed casual cross-attention.
        """
        # target_padded = target_padded[:, :-1] # Remove the <END> token for max length sentences

        # Construct a tensor Y of observed translated sentences with a shape of (b, tgt_len, e) using the
        # target model embeddings where tgt_len = maximum target sentence length and e = embedding size.
        # We use these actual translated words in our training set to score our model's predictions
        # Convert from word_ids (b, tgt_len) to word vectors ->  b, tgt_len, e)
        Y = self.target_embeddings(target_padded)  # (b, tgt_len, e)
        if self.pos_emb == "learned":  # Add the positional embeddings to the token embeddings
            Y += self.pos_embeddings[:, :Y.shape[1], :]
        Y = self.ln_dec(Y)  # Apply layer normalization before being fed into the decoder blocks

        # Pass the full Y sequence of words (gold-standard translations) into the decoder model for processing
        # in parallel. Also provide the enc_hiddens that will be used for cross-attention calculations. We
        # can compute this all at once by using masking to make sure that decoder tokens at time step t
        # attend only to those decoder tokens prior.
        decoder_outputs = self.decoder(Y, enc_hiddens, enc_masks, step=False)  # (b, tgt_len, h)

        # This produces an output vector of values at each time step that can then be used to compute logits
        # for each possiable output word in the vocab after passing them through a linear projection layer
        return decoder_outputs  # (batch_size, tgt_len, hidden_size)

    def forward(self, source: List[List[str]], target: List[List[str]], eps: float = 0.0) -> torch.Tensor:
        """
        Takes a batch of source and target sentences, compute the log-likelihood of the target sentences
        under the language models learned by the NMT system. Essentially, pass the source words into the
        encoder, then make the first prediction using the decoder. Compare that prediction to the actual
        first word of the target language true translation and compute a log-likelihood loss. Feed the true y
        of the target language into the decoder (instead of the y-hat predicted at this time step) for the
        next time-step.

        The length of source and target must be equal i.e. they contain paired parallel sentences.

        Parameters
        ----------
        source : List[List[str]]
            A list of input source language sentences i.e. a list of sentences where each sentence is a list
            of sub-word tokens.
        target : List[List[str]]
            A list of target source language sentences i.e. a list of sentences where each sentence is a list
            of sub-word tokens wrapped by <s> and </s>.
        eps : float
            An epsilon value for label smoothing i.e. how much weight to re-allocate away from the true y
            class label and disperse uniformly across all other output classes we can predict i.e. word
            tokens. This serves as a method of regularization and is 0.05 by default.

        Returns
        -------
        scores : torch.Tensor
            A Tensor of size (batch_size, ) representing the negative log-likelihood of generating the target
            sentence for each example in the input batch. This is used for back-prop in gradient descent.
            This computes the loss of the model over the input batch.
        """
        assert len(source) == len(target), "The number of source and target sentences must be equal"
        assert 0 <= eps <= 0.3, "eps must be a float value between 0 and 0.3"
        # Record the length of each input sentence with a truncated limit of block_size upper bound
        source_lengths = [min(len(s), self.block_size) for s in source]

        # Convert from a list of lists into tensors of word_ids where src_len is the max length of sentences
        # among the input source sentences and tgt_len is the max length of sentences among the output
        # sentences and b = batch_size i.e. how many sentences in total (which should be equal in both)
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor (b, src_len)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor (b, tgt_len)

        # Enforce the block_size as the context size limit of the inputs, truncate anything larger
        if source_padded.shape[1] > self.block_size:
            source_padded = source_padded[:, :self.block_size]
        if target_padded.shape[1] > self.block_size:
            target_padded = target_padded[:, :self.block_size]

        # Call the encoder on the padded source sentences which will be re-used for each step of the decoder
        enc_hiddens = self.encode(source_padded, source_lengths)  # (batch_size, src_len, hidden_size)

        # Generate a set of token masks for each source sentence so that we don't attend to padding tokens
        # in the decoder when computing attention scores
        enc_masks = self.generate_sentence_masks(enc_hiddens, source_lengths)  # (b, src_len)

        # Call decode to run the full set of decoder operations which passes in the true tgt word tokens to
        # the decoder at each timestep regardless of what the model would predict. This call to decode returns
        # the final sub-word token hidden state at each time-step of the decoder i.e. what we would use to
        # make next-word predictions at each time step
        decoder_outputs = self.decode(enc_hiddens, enc_masks, target_padded)  # (b, tgt_len, h)

        # Compute the prob distribution over the vocabulary for each prediction timestep from the decoder,
        # decoder_outputs is what would be used at each timestep, we can process them all at once since we
        # have them all here at once as one big tensor of size (b, tgt_len, V)
        log_prob = F.log_softmax(self.target_vocab_proj(decoder_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text i.e. the padding, create a bool
        # mask of 0s and 1s by checking that each entry is not equal to the <pad> token, 0s == padding token
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()  # (b, tgt_len)

        # Compute log probability of generating the true target words provided in this example i.e. compute
        # the cross-entropy loss by pulling out the model's y-hat values for the true target words. For each
        # word in each sentence, pull out the y_hat prob associated with the true target word at time t.
        # log_prob is (b, tgt_len, V) and describes the probability distribution over the next word after the
        # current time step t. I.e. the first Y_t token is <s> and the first y_hat is the distribution of
        # what the model thinks should come afterwards. Hence log_prob[:, :-1, :] aligns wtih the true Y_t
        # words. target_padded[:, 1:]. tgt_len includes <s> at the start and </s> at the end. We don't want to
        # include the prob of <s> but we do want to include the prob of predicting </s> to end the sentence.
        target_words_log_prob = torch.gather(log_prob[:, :-1, :], index=target_padded[:, 1:].unsqueeze(-1),
                                             dim=-1).squeeze(-1)  # (b, tgt_len - 1) result
        if eps > 0:  # Apply label smoothing, put (1-eps) weight on the true class and eps / (|V|-1) on all
            # others when computing the cross-entropy loss values. From the above, we already have the values
            # for the true class label, so we can down-weight that by (1-eps) and then add to reach the goal
            sum_all_others = log_prob[:, :-1, :].sum(-1) - target_words_log_prob  # Sum log prob of all others
            mean_all_others = sum_all_others / (log_prob.shape[-1] - 1)  # Divide by (|V| - 1) to normalize
            # Take the weighted sum, down-weight the log-prob of the true class to (1-eps) and add all the
            # others at a weight of eps each i.e. the sum of all others gets a collective weight of eps
            target_words_log_prob = target_words_log_prob * (1 - eps) + mean_all_others * (eps)

        # Zero out the y_hat values for the padding tokens so that they don't contribute to the sum
        target_words_log_prob = target_words_log_prob * target_masks[:, 1:]  # (b, tgt_len - 1)

        # Return the sum of negative log-likelihoods across all target tokens for each sentence
        return -target_words_log_prob.sum(dim=1)  # Returns a tensor of floats of size (batch_size, )

    def clear_decoder_KV_cache(self) -> None:
        """
        This method is used to clear the key-value cache of the attention layers within the decoder.
        """
        for block in self.decoder:
            block.clear_KV_cache()

    def translate(self, src_sentences: Union[List[str], List[List[str]]], beam_size: int = 1,
                  k_pct: float = 0.1, max_decode_lengths: Union[List[int], int] = None,
                  tokenized: bool = True) -> List[List[Union[Union[str, List[str]], float]]]:
        """
        Given a list of source sentences (either a list of strings or a list of sub-word tokens), this method
        generates output translations using either greedy search (if beam_size == 1) or beam search (if
        beam_size > 1). src_sentences is processed in batches to speed up calculations, but this computation
        can be slow for large sets of input source sentences.

        Greedy search translates by sequentially predicting the next token by randomly sampling among the
        sub-words that make up the top k% of the probability distribution among all possible output sub-word
        tokens, according to their relative probabilities. k_pct is the parameter that governs this behavior.

        If k_pct is let as None, then the most probable sub-word token is always chosen and the output has no
        variation from one call to another. By default, k_pct is set to 10% which means that the model will
        sample from the sub-words that make up the top 10% of the probability distribution at each prediction
        step. k_pct must be a float value (0, 1]. E.g. if the most likely work token has a prob of 50% and
        k_pct = 10%, then it will be selected with probability 100%. If instead the top 2 most probably tokens
        have probs of 7% and 5% respectively, then the next token will be sampled from just those 2 with more
        of a chance given the first due to its higher relative probability.

        max_decode_lengths specifies the max length of the translation output for each input sentence. If an
        integer is provided, then that value is applied to all sentences. If not specified, then the default
        value will be len(src_sentence) * 2.5 for each src_sentence in src_sentences. The values of
        max_decode_lengths are capped at 250 globally.

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
        beam_size : int
            An integer denoting the beam size for the translation generations. If set to 1, then greedy search
            is used with the k_pct parameter defined below. Otherwise, beam search is used with this parameter
            denoting the beam size. Note, beam size is much slower than greedy search but may return higher
            quality output translations.
        k_pct : float
            This method builds an output translation by sampling among the eligible candidate sub-word tokens
            according to their relative model-assigned probabilities at each time step. If k_pct is set to
            None, then the most likely word is always chosen (100% greedy). Otherwise, the most probably
            words making up k_pct of the overall probability distribution are used. As k_pct is lowered, the
            variance of the model's outputs increases.
        max_decode_lengths : Union[List[int], int], optional
            The max number of time steps to run the decoder unroll sequence for each input sentence. The
            output machine translation produced for each sentence will be capped in length to a certain
            amount of sub-word tokens specified here. The default is 2.5 * len(src_sentence) and all values
            must be <= 250.
        tokenized : bool, optional
            Denotes whether src_sentences has already been tokenized.

            If False, then src_sentences is assumed to be a list of sentences stored as strings which will be
            tokenized internally before being fed into the model. If False, then the output list of machine
            translations for each input sentence will also be sentences stored as strings.
            E.g. [['Where is the Bank?', 0.9648], ...]

            If True, the src_sentences is assumed to be a list of sub-word token lists which can be fed into
            the model directly. If True, then the output list of machine translations for each input sentence
            will be a list of sub-word tokens similar to the way src_sentences was input.
            E.g. [[['<s>', '▁Where', '▁is', '▁the', '▁Bank', '?', '</s>'], 0.9648], ...]

        Returns
        -------
        List[List[Union[Union[str, List[str]], float]]]
        Returns a list of hypotheses i.e. length 2 lists each containing:
            - The predicted translation from the model as either a string (if tokenize is True) or a
              list of sub-word tokens (if tokenize is False).
            - The negative log-likelihood score of the decoding as a float
        """
        b = len(src_sentences)  # Record how many input sentences there are i.e. the batch size
        assert b > 0, "len(src_sentences) must be >= 1"
        if tokenized is False:  # Convert the input sentences from strings to lists of subword strings
            src_sentences = util.tokenize_sentences(src_sentences, self.lang_pair[0], is_tgt=False)
        elif isinstance(src_sentences[0], str):  # If 1 sentence is passed in, then add an outer list wrapper
            src_sentences = [src_sentences]  # Make src_sentences a list of lists
            b = len(src_sentences)  # Redefine to be 1
        msg = f"beam_size must be an int [1, 5], got, {beam_size}"
        assert isinstance(beam_size, int) and 0 < beam_size <= 5, msg
        if k_pct is not None:  # If not None, then perform data-validation
            assert 0 < k_pct <= 1.0, "k_pct must be in (0, 1] if not None"
        if max_decode_lengths is None:  # Default to allow for 250% more words per sentence if not specified
            max_decode_lengths = [int(len(s) * 2.5) for s in src_sentences]
        if isinstance(max_decode_lengths, int):  # Convert to a list if provided as an int
            max_decode_lengths = [max_decode_lengths for i in range(b)]
        max_decode_lengths = max_decode_lengths.copy()  # Copy to avoid mutation
        for i, n in enumerate(max_decode_lengths):  # Check all are integer valued and capped at 250
            assert isinstance(n, int) and n > 0, "All max_decode_lengths must be integers > 0"
            max_decode_lengths[i] = min(n, 250)

        msg = "src_sentences and max_decode_lengths must be the same length"
        assert len(max_decode_lengths) == len(src_sentences), msg

        self.eval()  # Set the model to eval mode so that dropout is not applied when generating values

        with torch.no_grad():  # no_grad() signals backend to throw away all gradients

            # Record the length of each input sentence with a truncated limit of block_size upper bound
            source_lengths = [min(len(s), self.block_size) for s in src_sentences]

            # Convert the input source sentence into a tensor object of size (batch_size, src_len) of word id
            # ints with padding so that all are the same length
            source_padded = self.vocab.src.to_input_tensor(src_sentences,
                                                           self.device)  # (batch_size, src_len)

            # Enforce the block_size as the context size limit of the inputs, truncate anything larger
            if source_padded.shape[1] > self.block_size:
                source_padded = source_padded[:, :self.block_size]

            # Pass it through the encoder to generate the encoder hidden states for each word of each input
            # sentence, this gives us a tensor of shape (batch_size, src_len, hidden_size)
            enc_hiddens = self.encode(source_padded, source_lengths)

            # Generate a set of masks for each source sentence so that we don't attend to padding tokens
            # in the decoder when computing attention scores
            enc_masks = self.generate_sentence_masks(enc_hiddens, source_lengths)  # (b, src_len)

            if beam_size == 1:  # Proceed with greedy search
                mt = self._greedy_search(enc_hiddens, enc_masks, k_pct, max_decode_lengths)
            else:  # Otherwise, utilize beam search to generate output translations
                mt = [
                    self._beam_search(enc_hiddens[i, :, :], enc_masks[i, :], beam_size, max_decode_lengths[i])
                    for i, src_s in enumerate(src_sentences)]

        if tokenized is False:  # Convert the outputs into concatenated sentences to match the input format
            mt = [[util.tokens_to_str(x[0]), x[1]] for x in mt]  # Convert each to a string sentence
        return mt

    def _greedy_search(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor, k_pct: float,
                       max_decode_lengths: List[int]) -> List[List[Union[List[str], float]]]:
        """
        This method performs greedy search on the input source sentences provided (enc_hiddens) using a given
        k-percent cutoff (k_pct). This method is built to be called only within the translate() method.

        Parameters
        ----------
        enc_hiddens : torch.Tensor
            A tensor of size (batch_size, src_len, embed_size) corresponding to this input src_sentence after
            it has been passed through the encoder.
        enc_masks : torch.Tensor
            A tensor of size (batch_size, src_len) corresponding to this input src_sentences which identify
            where the padding tokens are.
        k_pct : float
            This method builds an output translation by sampling among the eligible candidate sub-word tokens
            according to their relative model-assigned probabilities at each time step. If k_pct is set to
            None, then the most likely word is always chosen (100% greedy). Otherwise, the most probably
            words making up k_pct of the overall probability distribution are used. As k_pct is lowered, the
            variance of the model's outputs increases.
        max_decode_lengths : List[int]
            The max number of time steps to run the decoder unroll sequence for each input sentence.

        Returns
        -------
        List[List[Union[List[str], float]]]
        Returns a list of hypotheses i.e. length 2 lists each containing:
            - The predicted translation from the model as a list of sub-word tokens
            - The negative log-likelihood score of the decoding as a float
        """
        b = enc_hiddens.shape[0]  # The batch_size of the inputs
        # Create output translations for each input sentence, begin with the start-of-sentence begin token
        # and also record the negative log likelihood of the sentence
        mt = [[['<s>'], 0] for _ in range(b)]  # Machine translation outputs

        # Use the last output word Y_hat_(t-1) as the next input word (Y_t) going into the decoder, we always
        # start with the <s> sentence start token for each output translation
        Y_t = torch.tensor([self.vocab.tgt[mt[i][0][-1]] for i in range(b)],
                           dtype=torch.long, device=self.device)  # (b, )

        # Iterate until we've a complete output translations or we reach the max output len
        finished = 0  # Track how many output translation sentences are finished
        finished_flags = [0 for i in range(b)]  # Mark which sentences have been completed

        # Reset the key-value caches of the decoder layers before starting to clear out anything prior
        self.clear_decoder_KV_cache()

        while finished < b:  # Iterate until all output translations are finished generating
            Y_t = self.target_embeddings(Y_t).unsqueeze(1)  # (b, 1, e) From word_ids to word vecs
            if self.pos_emb == "learned":  # Add the positional embeddings to the token embeddings
                Y_t += self.pos_embeddings[:, :Y_t.shape[1], :]
            Y_t = self.ln_dec(Y_t)  # Apply layer norm before being fed into the decoder blocks

            # Put in the encoder hiddens and their padding masks + the most recent target word (starting with
            # <s>) and get out the decoder outputs for the most recent time-step i.e. 1 per sentence of size
            # hidden_size -> this is what's used to generate the y_hat dist over the next token to predict
            dec_outputs = self.decoder(Y_t, enc_hiddens, enc_masks, step=True)  # (b, 1, h)
            dec_outputs = dec_outputs.squeeze(1)  # (b, 1, h) -> (b, h)

            # Compute the log probs over all possible next target words using the last decoder hiddens i.e.
            # the one that is to be fed to self.target_vocab_projection to get (b, |V|)
            log_p_t = F.log_softmax(self.target_vocab_proj(dec_outputs), dim=-1)  # (b, |V|)

            if k_pct is None:  # Select the word with the highest modeled probability always
                # Find which word has the highest log prob for each sentence, idx = word_id in the vocab
                Y_hat_t = torch.argmax(log_p_t, dim=1)  # (b, ) the most probably next word_id for each
            else:  # Randomly sample from the sub-words at or above the kth most probably percentile
                prob_t = torch.exp(log_p_t)  # Exponentiate to convert to a prob dist (b, |V|)
                # Find what cutoff is required to make it into the words that collectively sum to form the
                # top k percent of the probability distribution i.e. for a flat distribution there will be
                # more words, for a more concentrated dist, there will be fewer words that make the cut
                Y_hat_t = torch.zeros(b, dtype=int, device=self.device)  # Start off with all zeros
                for i in range(b):
                    if finished_flags[i] == 0:  # Compute if this sentence is not already finished
                        sorted_probs = prob_t[i, :].sort(descending=True)  # Sort the probs of this dist
                        bool_vec = sorted_probs.values.cumsum(0) <= k_pct  # The entries in the top k %
                        bool_vec[0] = True  # Always have at least 1 entry set to true i.e. this happens if
                        # the most likely word has a higher prob than k
                        idx, prob = sorted_probs.indices[bool_vec], sorted_probs.values[bool_vec]
                        prob /= prob.sum()  # Re-normalize to 1 and then sample to get the next prediction
                        Y_hat_t[i] = idx[prob.multinomial(num_samples=1, replacement=True).item()]
                    # Else leave the word_id as 0 which defaults to the padding token

            for i in range(b):  # Record the next predicted word for each output translation
                if finished_flags[i] == 0:  # Record if this sentence is not already finished
                    mt[i][0].append(self.vocab.tgt.id2word[Y_hat_t[i].item()])
                    mt[i][1] += -log_p_t[i, int(Y_hat_t[i].item())]  # Sum the log prob of y-hats
                    # Check if the translation has been complete i.e. we got a sentence stop token or the
                    # max decode length was reached for this sentence
                    if mt[i][0][-1] == "</s>" or len(mt[i][0]) - 1 == max_decode_lengths[i]:
                        # mt[i][0] is the list of output sub-word tokens, which beings with </s> for all
                        # so it is already length 1, so we subtract 1 to trigger when the output tokens
                        # added after </s> are max_decode_lengths[i]
                        finished += 1  # Record that 1 more sentence was finished
                        finished_flags[i] = 1  # Mark this sentence off as finished

            # Update relevant state variables for next iteration
            Y_t = Y_hat_t  # For next iter, set the current y_hat output as the next y target inputs (b, )

        self.clear_decoder_KV_cache()  # Clear the key-value caches again after we're done to clean up
        return mt

    def _beam_search(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor, beam_size: int,
                     max_decode_length: int, alpha: float = 0.8) -> List[Union[List[str], float]]:
        """
        This method performs beam search on the input source sentence provided (enc_hiddens) using a given
        beam size (beam_size). This method is built to be called only within the translate() method.

        Parameters
        ----------
        enc_hiddens : torch.Tensor
            A tensor of size (src_len, embed_size) corresponding to this input src_sentence after it has been
            passed through the encoder.
        enc_masks : torch.Tensor
            A tensor of size (src_len, ) corresponding to this input src_sentence which identify where the
            padding tokens are.
        beam_size : int
            An integer [1, 5] denoting the beam size i.e. how many hypotheses to track during decoding.
        max_decode_length : int
            An integer denoting the max output decode length for the returned translation.
        alpha : float, optional
            A length normalization parameter (see Google NMT length penalty for details) used to compare
            the log-probabilities of hypotheses of various lengths to one another i.e. helps to normalize
            by word count so that we do not unfairly penalize longer hypotheses. The default is 0.8.

        Returns
        -------
        List[Union[List[str], float]]
        Returns the most likely hypothesis found during beam search as a list containing:
            - The predicted translation from the model as a list of sub-word tokens
            - The negative log-likelihood score of the decoding as a float
        """
        assert isinstance(beam_size, int) and 0 < beam_size <= 5, "beam_size must be an int [1, 5]"
        assert len(enc_hiddens.shape) == 2, "enc_hiddens should be 2 dimensional"
        assert len(enc_masks.shape) == 1, "enc_masks should be 1 dimensional"
        assert enc_hiddens.shape[0] == enc_masks.shape[0], "enc_hiddens and enc_masks should match in dim 0"
        assert 0.6 <= alpha <= 1.0, "alpha must be between 0.5 and 1.0"
        mdl = max_decode_length  # Shorter alias

        # enc_hiddens comes in as size (src_len, embed_size), expand to (beam_size, src_len, embed_size)
        enc_hiddens = enc_hiddens.expand((beam_size, enc_hiddens.shape[0], enc_hiddens.shape[1]))
        # enc_masks comes in as size (src_len, ), expand to (beam_size, src_len)
        enc_masks = enc_masks.expand((beam_size, enc_masks.shape[0]))

        # Maintain a list of hypotheses which can be sorted by the first element to maintain the k best where
        # k = beam_size and each records (log_prob_sum, decoded_sub_word_tokens, dec_inputs) with
        # decoded_sub_word_tokens being list of strings and dec_inputs being a tensor of size (dec_len, embed)
        h = [0, ["<s>"]]  # Start off with just 1 hypothesis i.e. the sentence start token
        dec_inputs = self.target_embeddings(torch.tensor(self.vocab.tgt[h[1][-1]], dtype=torch.long,
                                                         device=self.device).unsqueeze(0))  # (T=1, e)
        if self.pos_emb == "learned":  # Add the positional embeddings to the token embeddings
            dec_inputs += self.pos_embeddings[:, 0, :]
        dec_inputs = self.ln_dec(dec_inputs)  # Apply layer norm before being fed into the decoder
        h.append(dec_inputs)  # Add this tensor as the 3rd element of every hypothesis
        hypotheses = [h, ]  # Move into a list for iteration

        complete_hypotheses = []  # Collect the completed hypotheses and iter until we get k = beam_size

        while len(complete_hypotheses) < beam_size:  # Iterate until we get the desired number of hypotheses
            new_hypotheses = []  # Create a new hypothesis list to replace the existing one

            # Collect together all the decoder input tensors from each hypothesis to feed into the decoder
            dec_inputs = torch.concat([h[-1].unsqueeze(0) for h in hypotheses])  # (beam_size, T, embed_size)
            dec_outputs = self.decoder(dec_inputs, enc_hiddens[0:len(dec_inputs):, :, :],
                                       enc_masks[0:len(dec_inputs):, :], step=False)  # (beam_size, T, h)

            # Compute the log probabilities over all possible next target words using the last decoder
            # hiddens i.e. the one that is to be fed to self.target_vocab_projection, gives us (b, |V|)
            log_p_t = F.log_softmax(self.target_vocab_proj(dec_outputs[:, -1, :]), dim=-1)  # (beam_size, |V|)

            # For each prior hypothesis, find the top k=beam_size ways to extend it, add each of those to the
            # new hypothesis list, which will later be sorted and pruned to retain the top k=beam_size
            log_probs, idx = torch.topk(log_p_t, k=beam_size, dim=-1, largest=True)

            for i, h in enumerate(hypotheses):  # Iter over each prior hypothesis and extend each by 1
                for j in range(beam_size):
                    new_h = [h[0] + log_probs[i, j], h[1] + [self.vocab.tgt.id2word[idx[i, j].item()]]]
                    new_word_embed = self.target_embeddings(torch.tensor(idx[i, j].item(), dtype=torch.long,
                                                                         device=self.device).unsqueeze(0))
                    if self.pos_emb == "learned":  # Add the positional embeddings to the token embeddings
                        new_word_embed += self.pos_embeddings[:, (len(h[1]) + 1), :]
                    new_word_embed = self.ln_dec(new_word_embed)  # Apply layer norm
                    new_h.append(torch.concat([h[-1], new_word_embed], dim=0))  # Add the new word's
                    # embedding to the existing decoder input values for the prior decoded word tokens

                    # Check if this hypothesis has been completed, if so, add it to complete_hypotheses
                    # instead of new_hypotheses so that we can exit the while loop. We don't count the start
                    # token as part of the max_decode_length, hence we minus 1 from the length of the decoded
                    # sub-word token list of the hypothesis when checking for completion conditions. Also
                    # do not record [<s>, </s>] as a blank sentence either, require at least 3 tokens
                    if (new_h[1][-1] == "</s>" and len(new_h[1]) > 2) or (len(new_h[1]) - 1 == mdl):
                        complete_hypotheses.append(new_h)
                    else:  # Otherwise this new hypothesis is not yet completed, keep it in the running
                        new_hypotheses.append(new_h)  # Add the new hypothesis to the new list of hypotheses

            # Sort the new hypotheses by the sum of log probs divided by the Google NMT length penalty i.e.
            # a normalization constant of seq_len ^ (alpha) so that we do not unfairly penalize longer seqs
            # Sort this normalized avg log prob per token metric in descending order i.e. highest probs first
            new_hypotheses.sort(key=lambda x: -x[0] / (len(x[1]) ** alpha))
            hypotheses = new_hypotheses[:beam_size]  # Update for next iteration, keep only the top hypotheses

        # Once we've collected k = beam_size completed hypotheses, return the best one
        complete_hypotheses.sort(key=lambda x: -x[0] / (len(x[1]) ** alpha))
        return [complete_hypotheses[0][1],
                -complete_hypotheses[0][0]]  # (work_token_list, neg_log_likelihood)

    def save(self, model_path: str, verbose: bool = False) -> None:
        """
        Method for saving the model to disk.

        Parameters
        ----------
        model_path : str
            A file path detailing where the model should be saved e.g. saved_models/{model}/DeuEng/model.bin
        verbose : bool, optional
            If True, then the model_path is printed before saving. The default is False.
        """
        if verbose is True:
            print(f"Saving model parameters to {model_path}", file=sys.stderr)
        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                         n_heads=self.n_heads, dropout_rate=self.dropout_rate, block_size=self.block_size,
                         pos_emb=self.pos_emb),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path: str) -> EDTM:
        """
        Method for loading in a model saved to disk.

        Parameters
        ----------
        model_path : str
            A file path detailing where the model should be saved e.g. saved_models/{model}/DeuEng/model.bin

        Returns
        -------
        model : EDTM
            Returns an object instance of this model class with the weights saved to disk.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = cls(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        return model
