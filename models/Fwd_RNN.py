#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the source code of the forward recurrent neural network model (Fwd_RNN).

The basis of this code comes from Stanford XCS224N Assignment 4 code and has been heavily modified to fit the
needs of this project.
"""

from __future__ import annotations
import sys, os
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.util import NMT
from vocab.vocab import Vocab
import util


class Fwd_RNN(NMT):
    """
    Neural Machine Translation model comprised of:
        - A forward RNN encoder
        - A forward RNN decoder

    This is one of the simplest seq2seq model architectures there is.
    """

    def __init__(self, embed_size: int, hidden_size: int, num_layers: int, vocab: Vocab, *args, **kwargs):
        """
        Forward RNN encoder + forward RNN decoder model instantiation.

        Parameters
        ----------
        embed_size : int
            The size of the word vector embeddings (dimensionality).
        hidden_size : int
            The size of the hidden state (dimensionality) used by the encoder and decoder RNN.
        num_layers : int
            The number of forward RNN layers to use in the encoder.
        vocab : Vocab
            A Vocabulary object containing source (src) and target (tgt) language vocabularies.
        """
        super(Fwd_RNN, self).__init__()
        assert isinstance(num_layers, int) and (1 <= num_layers <= 5), "num_layers must be an int [1, 5]"
        self.num_layers = num_layers  # Record the number of layers in the encoder and decoder
        self.embed_size = embed_size  # Record the word vector embedding dimensionality
        self.hidden_size = hidden_size  # Record the hidden size of both the encoder and decoder
        self.vocab = vocab  # Use self.vocab.src_lang and self.vocab.tgt_lang to access the language labels
        self.name = "Fwd_RNN"
        self.lang_pair = (vocab.src_lang, vocab.tgt_lang)  # Record the language pair of the translation

        ######################################################################################################
        ### Define the model architecture

        # Create a word-embedding mapping for the source language vocab
        self.source_embeddings = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size,
                                              padding_idx=vocab.src['<pad>'])

        # Create a word-embedding mapping for the target language vocab
        self.target_embeddings = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=embed_size,
                                              padding_idx=vocab.tgt['<pad>'])

        # Takes in the word embedding for each input word of the source language (each of size embed_size)
        # and outputs a hidden state vector of size hidden_size, this layer is encoder
        self.encoder = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                              nonlinearity="tanh", bias=True, batch_first=True, bidirectional=False)

        # Takes in the word embedding of the prior predicted output word and rolls the prediction forward to
        # produce the predicted translation in the output language. This layer cannot be bidirectional since
        # we make y-hat predictions sequentially from left-to-right. The inputs are a concatenation of the
        # word embedding of the prior predicted word and the final context vector from the encoder
        self.decoders = []  # All layers MUST be defined as separate attributes of the model
        # The first layer expects the new word embedding input + the hidden state from the prior layer
        self.decoder_0 = nn.RNNCell(input_size=embed_size + hidden_size, hidden_size=hidden_size,
                                    nonlinearity="tanh", bias=True)
        self.decoders.append(self.decoder_0)

        for i in range(1, self.num_layers):  # Add additional decoder layers as needed, they take the input
            # of the prior hidden state from the same later 1 timestep back + the hidden state below
            setattr(self, f"decoder_{i}", nn.RNNCell(input_size=hidden_size * 2, hidden_size=hidden_size,
                                                     nonlinearity="tanh", bias=True))
            self.decoders.append(getattr(self, f"decoder_{i}"))

        # Takes in the last hidden state of layer for each sentence in a batch of sentences and applies a
        # weight matrix transformation via a hidden layer to initialize the hidden state of the decoder
        self.h_projection = nn.Linear(in_features=hidden_size * self.num_layers,
                                      out_features=hidden_size * self.num_layers, bias=True)

        # This is used to compute the final y-hat distribution of probabilities over the entire vocab for what
        # word token should come next. I.e. y_hat = softmax(W_{vocab} @ h_{t}) where y_hat is a length |V|
        # vector and h_{t}
        self.target_vocab_projection = nn.Linear(in_features=hidden_size, out_features=len(vocab.tgt),
                                                 bias=False)

    def forward(self, source: List[List[str]], target: List[List[str]], eps: float = 0.0) -> torch.Tensor:
        """
        Takes a mini-batch of source and target sentences, compute the log-likelihood of the target sentences
        under the language models learned by the NMT system. Essentially, pass the soruce words into the
        encoder, then make the first prediction using the decoder. Compare that prediction to the actual
        first word of the target language true translation and compute a log-likelihood loss. Feed the true y
        of the target language into the decoder (instead of the y-hat predicted at this time step) for the
        next time-step.

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
            sentence for each example in the input batch.
        """
        assert len(source) == len(target), "The number of source and target sentences must be equal"
        source_lengths = [len(s) for s in source]  # Compute the length of each input source sentence

        # Convert from a list of lists into tensors where src_len is the max length of sentences among the
        # input source sentences and tgt_len is the max length of sentences among the output sentences and
        # b = batch_size i.e. how many sentences in total (which should be equal in both)
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor (b, src_len)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor (b, tgt_len)

        # Call the encoder on the padded source sentences, get the initialization of the decoder hidden state
        dec_init_states = self.encode(source_padded, source_lengths)  # (batch_size, layers, hidden_size)

        # Call the decoder using the initialized decoder hidden state and the padded target sentences to
        # generate the top layer hidden states of the decoder at each time step for each sentence
        dec_hidden_states = self.decode(dec_init_states, target_padded)  # (batch_size, tgt_len, hidden_size)

        # Compute the prob distribution over the vocabulary for each prediction timestep from the decoder
        log_prob = F.log_softmax(self.target_vocab_projection(dec_hidden_states), dim=-1)  # (b, tgt_len, V)

        # Zero out, probabilities for which we have nothing in the target text i.e. the padding, create a bool
        # mask of 0s and 1s by checking that each entry is not equal to the <pad> token
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating the true target words provided in this example i.e. compute
        # the cross-entropy loss by pulling out the model's y-hat values for the true target words. For each
        # word in each sentence, pull out the y_hat log_prob associated with the true target word at time t.
        # log_prob is (b, tgt_len, V) and describes the probability distribution over the next word after the
        # current time step t. I.e. the first Y_t token is <s> and the first y_hat is the distribution of
        # what the model thinks should come afterwards. Hence log_prob[:, :-1, :] aligns with the true Y_t
        # words target_padded[:, 1:]
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

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Apply the encoder to a collection of padded source sentences of size (b, src_len) to obtain the
        encoder hidden states. Use them to create the initialized decoder hidden state for translation.

        Parameters
        ----------
        source_padded : torch.Tensor
            A tensor of padded source sentences of size (b, src_len) encoded as word id integer values
            where b=batch_size and src_len = the max sentence length in the batch of source sentences. These
            have been pre-sorted in order of longest to shortest sentence.
        source_lengths : List[int]
            A list containing the length of each input sentence without padding in the batch. This list is of
            length b with src_len = max(source_lengths).

        Returns
        -------
        dec_init_states : torch.Tensor
            A tensor representing the decoder's initial hidden state for each sentence of size (b, h).

        """
        # Convert input sentences (padded to all be the same length src_len) stored as a tensor of size
        # (batch_size, src_len) into a tensor of size (batch_size, src_len, embed_dim)
        X = self.source_embeddings(source_padded)

        # pack_padded_sequence takes these padded sequences and their corresponding lengths as input. It then
        # transforms them into a "packed" format that only includes the non-padded portions of the sequences.
        # By removing unnecessary computations on padded portions of the sequences, the computations are
        # faster to perform. The size of X is (src_len, batch_size, embed_dim) so the batch_size is the first
        # so we set batch_first=True.
        X_packed = pack_padded_sequence(X, lengths=source_lengths, batch_first=True, enforce_sorted=True)

        # Pass in the packed sentence of sentences of size (batch_size, src_len, embed_dim). This returns
        # a tensor of size (batch_size, src_len, hidden_size) containing the hidden states of the RNN at each
        # timestep (i.e. word in each sentence) and also the last hidden state for each sentence encoding.
        enc_hiddens, last_hiddens = self.encoder(X_packed)  # last_hiddens = (layers, b, h)
        # Switch the dimensions to be (b, layers, h), then reshape to be (b, layer * h)
        last_hiddens = last_hiddens.transpose(0, 1)
        shp = last_hiddens.shape
        last_hiddens = last_hiddens.reshape(shp[0], shp[1] * shp[2])  # Size is now (b, layer * h)

        # last_hiddens is a tensor of size (b, layers * h), pass it through the h_projection layer to compute
        # the initial states of the decoder for each layer and for each sentence in the batch
        # (b, layers * h) @ (layers * h, layers * h) = (b, layers * h)
        shp = last_hiddens.shape
        dec_init_states = self.h_projection(last_hiddens).reshape(shp[0], self.num_layers, self.hidden_size)
        return dec_init_states  # (batch_size, layers, hidden_size)

    def decode(self, dec_init_states: torch.Tensor, target_padded: torch.Tensor) -> torch.Tensor:
        """
        Computes output hidden-state vectors for each word in each batch of target sentences i.e. runs the
        decoder to generate the output sequence of hidden states for each word of each sentence while using
        the true Y_t words provided in the target translation as inputs at each time step instead of the the
        prior Y_hat_values provided from the prior step. This method is used for training only.

        Parameters
        ----------
        dec_init_states : torch.Tensor
            An initial state for the decoder for each sentence of size (batch_size, hidden_size).
        target_padded : torch.Tensor
            A tensor of size (batch_size, tgt_len) of padded gold-standard output translations encoded as
            word id integer values where tgt_len is the max-length among all target sentences and
            b = batch size.

        Returns
        -------
        hidden_states : torch.Tensor
            A tensor of size (batch_size, tgt_len, hidden_size) containing the hidden states of the decoder
            at each time step for each sentence.
        """
        # target_padded = target_padded[:, :-1] # Remove the <END> token for max length sentences

        # Construct a tensor Y of observed translated sentences with a shape of (b, tgt_len, e) using the
        # target model embeddings where tgt_len = maximum target sentence length and e = embedding size.
        # We use these actual translated words in our training set to score our model's predictions
        Y = self.target_embeddings(target_padded)

        dec_states = dec_init_states  # Start the prior decoder states using the final hidden state from
        # the encoder, this is size (batch_size, layers, hidden_size)
        hidden_states = []  # Record the hidden states of the final top layer for each timestep, these are the
        # vectors that are used to make output y-hat word predictions

        # Use the torch.split function to iterate over the time dimension of Y, this will give us Y_t which
        # is a tensor of size (1, b, e) i.e. the word embedding of the ith target word from each sentence
        for Y_t in torch.split(tensor=Y, split_size_or_sections=1, dim=1):  # Split along dim=1 i.e. iter
            # over words in each sentence
            Y_t = torch.squeeze(Y_t, dim=1)  # Squeeze Y_t into (b, e). i.e. remove the 2nd dim
            dec_states = self.step(Y_t, dec_states)  # Perform a forward step of the decoder, update h_t
            # dec_states is (batch_size, layers, hidden_size)
            hidden_states.append(dec_states[:, -1, :])  # Record the top layer at each timestep (b, h)

        # Return a stacked tensor of size (batch_size, tgt_len, hidden_size) containing the hidden states of
        # the decoder at each timestep which would be used to generate y_hat predicted next words
        return torch.stack(hidden_states).transpose(0, 1)  # (batch_size, tgt_len, hidden_size)

    def step(self, Y_t: torch.Tensor, dec_states: torch.Tensor) -> torch.Tensor:
        """
        Computes one forward step of the RNN decoder, returns the hidden state after making an update.

        Parameters
        ----------
        Y_t : torch.Tensor
            A tensor containing the word embedding for the new target word coming in at time t of size
            (batch_size, embed_size).
        dec_states : torch.Tensor
            A tensor containing the prior hidden state for each sentence of size
            (batch_size, num_layers, hidden_size).

        Returns
        ------
        dec_state : torch.Tensor
            The updated decoder hidden state updated using the input word vector Y_t and the prior hidden
            state passed in as inputs. Returns a tensor of size (batch_size, hidden_size).
        """
        # Update the decoder hidden states for the current time step using the input word target word Y_t of
        # size (b, h) and the decoder hidden states from the prior step at time t-1 of size (b, layers, h)
        hidden_states = []  # Collect the hidden state computes at each layer, each of size (b, h)
        # Start off with the first decoder layer, the bottom-most, which takes in the Y_t next word and the
        # hidden state of the lowest layer prior
        hidden_states.append(self.decoders[0](torch.cat([Y_t, dec_states[:, 0, :]], axis=1)))

        # Then we work upwards through the layers from bottom to top and pass up the hidden state
        for i in range(1, self.num_layers):  # The inputs to each decoder cell are the prior hidden state
            # nodes from (t-1) and the hidden state of the layer below
            hidden_states.append(self.decoders[0](torch.cat([hidden_states[-1], dec_states[:, i, :]],
                                                            axis=1)))
        # Return the combined / stacked hidden states from the decoder run at this time stamp
        return torch.stack(hidden_states).transpose(0, 1)  # (batch_size, layers, hidden_size)

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
        value will be len(src_sentence) * 1.2 for each src_sentence in src_sentences. The values of
        max_decode_lengths are capped at 200 globally.

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
            amount of sub-word tokens specified here. The default is 1.2 * len(src_sentence) and all values
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
            -  negative log-likelihood score of the decoding as a float
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
        if max_decode_lengths is None:  # Default to allow for 20% more words per sentence if not specified
            max_decode_lengths = [int(len(s) * 1.2) for s in src_sentences]
        if isinstance(max_decode_lengths, int):  # Convert to a list if provided as an int
            max_decode_lengths = [max_decode_lengths for i in range(b)]
        max_decode_lengths = max_decode_lengths.copy()  # Copy to avoid mutation
        for i, n in enumerate(max_decode_lengths):  # Check all are integer valued and capped at 250
            assert isinstance(n, int) and n > 0, "All max_decode_lengths must be integers > 0"
            max_decode_lengths[i] = min(n, 250)

        msg = "src_sentences and max_decode_lengths must be the same length"
        assert len(max_decode_lengths) == len(src_sentences), msg

        # Figure out the sort order to arrange the sentences in decreasing length order
        argsort_idx = np.argsort([len(s) for i, s in enumerate(src_sentences)])[::-1]
        new_to_orig_idx = {int(x): i for i, x in enumerate(argsort_idx)}  # Reverse the mapping backwards
        src_sentences = [src_sentences[idx] for idx in argsort_idx]  # Re-order by sentence length (desc)

        self.eval()  # Set the model to eval mode so that dropout is not applied when generating values

        with torch.no_grad():  # no_grad() signals backend to throw away all gradients

            # Convert the input source sentence into a tensor object of size (b, src_len) of word indices
            src_sentence_tensor = self.vocab.src.to_input_tensor(src_sentences, self.device)  # (b, src_len)

            # Pass it through the encoder to generate the encoder hidden states for each word of each input
            # sentence and also the the decoder initial hidden state (h of t minus 1) for each sentence
            # which will in aggregate be of size (batch_size, num_layers, hidden_size)
            dec_init_state = self.encode(src_sentence_tensor, [len(s) for s in src_sentences])

            if beam_size == 1:  # Proceed with greedy search
                mt = self._greedy_search(dec_init_state, k_pct, max_decode_lengths)
            else:  # Otherwise, utilize beam search to generate output translations
                pass
                mt = [self._beam_search(dec_init_state[i, :, :], beam_size, max_decode_lengths[i])
                      for i, src_s in enumerate(src_sentences)]

        # Re-order before returning to re-instate the original sentence ordering
        mt = [mt[new_to_orig_idx[idx]] for idx in range(len(mt))]
        if tokenized is False:  # Convert the outputs into concatenated sentences to match the input format
            mt = [[util.tokens_to_str(x[0]), x[1]] for x in mt]  # Convert each to a string sentence
        return mt

    def _greedy_search(self, dec_init_state: torch.Tensor, k_pct: float,
                       max_decode_lengths: List[int]) -> List[List[Union[List[str], float]]]:
        """
        This method performs greedy search on the input source sentences provided (dec_init_state) using a
        given k-percent cutoff (k_pct). This method is built to be called only within the translate() method.

        Parameters
        ----------
        dec_init_state : torch.Tensor
            A tensor of size (batch_size, num_layers, hidden_size) corresponding to the initial state of the
            decoder based on the input src_sentences passed through the encoder.
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
        dec_state = dec_init_state  # Will be used during decoding and re-defined as we go at each step

        b = dec_state.shape[0]  # The batch_size of the inputs
        # Create output translations for each input sentence, begin with the start-of-sentence begin
        # token and also record the negative log likelihood of the sentence
        mt = [[['<s>'], 0] for _ in range(b)]  # Machine translations

        # Use the last output word Y_hat_(t-1) as the next input word (Y_t) going into the decoder, we
        # always start with the <s> sentence start token for each output translation
        Y_t = torch.tensor([self.vocab.tgt[mt[i][0][-1]] for i in range(b)],
                           dtype=torch.long, device=self.device)  # (b, )

        # Iterate until we've a complete output translations or we reach the max output len
        finished = 0  # Track how many output translation sentences are finished
        finished_flags = [0 for i in range(b)]  # Mark which sentences have been completed

        while finished < b:  # Iterate until all output translations are finished generating
            Y_t_embed = self.target_embeddings(Y_t)  # (b, embed_size) convert to a word vector

            # Compute an updated hidden state using the last y_hat and the prior hidden state
            dec_state = self.step(Y_t_embed, dec_state)
            # dec_state is a tensor with shape (batch_size, layers, hidden_size)

            # Compute the log probabilities over all possible next target words using the last hidden
            # layer i.e. the one that is to be fed to self.target_vocab_projection, gives us (b, |V|)
            # Feed in the hidden state of the top layer for each sentence:
            log_p_t = F.log_softmax(self.target_vocab_projection(dec_state[:, -1, :]), dim=-1)  # (b, |V|)

            if k_pct is None:  # Select the word with the highest modeled probability always
                # Find which word has the highest log prob for each sentence, idx = word_id in the vocab
                Y_hat_t = torch.argmax(log_p_t, dim=1)  # (b, ) the most probably next word_id for each
            else:  # Randomly sample from the sub-words at or above the kth most probably percentile
                prob_t = torch.exp(log_p_t)  # Exponentiate to convert to a prob dist (b, |V|)
                # Find what cutoff is required to make it into the words that collectively sum to form
                # the top k percent of the probability distribution i.e. for a flat distribution there
                # will be more words, for a more concentrated distribution, there will be fewer words
                # that make the cut
                Y_hat_t = torch.zeros(b, dtype=int, device=self.device)  # Start off with all zeros
                for i in range(b):
                    if finished_flags[i] == 0:  # Compute if this sentence is not already finished
                        sorted_probs = prob_t[i, :].sort(descending=True)  # Sort the probs of this dist
                        bool_vec = sorted_probs.values.cumsum(0) <= k_pct  # The entries in the top k %
                        bool_vec[0] = True  # Always have at least 1 entry set to true i.e. this happens
                        # if the most likely word has a higher prob than k
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
            Y_t = Y_hat_t  # For next iter, set the current y_hat output as the next y (b, )
            # dec_state was already updated in the step above so we do not need to do anything further
        return mt

    def _beam_search(self, dec_init_state: torch.Tensor, beam_size: int, max_decode_length: int,
                     alpha: float = 0.8) -> List[Union[List[str], float]]:
        """
        This method performs beam search on the input source sentence provided (dec_init_state) using a given
        beam size (beam_size). This method is built to be called only within the translate() method.

        Parameters
        ----------
        dec_init_state : torch.Tensor
            A tensor of size (num_layers, hidden_size) corresponding to the initial state of the decoder
            based on the input src_sentence passed through the encoder.
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
        assert len(dec_init_state.shape) == 2, "dec_init_state should be 2 dimensional"
        assert 0.6 <= alpha <= 1.0, "alpha must be between 0.5 and 1.0"
        mdl = max_decode_length  # Shorter alias

        # Maintain a list of hypotheses which can be sorted by the first element to maintain the k best where
        # k = beam_size and each records (log_prob_sum, decoded_sub_word_tokens, dec_state) with
        # decoded_sub_word_tokens being list of strings and dec_state a tensor (1, num_layers, hidden_size)
        hypotheses = [[0, ["<s>"], dec_init_state], ]  # Start off with just 1 hypothesis, the start token and
        # record the dec_init_state which can be combined with the prior token to update h_t and predict the
        # next token in the sequence

        complete_hypotheses = []  # Collect the completed hypotheses and iter until we get k = beam_size

        while len(complete_hypotheses) < beam_size:  # Iterate until we get the desired number of hypotheses
            new_hypotheses = []  # Create a new hypothesis list to replace the existing one

            # Collect together all the prior hidden states and tokens, combine them and take a step to update
            # the hidden states for making predictions
            dec_states = torch.concat([h[-1].unsqueeze(0) for h in hypotheses])  # (beam_size, n_layers, h)
            # Collect together all the word embeddings of the last word from each hypothesis (beam_size, e)
            Y_t = self.target_embeddings(torch.tensor([self.vocab.tgt[h[1][-1]] for h in hypotheses],
                                                      dtype=torch.long, device=self.device))
            # Update all the hidden states by taking 1 step
            dec_states = self.step(Y_t, dec_states)  # (beam_size, num_layers, hidden_size)

            # Compute the log probabilities over all possible next target words using the last hidden
            # layer i.e. the one that is to be fed to self.target_vocab_projection, gives us (b, |V|)
            # Feed in the hidden state of the top layer for each sentence:
            log_p_t = F.log_softmax(self.target_vocab_projection(dec_states[:, -1, :]), dim=-1)  # (b, |V|)

            # For each prior hypothesis, find the top k=beam_size ways to extend it, add each of those to the
            # new hypothesis list, which will later be sorted and pruned to retain the top k=beam_size
            log_probs, idx = torch.topk(log_p_t, k=beam_size, dim=-1, largest=True)

            for i, h in enumerate(hypotheses):  # Iter over each prior hypothesis and extend each by 1
                for j in range(beam_size):
                    new_h = [h[0] + log_probs[i, j], h[1] + [self.vocab.tgt.id2word[idx[i, j].item()]],
                             dec_states[i, :, :]]

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
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size,
                         num_layers=self.num_layers),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path: str) -> Fwd_RNN:
        """
        Method for loading in a model saved to disk.

        Parameters
        ----------
        model_path : str
            A file path detailing where the model should be saved e.g. saved_models/{model}/DeuEng/model.bin

        Returns
        -------
        model : Fwd_RNN
            Returns an object instance of this model class with the weights saved to disk.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = cls(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        return model
