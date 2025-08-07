#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


class LSTM_Att(NMT):
    """
    Neural Machine Translation model comprised of:
        - A bi-directional LSTM encoder
        - A LSTM decoder with multiplicative attention
    """
    def __init__(self, embed_size: int, hidden_size: int, dropout_rate: float, vocab: Vocab, *args, **kwargs):
        """
        Bi-Directional LSTM encoder + LSTM decoder with Attention model instantiation.

        Parameters
        ----------
        embed_size : int
            The size of the word vector embeddings (dimensionality).
        hidden_size : int
            The size of the hidden states (dimensionality) used by the encoder and decoder LSTM.
        dropout_rate : float
            The dropout rate used in the attention mechanism, specifies the probability of a node being
            switched off during training, something around 0.2 is typical.
        vocab : Vocab
            A Vocabulary object containing source (src) and target (tgt) language vocabularies.
        """
        super(LSTM_Att, self).__init__()
        # assert isinstance(num_layers, int) and (1 <= num_layers <= 5), "num_layers must be an int [1, 5]"
        self.embed_size = embed_size  # Record the word vector embedding dimensionality
        self.hidden_size = hidden_size # Record the size of the hidden states used by the LSTMs
        self.dropout_rate = dropout_rate # Record the dropout rate parameter
        self.vocab = vocab # Use self.vocab.src_lang and self.vocab.tgt_lang to access the language labels
        self.name = "LSTM_Att"
        # self.lang_pair = (vocab.src_lang, vocab.tgt_lang) # Record the language pair of the translation


        ######################################################################################################
        ### Define the model architecture

        # Create a word-embedding mapping for the source language vocab
        self.source_embeddings = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size,
                                              padding_idx=vocab.src['<pad>'])

        # Create a word-embedding mapping for the target language vocab
        self.target_embeddings = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=embed_size,
                                              padding_idx=vocab.tgt['<pad>'])


        # This is the bi-directional LSTM encoder that takes in the word embedding for each input word of the
        # source language (each of size embed_size) and outputs a hidden state vector of size hidden_size and
        # a cell memory vector (also of size hidden_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bias=True,
                              batch_first=True, bidirectional=True)

        # This is the LSTM decoder section of the model that is one-directional since it is making the y_hats
        # Takes in the word embedding of the prior predicted output word and rolls the prediction forward to
        # produce the predicted translation in the output language. This layer cannot be bi-directional since
        # we make y-hat predictions sequentially from left-to-right. The inputs are a concatenation of the
        # word embedding of the prior predicted word and the final context vector from the encoder
        # The inputs are a concatenated vector of the input word y_t and the prior hidden state
        self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size, hidden_size=hidden_size, bias=True)

        # Takes in the concatenated input of:
        #   [last hidden_state of the forward LSTM] + [first hidden state of the backward LSTM]
        # which is of size hidden_size * 2 and outputs h_0 for the decoder to initialize it
        self.h_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)

        # Takes in the concatenated input of:
        #   [last cell state of the forward LSTM] + [first cell state of the backward LSTM]
        # which is of size hidden_size * 2 and outputs c_0 for the decoder to initialize it
        self.c_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)

        # This is used to compute e_{t,i} = h_{t}^{dec}^T @ W_{attProj} @ h_{i}^{enc} where h_{t}^{dec} is
        # the current hidden state of the decoder LSTM at time step t which changes each step and h_{i}^{enc}
        # is a concatenation of the forward and reverse hidden states of the bi-directional LSTM of the ith
        # input word from the original text. These 2 together are used to compute the attention scores
        self.att_projection = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)

        # Used to compute v_{t} = W_{u} @ u_{t} where u_{t} is the concatenation of h_{t}^{dec} i.e. the
        # hidden state of the decoder LSTM at time stamp t and also a_{t} which is the attention output
        # which was the weighted sum of the encoder hidden states for each input word weighted by their
        # softmax attention probability scores. So we're making y_hat predictions based on the attention
        # score outputs (which are 2h in size because they're based on bi-directional encoder hiddens) and
        # the current hidden state of the LSTM decoder (which is also of size h)
        self.combined_output_projection = nn.Linear(in_features=hidden_size * 3, out_features=hidden_size,
                                                    bias=False)

        # This is used to compute the final y-hat distribution of probabilities over the entire vocab for what
        # word token should come next. I.e. y_hat = softmax(W_{vocab} @ o_{t}) where y_hat is a length |V|
        # vector and o_{t} = dropout(tanh(v_{t})) using the v_t from above
        self.target_vocab_projection = nn.Linear(in_features=hidden_size, out_features=len(vocab.tgt),
                                                 bias=False)
        # Create a dropout layer for the attention with a probability of dropout_rate of an element being
        # zeroed during training, this helps with regularization in the network training
        self.dropout = nn.Dropout(p=dropout_rate)


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


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
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

        Returns
        -------
        scores : torch.Tensor
            A Tensor of size (batch_size, ) representing the log-likelihood of generating the target
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
        # enc_hiddens is (b, src_len, h*2) for the bi-directional hidden state of each word in each sentence
        # from the encoder, dec_init_state is a length 2 tuple containing (b, h) and (b, h) for the initial
        # states of the decoder hidden and decoder cell
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)

        # Generate a set of word masks for each source sentence so that we don't attend to padding tokens
        # in the decoder when computing attention scores
        enc_masks = self.generate_sentence_masks(enc_hiddens, source_lengths) # (b, src_len)

        # Call the decoder using the decoder initializations from above. Pass in the enc_hiddens and enc_masks
        # for computing attentions scores, pass in dec_init_state to initialize the decoder and target_padded
        # to feed in the gold-standard Y_t translation outputs at each decoder time step (b, tgt_len, h)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)

        # Compute the prob distribution over the vocabulary for each prediction timestep from the decoder
        prob = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1) # (b, tgt_len, V)

        # Zero out, probabilities for which we have nothing in the target text i.e. the padding, create a bool
        # mask of 0s and 1s by checking that each entry is not equal to the <pad> token
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating the true target words provided in this example i.e. compute
        # the cross-entropy loss by pulling out the model's y-hat values for the true target words. For each
        # word in each sentence, pull out the y_hat prob associated with the true target word at time t.
        # probs is (b, tgt_len, V) and describes the probability distribution over the next word after the
        # current time step t. I.e. the first Y_t token is <s> and the first y_hat is the distribution of
        # what the model thinks should come afterwards. Hence probs[:, :-1, :] aligns with the true Y_t words
        # target_padded[:, 1:]
        target_words_log_prob = torch.gather(prob[:, :-1, :], index=target_padded[:, 1:].unsqueeze(-1),
                                             dim=-1).squeeze(-1) # (b, tgt_len - 1) result
        # Zero out the y_hat values for the padding tokens so that they don't contribute to the sum
        target_words_log_prob = target_words_log_prob * target_masks[:, 1:] # (b, tgt_len - 1)
        return target_words_log_prob.sum(dim=1) # Return the log prob per sentence


    def encode(self, source_padded: torch.Tensor,
               source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the bi-directional LSTM encoder to a collection of padded source sentences of size (b, src_len)
        to obtain the encoder hidden states. Use them to create the initialized decoder hidden state for
        decoder translation.

        Parameters
        ----------
        source_padded : torch.Tensor
            A tensor of padded source sentences of size (b, src_len) encoded as word id integer values
            where b=batch_size and src_len = the max sentence length in the batch of source sentences. These
            have been pre-sorted in order of longest to shortest sentence.
        source_lengths : List[int]
            A list containing the length of each input sentence without padding in the batch. This list is of
            length b with max(source_lengths) == src_len.

        Returns
        -------
        enc_hiddens : torch.Tensor
            A tensor of hidden states from the encoder of shape (b, src_len, h*2), with h*2 for the 2 encoder
            directions (bi-directional).

        dec_init_state : Tuple[torch.Tensor, torch.Tensor]
            Tuple of tensors representing the decoder's initial hidden state and cell state, each of size
            (b, h) which are used to initialize the decoder.

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

        # Pass in the packed sentense of sentences of size (batch_size, src_len, embed_dim). This returns
        # a tensor of size (batch_size, src_len, hidden_size*2) containing the hidden states of the LSTM at
        # each timestep (i.e. word in each sentence) for both direcitons, it will also return the last hidden
        # and cell of the encoder each having a size of (2, b, h) for the bi-directional design
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)

        # Apply the pad_packed_sequence function to the outputs which is the inverse operation of
        # pack_padded_sequence. This function returns a tuple of a Tensor containing the padded sequence, and
        # a Tensor containing the list of lengths of each sequence in the batch. We only care about the first
        # element, not the lengths of each sequence.
        enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first=True)[0] # (b, src_len, h*2)

        # last_hidden a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        # Reshape it to be (b, 2*h) and pass it through the h_projection layer to compute init_decoder_hidden
        init_decoder_hidden = self.h_projection(last_hidden.transpose(1, 0).flatten(1, 2)) # (b, h)

        # last_cell a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        # Reshape it to be (b, 2*h) and pass it through the c_projection layer to compute init_decoder_cell
        init_decoder_cell = self.c_projection(last_cell.transpose(1, 0).flatten(1, 2)) # (b, h)

        # Combine together the initialized h_{0}^{decoder} and c_{0}^{decoder} as the decoder init state
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        # enc_hiddens = (b, src_len, h*2), (init_decoder_hidden=(b, h), init_decoder_cell=(b, h))
        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor) -> torch.Tensor:
        """
        Computes output hidden-state vectors for each word in each batch of target sentences i.e. runs the
        decoder to generate the output sequence of hidden states for each word of each sentence while using
        the true Y_t words provided in the target translation as inputs at each time step instead of the the
        prior Y_hat_values provided from the prior step. This method is used for training only.

        Parameters
        ----------
        enc_hiddens : torch.Tensor
            A tensor of bi-directional hidden states from the encoder of size (b, src_len, h*2).
        enc_masks : torch.Tensor
            A tensor of sentence masks (0s and 1s) for masking out the padding tokens of size (b, src_len).
        dec_init_state : Tuple[torch.Tensor, torch.Tensor]
            A tuple containing 2 (b, h) tensors to initialize the hidden state and cell of the decoder.
        target_padded : torch.Tensor
            Gold-standard padded target sentences of size (b, tgt_len) i.e. good translations of the inputs.

        Returns
        -------
        combined_outputs : torch.Tensor
            Returns a tensor of combined outputs that are used to make y_hat predictions of size
            (b, tgt_len, h) which incorporates info from attention and the decoder hidden state.
        """
        # target_padded = target_padded[:, :-1] # Remove the <END> token for max length sentences

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step, will return at end
        combined_outputs = []

        # Apply the attention projection layer to enc_hiddens to compute enc_hiddens_proj
        # Compute: e_{t,i} = h_{t}^{dec}^T @ W_{attProj} @ h_{i}^{enc}
        enc_hiddens_proj = self.att_projection(enc_hiddens) # Outputs a tensor that is (b, src_len, h)
        # Compute this 1x per iteration here so that we do not need to duplicate the calculation each step

        # Construct a tensor Y of observed translated sentences with a shape of (b, tgt_len, e) using the
        # target model embeddings where tgt_len = maximum target sentence length and e = embedding size.
        # We use these actual translated words in our training set to score our model's predictions
        Y = self.target_embeddings(target_padded) # (b, tgt_len, e)

        # Use the torch.split function to iterate over the time dimension of Y, this will give us Y_t which
        # is a tensor of size (1, b, e) i.e. the word embedding of the ith target word from each sentence
        for Y_t in torch.split(tensor=Y, split_size_or_sections=1, dim=1): # Spling along dim=1 i.e. iter
            # over words in each sentence
            Y_t = torch.squeeze(Y_t, dim=1) # Squeeze Y_t into (b, e). i.e. remove the 2nd dim
            # Construct Ybar_t by concatenating Y_t with o_prev on their last dimension, Y_t is (b, e) and
            # o_prev is (b, h) so we get a result that is of size (b, e + h)
            Ybar_t = torch.cat(tensors=(Y_t, o_prev), dim=1)

            # Pass in Ybar_t = concat(Y_t, o_{t-1}), dec_state = [h_{t-1}^{decoder}, c_{t-1}^{decoder}] along
            # with enc_hiddens = all the encoder hidden states for each word in the x input, and also
            # enc_hiddens_proj which is W_{attProj} @ h_{i}^{enc} and get back the new updated dec_state
            # which is [h_{t}^{decoder}, c_{t}^{decoder}] along with the updated combined-output vector o_t
            # and also e_t of size (b, src_len) which are the attention scores distribution.
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)

            combined_outputs.append(o_t) # Append o_t to a list of all such vectors
            o_prev = o_t # Update for next iteration

        # Use torch.stack to convert combined_outputs from a list length tgt_len of tensors shape (b, h), to
        # a single tensor shape (b, tgt_len, h) where tgt_len = maximum target sentence length, b = batch
        # size, h = hidden size.
        combined_outputs = torch.stack(combined_outputs).transpose(0, 1) # (batch_size, tgt_len, hidden_size)
        return combined_outputs


    def step(self, Ybar_t: torch.Tensor, dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor, enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """
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
        Method for loading in model weights saved locally to disk.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model = cls(vocab=params['vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, model_path: str):
        """
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
