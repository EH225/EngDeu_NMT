#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys, os
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .util import NMT, Hypothesis
from vocab.vocab import Vocab


class Fwd_RNN(NMT):
    """
    A simple forward RNN encoder and forward RNN decoder. One of the simpliest seq2seq structures there is.
    """
    def __init__(self, embed_size: int, hidden_size: int, num_layers: int, vocab: Vocab, *args, **kwargs):
        """
        Model 1a: Forward RNN model initialization.

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
        self.num_layers = num_layers
        self.embed_size = embed_size  # Record the word vector embedding dimensionality
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.name = f"Fwd_RNN_{num_layers}"

        # Create a word-embedding mapping for the source language vocab
        self.source_embeddings = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size,
                                              padding_idx=vocab.src['<pad>'])

        # Create a word-embedding mapping for the target language vocab
        self.target_embeddings = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=embed_size,
                                              padding_idx=vocab.tgt['<pad>'])

        ######################################################################################################
        ### Define the model architecture

        # Takes in the word embedding for each input word of the source language (each of size embed_size)
        # and outputs a hidden state vector of size hidden_size, this layer is encoder
        self.encoder = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                              nonlinearity="tanh",  bias=True, batch_first=True, bidirectional=False)

        # Takes in the word embedding of the prior predicted output word and rolls the prediction forward to
        # produce the predicted translation in the output language. This layer cannot be bi-directional since
        # we make y-hat predictions sequentially from left-to-right. The inputs are a concatenation of the
        # word embedding of the prior predicted word and the final context vector from the encoder
        self.decoders = [] # All layers MUST be defined as separate attributes of the model
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
            A list of source sentence tokens.
        target : List[List[str]]
            A list of target sentence toakens wrapped by <s> and </s>.

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
        dec_init_states = self.encode(source_padded, source_lengths) # (batch_size, layers, hidden_size)

        # Call the decoder using the initialized decoder hidden state and the padded target sentences to
        # generate the top layer hidden states of the decoder at each time step for each sentence
        dec_hidden_states = self.decode(dec_init_states, target_padded) # (batch_size, tgt_len, hidden_size)

        # Compute the prob distribution over the vocabulary for each prediction timestep from the decoder
        prob = F.log_softmax(self.target_vocab_projection(dec_hidden_states), dim=-1) # (b, tgt_len, V)

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

        # Pass in the packed sentense of sentences of size (batch_size, src_len, embed_dim). This returns
        # a tensor of size (batch_size, src_len, hidden_size) containing the hidden states of the RNN at each
        # timestep (i.e. word in each sentence) and also the last hidden state for each sentence encoding.
        enc_hiddens, last_hiddens = self.encoder(X_packed) # last_hiddens = (layers, b, h)
        # Switch the dimensions to be (b, layers, h), then reshape to be (b, layer * h)
        last_hiddens = last_hiddens.transpose(0, 1)
        shp = last_hiddens.shape
        last_hiddens = last_hiddens.reshape(shp[0], shp[1] * shp[2]) # Size is now (b, layer * h)

        # last_hiddens is a tensor of size (b, layers * h), pass it through the h_projection layer to compute
        # the inital states of the decoder for each layer and for each sentence in the batch
        # (b, layers * h) @ (layers * h, layers * h) = (b, layers * h)
        shp = last_hiddens.shape
        dec_init_states = self.h_projection(last_hiddens).reshape(shp[0], self.num_layers, self.hidden_size)
        return dec_init_states # (batch_size, layers, hidden_size)


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
        hidden_states : torch.tensor
            A tensor of size (batch_size, tgt_len, hidden_size) containing the hidden states of the decoder
            at each time step for each sentence.
        """
        # target_padded = target_padded[:, :-1] # Remove the <END> token for max length sentences

        # Construct a tensor Y of observed translated sentences with a shape of (b, tgt_len, e) using the
        # target model embeddings where tgt_len = maximum target sentence length and e = embedding size.
        # We use these actual translated words in our training set to score our model's predictions
        Y = self.target_embeddings(target_padded)

        dec_states = dec_init_states # Start the prior decoder states using the final hidden state from
        # the encoder, this is size (batch_size, layers, hidden_size)
        hidden_states = [] # Record the hidden states of the final top layer for each timestep, these are the
        # vectors that are used to make output y-hat word predictions

        # Use the torch.split function to iterate over the time dimension of Y, this will give us Y_t which
        # is a tensor of size (1, b, e) i.e. the word embedding of the ith target word from each sentence
        for Y_t in torch.split(tensor=Y, split_size_or_sections=1, dim=1): # Spling along dim=1 i.e. iter
            # over words in each sentence
            Y_t = torch.squeeze(Y_t, dim=1) # Squeeze Y_t into (b, e). i.e. remove the 2nd dim
            dec_states = self.step(Y_t, dec_states) # Perform a forward step of the decoder, update h_t
            # dec_states is (batch_size, layers, hidden_size)
            hidden_states.append(dec_states[:, -1, :]) # Record the top layer at each timestep (b, h)

        # Return a stacked tensor of size (batch_size, tgt_len, hidden_size) containing the hidden states of
        # the decoder at each timestep which would be used to generate y_hat predicted next words
        return torch.stack(hidden_states).transpose(0, 1) # (batch_size, tgt_len, hidden_size)


    def step(self, Y_t: torch.tensor, dec_states: torch.tensor) -> torch.Tensor:
        """
        Computes one forward step of the RNN decoder, returns the hidden state after making an update.

        Parameters
        ----------
        Y_t : torch.tensor
            A tensor containing the word embedding for the new target word coming in at time t of size
            (batch_size, embed_size).
        dec_states : torch.tensor
            A tensor containing the prior hidden state for each sentence of size (batch_size, hidden_size).

        Returns
        ------
        dec_state : torch.Tensor
            The updated decoder hidden state updated using the input word vector Y_t and the prior hidden
            state passed in as inputs. Returns a tensor of size (batch_size, hidden_size).
        """
        # Update the decoder hidden states for the current time step using the input word target word Y_t of
        # size (b, h) and the decoder hidden states from the prior step at time t-1 of size (b, layers, h)
        hidden_states = [] # Collect the hidden state computes at each layer, each of size (b, h)
        # Start off with the first decoder layer, the bottom-most, which takes in the Y_t next word and the
        # hidden state of the lowest layer prior
        hidden_states.append(self.decoders[0](torch.cat([Y_t, dec_states[:, 0, :]], axis=1)))

        # Then we work upwards through the layers from bottom to top and pass up the hidden state
        for i in range(1, self.num_layers): # The inputs to each decoder cell are the prior hidden state
            # nodes from (t-1) and the hidden state of the layer below
            hidden_states.append(self.decoders[0](torch.cat([hidden_states[-1], dec_states[:, i, :]],
                                                            axis=1)))
        # Return the combined / stacked hidden states from the decoder run at this time stamp
        return torch.stack(hidden_states).transpose(0, 1) # (batch_size, layers, hidden_size)


    def greedy_search(self, src_sentence: List[str], max_decode_length: int = 70) -> Hypothesis:
        """
        Given a single source sentence, this method performs greedy search yielding a translation in the
        target langauge by sequentially predicting the next token and always choosing the most probable
        according to the model's y-hat output distribution over the target vocab.

        Parameters
        ----------
        src_sentence : List[str]
            A single input source sentence to perform greedy search on i.e. a list of word tokens.
        max_decode_length : int, optional
            The max number of time steps to run the decoder unroll aka the max decoder prediction size.
            The default is 70 words.
        Returns
        -------
        Hypothesis
            A hypothesis with 2 fields:
                - value: List[str]: The decoded target sentence, represented as a list of words.
                - score: float: the log-likelihood of the decoded, predicted target sentence.
        """
        with torch.no_grad():  # no_grad() signals backend to throw away all gradients

            # Convert the input source sentence into a tensor object of size (1, src_len) of word indices
            src_sentence_tensor = self.vocab.src.to_input_tensor([src_sentence], self.device) # (b=1, src_len)

            # Pass it through the encoder to generate the decoder initial hidden state (h of t minus 1)
            h_tm1 = self.encode(src_sentence_tensor, [len(src_sentence)]) # (batch_size=1, layers, hidden)
            hypothesis = [['<s>'], 0] # An output translation beginning with the start sentence token

            # Use the last output word Y_hat_(t-1) as the next input word (Y_t) going into the decoder, we
            # always start with the <s> sentence start token
            Y_t = torch.tensor([self.vocab.tgt[hypothesis[0][-1]]], dtype=torch.long, device=self.device)

            t = 0 # Track the time-step evolution of the decoder
            # Iterate until we've a complete output translations or we reach the max output len
            while hypothesis[0][-1] != "</s>" and t < max_decode_length:
                t += 1 # Incriment up the timestep counter
                Y_t_embed = self.target_embeddings(Y_t) # (b=1, embed_size) convert to a word vector

                # Compute an updated hidden state using the last y_hat and the prior hidden state
                h_t = self.step(Y_t_embed, h_tm1)

                # Compute the log probabilities over all possiable next target words using the last hidden
                # layer i.e. the one that is to be fed to self.target_vocab_projection, gives us (b, |V|)
                log_p_t = F.log_softmax(self.target_vocab_projection(h_t[:, -1, :]), dim=-1).squeeze(0)
                Y_hat_t = torch.argmax(log_p_t) # Find which word has the highest log prob

                hypothesis[0].append(self.vocab.tgt.id2word[Y_hat_t.item()]) # Record the predicted next word
                hypothesis[1] += log_p_t[Y_hat_t] # Sum the log prob of all y-hats according to the model
                h_tm1 = h_t # Update the prior hidden state variable next iteration
                Y_t = Y_hat_t.unsqueeze(0) # For next iter, set the current y_hat output as the next y input

        return hypothesis


    def beam_search(self, src_sentence: List[str], beam_size: int = 5,
                    max_decode_length: int = 70) -> List[Hypothesis]:
        """
        Given a single source sentence, this method perform beam search yielding a translation in the target
        language. Beam search rolls out a few simultaneous decoder predictions and returns the one that is
        most probable according to the model.

        Note, this is generally perferred to greedy step-wise decoder prediction which may be less optimal
        over many sequence steps. What is most probable at 1 time step might not lead to the best outcomes
        later down the line, hence we track the evolution of a few and take the best among them.

        Parameters
        ----------
        src_sentence : List[str]
            A single input source sentence to perform beam search on i.e. a list of word tokens.
        beam_size : int, optional
            The number of output candidate sentences to generate (at most). The default is 5.
        max_decode_length : int, optional
            The max number of time steps to run the decoder unroll aka the max decoder prediction size.
            The default is 70 words.

        Returns
        -------
        List[Hypothesis]
            A list of hypotheses, where each hypothesis has 2 fields:
                - value: List[str]: The decoded target sentence, represented as a list of words.
                - score: float: the log-likelihood of the decoded, predicted target sentence.
        """
        ## TODO: This needs more work, something is broken internally
        # Convert the input source sentence into a tensor object of size (1, src_len) of word indices
        src_sentence_tensor = self.vocab.src.to_input_tensor([src_sentence], self.device) # (1, src_len)

        # Pass it through the encoder to generate the decoder initial hidden state
        dec_init_vec = self.encode(src_sentence_tensor, [len(src_sentence)]) # (batch_aize=1, hidden_size)

        h_tm1 = dec_init_vec # Track the hidden state of the prior time step (t minus 1)
        # eos_id = self.vocab.tgt['</s>'] # The end-of-sentence (eos) token id

        hypotheses = [['<s>']] # Create a list of potential decoder translation outputs
        # Track the likelihood scores of each hypothesized output translation as well
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = [] # Record a collection of completed output translations

        # TODO: something isn't quite right here, the beam search isn't working as it should, need to revise

        t = 0 # Track the time-step evolution of the decoder
        # Iterate until we've generated beam_size complete output translations or we reach the max output len
        while len(completed_hypotheses) < beam_size and t < max_decode_length:
            t += 1 # Incriment up the timestep counter
            # hyp_num = len(hypotheses) # The number of unfinished hypotheses current in the queue

            # For each hypothesis output translation, get a tensor of the prior ending words predicted
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long,
                                 device=self.device) # (hyp_num, hidden_size)
            # Use the prior predicted output words as the input words for the next step, convert them to
            # word embeddings to be fed into the decoder network block, perform a forward roll-out
            y_t_embed = self.target_embeddings(y_tm1) # (hyp_num, hidden_size)
            # Compute an updated hidden state using the last y_hat and the prior hidden state
            h_t = self.step(y_t_embed, h_tm1) # (hyp_num, hidden_size)

            # Compute the log probabilities over possiable next target words, for each hypothesis, compute the
            # prob of what is likely to come next over all possiable words in the target language vocab
            log_p_t = F.log_softmax(self.target_vocab_projection(h_t), dim=-1) # (hyp_num, |V|)

            live_hyp_num = beam_size - len(completed_hypotheses) # The number of translations we still want
            # Compute the log-prob of continuing each current hypothesis with each possiable next word
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            # Find the k largest log prob scores among all the unfinished output hypotheses
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            # Standard Python floor division (//) and modulo (%) operators are not supported on MPS.
            # Replace them using PyTorch functions
            # top_cand_hyp_pos // len(self.vocab.tgt)
            vocab_n = len(self.vocab.tgt)
            prev_hyp_ids = top_cand_hyp_pos.div(vocab_n, rounding_mode="floor")
            # top_cand_hyp_pos % len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos - top_cand_hyp_pos.div(vocab_n, rounding_mode="floor") * vocab_n

            # Update our collection of hypotheses
            new_hypotheses, live_hyp_ids, new_hyp_scores = [], [], []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                    top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id] # Get the word associated with this word id
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word] # Create a new hypothesis sentence
                if hyp_word == '</s>': # If the new word token being added to the end is the end sentence
                    # token, then record this hypothesis as a completed sentence, remove the start <s> and
                    # end </s> tokens and record the log prob score of the output
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else: # Otherwise add the updated hypothesis back into the hypothesis set for continuation
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size: # If we generate the desired number of output sentences
                break # stop iterating and break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = h_t[live_hyp_ids, :] # Update the prior hidden state for next time step (hyp_num, h)
            hypotheses = new_hypotheses # Update for next iteration (max size = beam_size)
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device) # (hyp_num, )

        # If we reach this point, then we have run the search process to the max allowable timestep without
        # generating the desired number of complete hypotheses or we have generated the required number of
        # hypotheses. Check if we have no completed hypotheses, if so, then use the one best so far to have 1
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item()))

        # Sort the completed hypotheses in descending order i.e. best to worse in terms of log-prob
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

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
                         num_layers=self.num_layers),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)


######################
#### UNIT TESTING ####
######################

# ### TESTING below
# vocab_save_dir = "C:/Users/EH225/Desktop/Online Coursework/CS_224N_NLP/EngDeu_NMT/vocab/deu_to_eng_vocab"
# vocab = Vocab.load(vocab_save_dir) ## NEED TO LOAD THE CORRECT VOCAB
# model = Fwd_RNN(64, 64, vocab) # Create a model instance with randomly initialized weights for development


# ## TODO: Make sure the description here is accurate. Also consider moving this to the general utils section
# src_sentence = ["ich", "bin", "froh"]
# beam_size = 5
# max_decode_length = 10
# model.beam_search(src_sentence, beam_size, max_decode_length)

