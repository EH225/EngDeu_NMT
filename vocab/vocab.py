#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import Counter
from docopt import docopt
from itertools import chain
import json, math
import torch
from typing import List, Tuple
import sentencepiece as spm
import numpy as np

########################
### Helper Functions ###
########################

def pad_sentences(sentences: List[List[str]], pad_token: str) -> List[List[str]]:
    """
    Pads a list of sentences on the right with a pad_token so that they are all of the same length i.e.
    the length of the longest sentence in the collection.

    This function can also be used when the inputs are List[List[int]] and int.

    Parameters
    ----------
    sentences : List[List[str]]
        A list of sentences where each sentence is represented as a list of word tokens.
    pad_token : str
        The padding token to be used to right-pad to make all sentences the same length.

    Returns
    -------
    List[List[str]]
        Returns a list of padded sentences where each sentence in the batch is the same length.
    """
    padded_sentences = []
    max_len = max([len(s) for s in sentences]) # Find how long the max length sentence is
    for s in sentences: # Add padding to each sentence as needed
        padded_sentences.append(s + [pad_token] * (max_len - len(s)))
    return padded_sentences


def read_and_tokenize_corpus(file_path: str, language_abbv: str, vocab_size: int):
    """
    Uses the SentencePiece package to tokenize and create a list of unique subwords for a given text corpus.

    Parameters
    ----------
    file_path : str
        File path to a corpus of text from a given input language to tokenize.
    language_abbv : str
        A string abbreviation of the language for naming purposes.
    vocab_size : int
        The max allowable size for the word token vocabulary created.

    Returns
    -------
    sp_list : List[List[str]]
        A list of unique word tokenized derived from the corpus used for creating a vocab.
    """
    # Fit a tokenizer to the training data, create a set of word tokens for this language
    spm.SentencePieceTrainer.train(input=file_path, model_prefix=f"{language_abbv}/{language_abbv}",
                                   vocab_size=vocab_size)
    sp = spm.SentencePieceProcessor()
    sp.load(f'{language_abbv}/{language_abbv}.model')  # loads {language_abbv}.model file generated above
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())] # The list of subwords
    return sp_list


#########################
### Class Definitions ###
#########################

class VocabEntry:
    """
    Vocabulary Entry data structure for 1 language (i.e. either the source or target).
    """
    def __init__(self, word2id: dict = None):
        """
        Instantiates a VocabEntry instance.

        Parameters
        ----------
        word2id : dict, optional
            An optional dictionary mapping of words to indices. The default is None.
        """
        if word2id is not None: # If a word-to-index mapping is provided then use it
            self.word2id = word2id
        else: # Otherwise create a new one
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1 # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>'] # Record the index id of the unknown token
        self.id2word = {v: k for k, v in self.word2id.items()} # Add a reverse mapping from int to token

    def __getitem__(self, word: str) -> int:
        """
        Retrieves a word's id. Returns the id for the <unk> token if the word is out of the vocab.

        Parameters
        ----------
        word : str
            The input word for which to return the associated id.

        Returns
        -------
        int
            Returns the index of the word requested.
        """
        return self.word2id.get(word, self.unk_id)  # Return the <unk> id if not found

    def __contains__(self, word: str) -> bool:
        """
        Checks if a given word is part of the vocabulary.

        Parameters
        ----------
        word : str
            The input word for which to check if it is known.

        Returns
        -------
        bool
            True or False indicating whether the word is known to this obj.
        """
        return word in self.word2id

    def __setitem__(self, key, value) -> None:
        """
        Raise an error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self) -> int:
        """
        Returns the size of the vocabulary i.e. the number of unique tokens.
        """
        return len(self.word2id)

    def __repr__(self) -> str:
        """
        String representation of VocabEntry to be used when printing this object.
        """
        return f'Vocabulary[size={len(self)}]'

    def id2word(self, word_id: int) -> str:
        """
        Returns the word associated with a given id if it exists, otherwise the <unk> token is returned.

        Parameters
        ----------
        word_id : int
            A word id to reverse look up the word of.

        Returns
        -------
        str
            The word token associated with the input word_id.
        """
        return self.id2word.get(word_id, '<unk>')

    def add(self, word: str) -> int:
        """
        Adds a word to the VocabEntry if it is not already part of the vocab. Returns the id assoicated with
        the word (either the new id created or the existing one if already known to this object).

        Parameters
        ----------
        word : str
            A word to be added to the vocab if it does not already exist.

        Returns
        -------
        int
            The word id of the input word internally assigned.
        """
        if word not in self.word2id: # Add to the vocab if word is not already part of it
            word_id = self.word2id[word] = len(self)
            self.id2word[word_id] = word
            return word_id
        else: # If already part of the vocab, look up the id already assigned to it
            return self[word]

    def words2indices(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Converts a list of sentences (a list of list of strings) into word ids. Also accepts an single list
        of words and returns a single list of word ids.

        Parameters
        ----------
        sentences : List[List[str]]
            A list of sentences where each sentence is a list of strings.

        Returns
        -------
        List[List[str]]
            A list of sentences where each sentence is a list of word ids.
        """
        if isinstance(sentences[0], list): # A list of sentences passed in
            return [[self[word] for word in sentence] for sentence in sentences]
        else: # Only 1 sentence passed in
            return [self[word] for word in sentences]

    def indices2words(self, word_ids: List[int]) -> List[str]:
        """
        Converts a list of word id values into a list of words.

        Parameters
        ----------
        word_ids : List[int]
            An input list of word ids.

        Returns
        -------
        List[str]
            A list of words corresponding to each input word id.
        """
        return [self.id2word[word_id] for word_id in word_ids]

    def to_input_tensor(self, sentences: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        Converts a list of sentences (words) into a tensor with the necessary padding for shorter sentences.

        Parameters
        ----------
        sentences : List[List[str]]
            A list of sentences i.e. a list of lists of strings.
        device : torch.device
            The device on to load the tensor onto i.e. CPU or GPU.

        Returns
        -------
        torch.Tensor
            A tensor of size (num_sentences, max_sentence_length) = (batch_size, max_len)
        """
        padded_sentences = pad_sentences(self.words2indices(sentences), self['<pad>'])
        sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long, device=device)
        return sentences_tensor

    @staticmethod
    def from_corpus(corpus: List[str], max_size: int, freq_cutoff: int = 2) -> VocabEntry:
        """
        Given a text corpus, construct a Vocab Entry.

        Parameters
        ----------
        corpus : List[str]
            A corpus of text produced by the read_corpus function.
        max_size : int
            The max size allowed for the vocabulary.
        freq_cutoff : int, optional
            If a word occurs n < freq_cutoff times in the corpus, drop it. The default is 2.

        Returns
        -------
        vocab_entry : VocabEntry
            A VocabEntry instance produced from the provided text corpus.
        """
        vocab_entry = VocabEntry() # Instantiate a new object instance
        word_freq = Counter(chain(*corpus)) # Count the freq of occurence for each word
        print(f"Total number of unique word tokens: {len(word_freq)}")
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff] # Drop infreq words
        print(f"Total number of unique word tokens with freq >= {freq_cutoff}: {len(valid_words)}")
        # Limit to using the max_size number of most frequently occuring tokens
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:max_size]
        for word in top_k_words: # Add each of these word tokens to the vocabulary
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_token_list(word_token_list: List[str]) -> VocabEntry:
        """
        Given a list of word tokens, construct a Vocab Entry.

        Parameters
        ----------
        subword_list : List[str]
            A list of word tokens strings.

        Returns
        -------
        vocab_entry : VocabEntry
            A VocabEntry instance produced from the provided word_token_list.
        """
        vocab_entry = VocabEntry() # Instantiate a new object instance
        for subword in word_token_list: # Fill the vocab with the word tokens provided
            vocab_entry.add(subword)
        return vocab_entry


class Vocab:
    """
    A data structure containing a VocabEntry for both the source (src) and target (tgt) language.
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """
        Instantiates the combined bi-lingual Vocab object.

        Parameters
        ----------
        src_vocab : VocabEntry
            VocabEntry for source language.
        tgt_vocab : VocabEntry
            VocabEntry for target language.
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(soruce_word_tokens: List[str], target_word_tokens: List[str]) -> Vocab:
        """
        Constructs a Vocab object instance using a list of

        Parameters
        ----------
        soruce_word_tokens : List[str]
            A list of word tokens from the source language to construct a vocab out of.
        target_word_tokens : List[str]
            A list of word tokens from the target language to construct a vocab out of.

        Returns
        -------
        Vocab
            A Vocab object instance.
        """
        src = VocabEntry.from_token_list(soruce_word_tokens)
        tgt = VocabEntry.from_token_list(target_word_tokens)
        return Vocab(src, tgt)

    def save(self, file_path: str) -> None:
        """
        Saves the Vocab object to a JSON dump.

        Parameters
        ----------
        file_path : str
            File path for where to save the JSON vocab object.
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path: str) -> Vocab:
        """
        Loads in a saved Vocab from disk stored as a JSON dumb.

        Parameters
        ----------
        file_path : str
            File path for where to read the JSON vocab object data.

        Returns
        -------
        Vocab
            A Vocab object instance.
        """
        entry = json.load(open(file_path, 'r'))
        return Vocab(VocabEntry(entry['src_word2id']), VocabEntry(entry['tgt_word2id']))

    def __repr__(self) -> str:
        """
        String representation of Vocab object when called.
        """
        return f"Vocab({len(self.src)} source words, {len(self.tgt)} target words)"

torch.serialization.add_safe_globals([VocabEntry, Vocab])

if __name__ == '__main__':
    print("Running tokenization on eng/train.csv")
    eng_tokens = read_and_tokenize_corpus("../dataset/eng/train.csv", "eng", 30000)

    print("Running tokenization on deu/train.csv")
    deu_tokens = read_and_tokenize_corpus("../dataset/deu/train.csv", "deu", 30000)

    eng_to_deu_vocab = Vocab.build(eng_tokens, deu_tokens)
    eng_to_deu_vocab.save("eng_to_deu_vocab")

    deu_to_eng_vocab = Vocab.build(deu_tokens, eng_tokens)
    deu_to_eng_vocab.save("deu_to_eng_vocab")
