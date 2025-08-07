#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reads in the parallel sentence data set from csv stored in the dataset/paired_csv directory and
splits it into separate csv files (train, validation, test) for English (eng) and Deutsch (deu) respectively.
Because the training data set is so large, it is also split into 3 equal sized subsets. A debug train subset
is also saved which is small and useful for testing.

This data set was download from: https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german

The German word list was downloaded from:
    https://gist.github.com/MarvinJWendt/2f4f4154b8ae218600eb091a5706b5f4

The English word list was downloaded from:
    https://gist.github.com/h3xx/1976236
    https://raw.githubusercontent.com/arstgit/high-frequency-vocabulary/refs/heads/master/30k.txt
"""

import pandas as pd
import numpy as np
import os, sys, string
BASE_PATH = os.path.abspath(os.path.dirname( __file__))

def pct_words_recognized(sentence: str, word_set: set) -> float:
    """
    The input sentence is split into words on a basis of white space. Then each "word" is searched for in
    the word_set and the percent of total words recognized as being elements of the word set is returned.
    E.g. if 7 / 10 words from sentence are part of word_set, then 0.7 is returned.

    Matching comparisons are done after removing punctuation and converting to lowercase.

    Parameters
    ----------
    sentence : str
        An input sentence as a string.
    word_set : set
        A set of strings defining the word set of recognized words.

    Returns
    -------
    float
        A float value [0, 1] denoting what percentage of the input sentence words were located in word_set.
    """
    # Remove punctuation, replace wiht a space, then split on white spaces to create a list of words
    words = sentence.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).split()
    if len(words) > 0:
        return sum([1 for w in words if w.lower() in word_set]) / len(words)
    else:
        return 0


if __name__ == "__main__":
    word_count_limit = 100 # Max 100 words
    sentence_len_limit = 800 # Max 800 string characters long

    # Read in a set of unique English words commonly used
    eng_words = pd.read_csv(os.path.join(BASE_PATH, "lang_word_lists/english_words.txt")).iloc[:, 0]
    eng_words = set(eng_words.str.lower())

    # Read in a set of unique German words commonly used
    deu_words = pd.read_csv(os.path.join(BASE_PATH, "lang_word_lists/german_words.txt")).iloc[:, 0]
    deu_words = set(deu_words.str.lower())

    for file_name in os.listdir(os.path.join(BASE_PATH, "paired_csv")):
        print(f"\nProcessing {file_name}")
        new_file_name = file_name.split("_")[-1]
        data = pd.read_csv(os.path.join(os.path.join(BASE_PATH, "paired_csv"), file_name), engine='python')
        n_rows = data.shape[0] # Record how many sentence pairs there were originally
        data.dropna(inplace=True) # If there are NaNs, drop those entries
        data.drop_duplicates(inplace=True) # Drop duplicates if any
        print(f"{n_rows - data.shape[0]} rows dropped due to duplicates and NaNs")
        # Drop sentences that are too long in either language
        word_counts_1 = data.iloc[:, 0].str.count(" ") + 1
        word_counts_2 = data.iloc[:, 1].str.count(" ") + 1
        word_length_max = pd.concat([word_counts_1, word_counts_2], axis=1).max(axis=1)
        print(f"Dropping {(word_length_max > word_count_limit).sum()} rows due to word count limit")
        data = data.loc[word_length_max <= word_count_limit, :] # Drop ones that are too long

        sentence_len_1 = data.iloc[:, 0].str.len()
        sentence_len_2 = data.iloc[:, 1].str.len()
        sentence_len_max = pd.concat([sentence_len_1, sentence_len_2], axis=1).max(axis=1)
        print(f"Dropping {(sentence_len_max > sentence_len_limit).sum()} rows due to string len limit")
        data = data.loc[sentence_len_max <= sentence_len_limit, :] # Drop ones that are too long

        if new_file_name == "train.csv": # Only apply this filter to the training dataset
            # Drop the sentences that have too much of a length disparity i.e. they should be about the same
            # size, especially for the longer ones
            word_counts_1 = data.iloc[:, 0].str.count(" ") + 1 # Compute the number of words again here
            word_counts_2 = data.iloc[:, 1].str.count(" ") + 1 # Compute the number of words again here
            ratio = np.abs(np.log(word_counts_1) - np.log(word_counts_2)) # Compute the % diff in word length
            print(f"Dropping {(ratio > 0.75).sum()} rows due to sentence lengths differing too much")
            data = data.loc[ratio <= 0.75, :] # Drop the sentence pairs that have too lengths too far apart

            # Drop sentences where the number of recognized words isn't high i.e. if the English doens't look
            # like English or the German doesn't look like German, then drop it
            word_pct_eng = data["en"].apply(pct_words_recognized, word_set=eng_words)
            word_pct_deu = data["de"].apply(pct_words_recognized, word_set=deu_words)
            drop_vec = (word_pct_eng < 0.4) | (word_pct_deu < 0.4) # If either lang is too unrecognized, drop
            print(f"Dropping {(drop_vec).sum()} rows due to unrecognized words")
            data = data.loc[~drop_vec, :] # Drop sentence pairs that have too many unrecognized words

        print(f"{n_rows - data.shape[0]} rows dropped, data.shape = {data.shape}")

        # For the training data set, also save down a smaller version as well
        if new_file_name == "train.csv":
            data = data.sample(frac=1.0) # Randomly shuffle the data set

            # Generate a small subset of 3000 for debug training
            debug_train_subset = data.sample(n=3000, replace=False)
            debug_train_subset["de"].to_csv(os.path.join(BASE_PATH, "deu", "train_debug.csv"), index=False)
            debug_train_subset["en"].to_csv(os.path.join(BASE_PATH, "eng", "train_debug.csv"), index=False)

            data_blocks = [data.iloc[i::3, :] for i in range(3)] # Split the train data into 3 blocks
            for i, data_block in enumerate(data_blocks):
                data_block["de"].to_csv(os.path.join(BASE_PATH, "deu", f"train_{i+1}.csv"), index=False)
                data_block["en"].to_csv(os.path.join(BASE_PATH, "eng", f"train_{i+1}.csv"), index=False)

        else:
            # Save the results split by language
            data["de"].to_csv(os.path.join(BASE_PATH, "deu", new_file_name), index=False)
            data["en"].to_csv(os.path.join(BASE_PATH, "eng", new_file_name), index=False)

        print(f"{file_name} processed and split into {new_file_name}")

#############################
### Outputs from last run ###
#############################

# Processing wmt14_translate_de-en_test.csv
# 0 rows dropped due to duplicates and NaNs
# Dropping 0 rows due to word count limit
# Dropping 0 rows due to string len limit
# 0 rows dropped, data.shape = (3003, 2)
# wmt14_translate_de-en_test.csv processed and split into test.csv

# Processing wmt14_translate_de-en_train.csv
# 42312 rows dropped due to duplicates and NaNs
# Dropping 6746 rows due to word count limit
# Dropping 103 rows due to string len limit
# Dropping 209921 rows due to sentence lengths differing too much
# Dropping 121893 rows due to unrecognized words
# 380975 rows dropped, data.shape = (4128810, 2)
# wmt14_translate_de-en_train.csv processed and split into train.csv

# Processing wmt14_translate_de-en_validation.csv
# 4 rows dropped due to duplicates and NaNs
# Dropping 0 rows due to word count limit
# Dropping 0 rows due to string len limit
# 4 rows dropped, data.shape = (2996, 2)
# wmt14_translate_de-en_validation.csv processed and split into validation.csv
