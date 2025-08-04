#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reads in the parallel sentence data set from csv stored in the dataset/paired_csv directory and
splits it into separate csv files (train, validation, test) for English (eng) and Deutsch (deu) respectively.
Because the training data set is so large, it is also split into 3 equal sized subsets. A debug train subset
is also saved which is small and useful for testing.

This data set was download from: https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
"""

import pandas as pd
import numpy as np
import os, sys
BASE_PATH = os.path.abspath(os.path.dirname( __file__))

if __name__ == "__main__":
    word_count_limit = 100 # Max 100 words
    sentence_len_limit = 800 # Max 800 string characters long

    for file_name in os.listdir(os.path.join(BASE_PATH, "paired_csv")):
        print(f"\nProcessing {file_name}")
        new_file_name = file_name.split("_")[-1]
        data = pd.read_csv(os.path.join(os.path.join(BASE_PATH, "paired_csv"), file_name), engine='python')
        n_rows = data.shape[0] # Record how many sentence pairs there were originally
        data.dropna(inplace=True) # If there are NaNs, drop those entries
        data.drop_duplicates(inplace=True) # Drop duplicates if any
        # Drop sentences that are too long in either language
        word_counts_1 = data.iloc[:, 0].str.count(" ") + 1
        word_counts_2 = data.iloc[:, 1].str.count(" ") + 1
        word_length_max = pd.concat([word_counts_1, word_counts_2], axis=1).max(axis=1)
        data = data.loc[word_length_max <= word_count_limit, :] # Drop ones that are too long

        sentence_len_1 = data.iloc[:, 0].str.len()
        sentence_len_2 = data.iloc[:, 1].str.len()
        sentence_len_max = pd.concat([sentence_len_1, sentence_len_2], axis=1).max(axis=1)
        data = data.loc[sentence_len_max <= sentence_len_limit, :] # Drop ones that are too long

        if new_file_name == "train.csv": # Only apply this filter to the training dataset
            # Drop the sentences that have too much of a length disparity i.e. they should be about the same
            # size, especially for the longer ones
            word_counts_1 = data.iloc[:, 0].str.count(" ") + 1 # Compute the number of words again here
            word_counts_2 = data.iloc[:, 1].str.count(" ") + 1 # Compute the number of words again here
            ratio = np.abs(np.log(word_counts_1) - np.log(word_counts_2)) # Compute the % diff in word length
            data = data.loc[ratio <= 0.75, :] # Drop the sentence pairs that have too lengths too far apart
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


### Outputs from last run:
# Processing wmt14_translate_de-en_test.csv
# 0 rows dropped, data.shape = (3003, 2)
# wmt14_translate_de-en_test.csv processed and split into test.csv

# Processing wmt14_translate_de-en_train.csv
# 258349 rows dropped, data.shape = (4251436, 2)
# wmt14_translate_de-en_train.csv processed and split into train.csv

# Processing wmt14_translate_de-en_validation.csv
# 4 rows dropped, data.shape = (2996, 2)
# wmt14_translate_de-en_validation.csv processed and split into validation.csv

