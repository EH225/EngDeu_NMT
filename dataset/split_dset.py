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
import os

if __name__ == "__main__":
    save_dir = "paired_csv"
    word_count_limit = 100 # Max 100 words
    sentence_len_limit = 800 # Max 800 string characters long

    for file_name in os.listdir(save_dir):
        print(f"\nProcessing {file_name}")
        data = pd.read_csv(os.path.join(save_dir, file_name), engine='python')
        n_rows = data.shape[0] # Record how many sentence pairs there were originally
        data.dropna(inplace=True) # If there are NaNs, drop those entries
        # Drop sentences that are too long in either language
        word_length_1 = data.iloc[:, 0].str.count(" ")
        word_length_2 = data.iloc[:, 1].str.count(" ")
        word_length_max = pd.concat([word_length_1, word_length_2], axis=1).max(axis=1)
        data = data.loc[word_length_max <= word_count_limit, :] # Drop ones that are too long

        sentence_len_1 = data.iloc[:, 0].str.len()
        sentence_len_2 = data.iloc[:, 1].str.len()
        sentence_len_max = pd.concat([sentence_len_1, sentence_len_2], axis=1).max(axis=1)
        data = data.loc[sentence_len_max <= sentence_len_limit, :] # Drop ones that are too long
        print(f"{n_rows - data.shape[0]} rows dropped")

        new_file_name = file_name.split("_")[-1]

        # For the training data set, also save down a smaller version as well
        if new_file_name == "train.csv":
            data = data.sample(frac=1.0) # Randomly shuffle the data set

            debug_train_subset = data.sample(n=3000) # Generate a small subset of 3000 for debug training
            debug_train_subset["de"].to_csv(os.path.join("deu", "train_debug.csv"), index=False)
            debug_train_subset["en"].to_csv(os.path.join("eng", "train_debug.csv"), index=False)

            data_blocks = [data.iloc[i::3, :] for i in range(3)] # Split the train data into 3 blocks
            for i, data_block in enumerate(data_blocks):
                data_block["de"].to_csv(os.path.join("deu", f"train_{i+1}.csv"), index=False)
                data_block["en"].to_csv(os.path.join("eng", f"train_{i+1}.csv"), index=False)

        else:
            # Save the results split by language
            data["de"].to_csv(os.path.join("deu", new_file_name), index=False)
            data["en"].to_csv(os.path.join("eng", new_file_name), index=False)

        print(f"{file_name} processed and split into {new_file_name}")
