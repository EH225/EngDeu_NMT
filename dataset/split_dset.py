#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

"""
This script reads in the parallel sentence data from csv stored in the dataset/paired_csv directory and splits
it into separate csv files (train, validation, test) for English (eng) and Deutsch (deu) respectively.
"""

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

        # Save the results split by language
        new_file_name = file_name.split("_")[-1]
        data["de"].to_csv(os.path.join("deu", new_file_name), index=False)
        data["en"].to_csv(os.path.join("eng", new_file_name), index=False)
        print(f"{file_name} processed and split into {new_file_name}")

        # For the training data set, also save down a smaller version as well
        if new_file_name == "train.csv":
            small_train_subset = data.sample(frac=0.20) # Randomly sample 20% of the original train size
            new_file_name = "train_small.csv" # Update the file name for saving
            small_train_subset["de"].to_csv(os.path.join("deu", new_file_name), index=False)
            small_train_subset["en"].to_csv(os.path.join("eng", new_file_name), index=False)
