# -*- coding: utf-8 -*-
"""
This module passes the input sentences for both languages from a particular data set (e.g. validation) to
the Google Translate API and records the output translations by writing them to disk as a csv.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import read_corpus, tokens_to_str, tokenize_sentences
from google.cloud import translate_v2 as translate
from typing import List
import math
from tqdm import tqdm
import pandas as pd


# Credentials are saved to: C:\Users\<USER>\AppData\Roaming\gcloud\application_default_credentials.json

def google_translate(src_sentences: List[str], src_lang: str, tgt_lang: str,
                     batch_size: int = 32) -> List[str]:
    """
    Calls the Google Translate API to translate a list of input source sentences (src_sentences) passed as
    list of strings. src_lang and tgt_lang must also be specified as either "eng" or "deu".

    Example usage:
        deu_sentences = ['Wo ist die Bank?', 'Was hast du gesagt?', 'Guten Tag.',
                         'Ich bin neunzehn Jahre alt.', "Wie geht's es dir?", "Wie viel Uhr ist es?",
                         "Wie kann ich Ihnen helfen?", "Was hast du am Wochenende gemacht?"]
        res = google_translate(deu_sentences, "deu", "eng")

        >> ['Where is the bank?', 'What did you say?', 'Good day.', 'I am nineteen years old.',
            'Are you a doctor?', 'How are you?', 'What time is it?', 'How can I help you?',
            'What did you do on the weekend?']

    Parameters
    ----------
    src_sentences : List[str]
        A list of strings where each element of the list is a sentence to be translated.
    src_lang : str
        The abbreviation of the source language e.g. "eng" or "deu".
    tgt_lang : str
        The abbreviation of the target language e.g. "eng" or "deu".
    batch_size : int
        The number of sentences to pass to the Google Translate API at once.

    Returns
    -------
    List[str]
        The src_sentences but translated to the target language.
    """
    assert src_lang != tgt_lang, "src_lang and tgt_lang must be different"
    allowed_lang = ["eng", "deu"]
    assert src_lang in allowed_lang, f"src_must be one of: {allowed_lang}"
    assert tgt_lang in allowed_lang, f"tgt_lang be one of: {allowed_lang}"
    src_lang = "en" if src_lang == "eng" else "de"
    tgt_lang = "en" if tgt_lang == "eng" else "de"
    if isinstance(src_sentences, str):
        src_sentences = [src_sentences]  # Convert to a list

    translate_client = translate.Client()  # Launch the API client instance using the default login params
    results = []  # Aggregate the results into 1 list
    n = len(src_sentences)  # The number of sentences to be processed

    n_batches = math.ceil(n / batch_size)  # How many total batches to iter over the whole data set
    for i in tqdm(range(n_batches), ncols=75):
        # Process the input data in batches so that we don't overwhelm the API
        batch = src_sentences[i * batch_size: (i + 1) * batch_size]
        batch_results = translate_client.translate(values=batch, target_language=tgt_lang,
                                                   source_language=src_lang)
        results.extend([x["translatedText"] for x in batch_results])
    return results


def translate_corpus(dataset_name: str, save_dir: str = "google_api") -> None:
    """
    Reads in the Eng and Deu versions of a data set and passes the sentences through the Google Translate API
    to generate state-of-the-art (SOTA) translations for comparison vs other models in this project.

    Parameters
    ----------
    dataset_name : str
        The name of the data set for which to generate Google API translations e.g. "validation", "test" etc.
    save_dir : str, optional
        A save directory if specified. The default is "".

    Returns
    -------
    None
        Does not return anything, writes to disk instead.
    """
    print(f"\nWorking on dataset_name='{dataset_name}'...")
    # Read in the English and German sentences from disk to be passed to the Google API
    eng_sentences = read_corpus("eng", dataset_name, is_tgt=False, tokenize=False)
    deu_sentences = read_corpus("deu", dataset_name, is_tgt=False, tokenize=False)
    output_df = pd.DataFrame({"eng": eng_sentences, "deu": deu_sentences})

    # Translate the German sentences into English using Google Translate
    output_df["eng_google"] = google_translate(deu_sentences, src_lang="deu", tgt_lang="eng")

    # Translate the English sentences into German using Google Translate
    output_df["deu_google"] = google_translate(eng_sentences, src_lang="eng", tgt_lang="deu")

    # Run these sentences through the tokenizer to ensure that they can be tokenized and de-tokenized
    # consistently as will be required later down the line in model_eval
    x = tokenize_sentences(list(output_df["eng_google"].values), lang="eng", is_tgt=False)
    output_df["eng_google"] = [tokens_to_str(sentence) for sentence in x]

    x = tokenize_sentences(list(output_df["deu_google"].values), lang="deu", is_tgt=False)
    output_df["deu_google"] = [tokens_to_str(sentence) for sentence in x]

    # Save the resulting dataframe to disk
    save_dir = f"{os.path.join(save_dir, f'{dataset_name}.csv')}"
    output_df.to_csv(save_dir, index=False)
    print(f"dataset_name='{dataset_name}' Google translate API predictions saved to: {save_dir}")


if __name__ == "__main__":
    pass  # Use with caution, we are limited in how much we can use of the API
    # translate_corpus("train_debug")
    # translate_corpus("validation")
    # translate_corpus("test")
