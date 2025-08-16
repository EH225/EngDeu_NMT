# -*- coding: utf-8 -*-
"""
This module contains helper functions for model evaluation e.g. functions to compute model perplexity, BLEU
scores, METEOR, BERTscores etc. and also model summary comparison tables.
"""

import models.all_models as all_models
import sentencepiece as spm
from typing import List, Tuple, Dict, Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.util import NMT
import torch, os
import util
from tqdm import tqdm
from termcolor import colored as c
import time

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

############################################
### Automatic Model Evaluation Functions ###
############################################

def compute_perplexity(model: NMT, eval_data: List[Tuple[List[str]]], batch_size: int = 32) -> float:
    """
    Computes a perplexity score of the model over an evaluation data set (eval_data). Note, this is a walk
    forward metric and is therefore a more generous evaluation metric. I.e. for each roll out step, we give
    the model the prior word from the translation provided and ask it to predict the next y_hat. At each step,
    the decoder model is prompted with the prior words from the "true" translation provided, the y_hats of the
    model are never auto-regressively fed into the model.

    Unlike the other evaluation metrics, this one requires the model as an input.

    Parameters
    ----------
    model : NMT
        A NMT model instance to be evaluated.
    eval_data : List[Tuple[List[str]]]
        A list of (src_sentence, tgt_sentence) tuples containing source and target sentences stored
        as lists of word-tokens.
    batch_size : int, optional
        The batch size to use when iterating over eval_data. The default is 32.

    Returns
    -------
    ppl : float
        A perplexity score of the model evaluated over the eval_data.
    """
    was_training = model.training # Check if the model was previously in training mode, save for later
    model.eval() # Switch the model to evaluation mode, do not track gradients

    cuml_loss = 0.0 # Track the cumulative loss over all the eval_data the model is evaluated on
    cuml_tgt_words = 0.0 # Track how many total target language output words were in the eval_data

    with torch.no_grad():  # no_grad() signals backend to throw away all gradients
        for src_sentences, tgt_sentences in util.batch_iter(eval_data, batch_size, shuffle=True):
            loss = model(src_sentences, tgt_sentences, eps=0).sum() # Compute the forward function i.e. the
            # negative log-likelihood of the output target words according to the model without any smoothing
            cuml_loss += loss.item() # Accumulate the loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sentences)  # omitting leading `<s>`
            cuml_tgt_words += tgt_word_num_to_predict # Count tgt words that are in this batch of eval_data

        ppl = np.exp(cuml_loss / cuml_tgt_words) # Compute the preplexity score of the model

    if was_training: # If the model was training, then set it back to that config before exiting
        model.train()

    return ppl


def compute_corpus_bleu_score(mt_df: pd.DataFrame, tgt_lang: str = "deu") -> float:
    """
    Computes a corpus level BLEU (Bilingual evaluation understudy) score for the input machine translation
    dataframe (mt_df) provided. See https://www.nltk.org/api/nltk.translate.bleu for details.

    This lexical-based automatic evaluation metric of translation quality utilizes n-gram matching to evaluate
    the similarity of 2 sentences in the same language by using word precision and is the most commonly cited
    machine translation evaluation metric. While simple and fast to compute, it does not allows for the usage
    of synonyms or stemming, this metric looks for exact n-gram matches and also does not account for recall
    i.e. does not consider what percent of the reference translation is matched, only how much of the
    hypothesis, hence the brevity penalty since it is easier for a higher percentage of the n-grams of the
    hypothesis to be matched if there are few of them.

    BLEU scores are obtained by computing the:
        1). n-gram precision of the machine translation hypothesis for
        n = 1, ..., 4 vs the reference sources i.e. what % of each n-gram set in the hypothesis can be matched
        to the reference(s) which is then averaged across all values of n with more weight given to the larger
        values of n

        2). applying a brevity penality for short output translations and

        3). clipping to prevent limit the number of n-gram matches that are allowed to be counted to the max
            number of their occurence in any of the reference examples provided.

    This metric essentially looks at how many of the n-grams of the hypothesis are found in one of the
    references provided. with some adjustments made along the way.

    The BLEU score per sentence ranges from [0, 1] and the corpus level score is obtained by averaging over
    all sentence scores. Sentence level BLEU scores are generally not very reliable and corpus level scores
    are preferred. Higher scores are considered better with 0.3-0.4 being good, 0.4-0.5 being considered
    high quality and 0.5-0.6 considered very high quality.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.
    tgt_lang : str
        The target language i.e. the language of the references and machine translation model outputs.

    Returns
    -------
    float
        A corpus-level BLEU score ranging from 0 to 1.
    """
    import nltk.translate
    from nltk.tokenize import word_tokenize
    # We can provide multiple references (gold-standard translations), but here we only have 1 associated
    # with each from the data set so we use only 1 reference each
    assert tgt_lang in ["eng", "deu"], "tgt_lang must be either 'eng' or 'deu'"
    lang = "english" if tgt_lang == "eng" else "german"
    return nltk.translate.bleu_score.corpus_bleu([[word_tokenize(s, language=lang)] for s in mt_df["tgt"]],
                                                 [word_tokenize(s, language=lang) for s in mt_df["mt"]])


def compute_corpus_nist_score(mt_df: pd.DataFrame, tgt_lang: str = "deu") -> float:
    """
    Computes a corpus level NIST (National Institute of Standards and Technology) score for the input machine
    translation dataframe (mt_df) provided. See https://www.nltk.org/api/nltk.translate.nist_score.html for
    details.

    This lexical-based automatic evaluation metric of translation quality utilizes n-gram matching (much like
    BLEU) to evaluate the similarity of 2 sentences in the same language by using word precision. The main
    difference of this measure vs BLEU is that n-gram matches are weighted according to their frequency of
    occurance with higher weight allocated to less frequeny n-grams (referred to as information weighting).

    While simple and fast to compute, it does not allows for the usage of synonyms or stemming, this metric
    looks for exact n-gram matches and also does not account for recall i.e. does not consider what percent
    of the reference translation is matched, only how much of the hypothesis, hence the brevity penalty since
    it is easier for a higher percentage of the n-grams of the hypothesis to be matched if there are few of
    them.

    The NIST score per sentence ranges from [0, ~12] and the corpus level score is obtained by averaging over
    all sentence scores. Sentence level NIST scores are generally not very reliable and corpus level scores
    are preferred. Higher scores are considered better with 3-5 being moderate, 5-7 being good, 7-9 being
    excellent and 9+ considered exceptional.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.
    tgt_lang : str
        The target language i.e. the language of the references and machine translation model outputs.

    Returns
    -------
    float
        A corpus-level NIST score ranging from 0 to 1.
    """
    import nltk.translate
    from nltk.tokenize import word_tokenize
    # We can provide multiple references (gold-standard translations), but here we only have 1 associated
    # with each from the data set so we use only 1 reference each
    assert tgt_lang in ["eng", "deu"], "tgt_lang must be either 'eng' or 'deu'"
    lang = "english" if tgt_lang == "eng" else "german"
    return nltk.translate.nist_score.corpus_nist([[word_tokenize(s, language=lang)] for s in mt_df["tgt"]],
                                                 [word_tokenize(s, language=lang) for s in mt_df["mt"]])


def compute_corpus_meteor_score(mt_df: pd.DataFrame, tgt_lang: str = "deu") -> float:
    """
    Computes a corpus level METEOR (Metric for Evaluation of Translation with Explicit ORdering) score for
    the input machine translation dataframe (mt_df) provided.
    See: https://www.nltk.org/api/nltk.translate.meteor_score.html for details.

    This lexical-based automatic evaluation metric of translation quality is an improvement on some of the
    short comings of BLEU and considers factors such as word order, synonyms, stemming, and exact word
    matches. But it still has a limited ability to compare the meanings of translations directly and assess
    their overall fluency. This measure tends to correlate better with human evaluation than BLEU scores.

    METEOR scores are obtained by computing a precision and recall of unigram matches by computing:
        Precision = matched unigrams / unigrams in hypothesis (% of translation content matched)
        Recall = matched unigrams / unigrams in reference (% of reference content matched)
    using synonyms and stemming to allow for more matches of similar words to be made than in BLEU. Unlike
    BLEU, METEOR has a recall component which makes it easier to balance the importance of both measures.

    METEOR is computed as the harmonic mean of precision and recall with more weight on precision:
        F-Score = (10 * P * R) / (R + 9 * P)
    There is also a penalty applied which rewards longer contigious matches.

    The METEOR score per sentence ranges from [0, 1] and the corpus level score is obtained by averaging over
    all sentence scores. Higher scores are considered better and a score of 0.5 or so indicates good
    translation quality. Above 0.6 is exceptional and below 0.4 is considered low quality.

    If running this encounters the following error: Resource wordnet not found. Then run the following:
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # For multilingual WordNet, including German and English
        nltk.download('punkt')    # For tokenization in various languages

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.
    tgt_lang : str
        The target language i.e. the language of the references and machine translation model outputs.

    Returns
    -------
    float
        A corpus-level METEOR score ranging from 0 to 1.
    """
    import nltk.translate
    from nltk.corpus import wordnet
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize
    assert tgt_lang in ["eng", "deu"], "tgt_lang must be either 'eng' or 'deu'"
    lang = "english" if tgt_lang == "eng" else "german"
    stemmer = SnowballStemmer(lang) # Initialize the word stemmer
    meteor_scores_list = [] # Record the score for each sentence
    for ref, hyp in zip([[word_tokenize(s, language=lang)] for s in mt_df["tgt"]],
                         [word_tokenize(s, language=lang) for s in mt_df["mt"]]):
        # Calculate METEOR score for each sentence pair
        score = nltk.translate.meteor_score.meteor_score(references=ref, hypothesis=hyp,
                                                         stemmer=stemmer, wordnet=wordnet)
        meteor_scores_list.append(score)
    return np.mean(meteor_scores_list)


def compute_corpus_rouge_score(mt_df: pd.DataFrame) -> float:
    """
    Computes a corpus level ROUGE score (Recall-Oriented Understudy for Gisting Evaluation) for the input
    machine translation dataframe (mt_df) provided. See https://huggingface.co/spaces/evaluate-metric/rouge
    for more details.

    This lexical-based automatic evaluation metric of translation quality combines n-gram matching with
    longest common subsequence metrics for a combined overall socre. ROUGE is often used for evaluating the
    quality of text summarization, but is also used for machine translation tasks. There are a number of
    variants used including:
        ROUGE-N: Measures the number of n-gram matches between the hypothesis and summary as a percentage of
            n-grams in the reference (recall) or hypothesis (precision). ROUGE-1 and ROUGE-2 are most common.
        ROUGE-L: Measures the length of the longest common subsequence of words between the hypothesis and
            reference and divides by the number of words in the reference (recall) or hypothesis (precision).
        ROUGE-W: Very similar to ROUGE-L but gives a higher performance score to matching subsequences with
            longer contiguous matches. This is a weighted version of the longest-common-subsequence approach.
        ROUGE-S: This metric looks at skip-bigrams i.e. bigram matches that may have some other non-matching
            word inbetween which is similar to ROUGE-2 but with a bit more flexibility.

    This function uses ROUGE-1, ROUGE-2, and ROUGE-L and returns an F1 score.

    Like many other lexical-based automatic evaluation metrics, ROUGE does not directly measure whether the
    translation is semantically correct, it looks for matching words between the hypothesis and reference
    provided and does not recognize the usage of synonyms and may penalized semantically equlivalent
    word-order differences.

    The ROUGE score per sentence ranges from [0, 1] and the corpus level score is obtained by averaging over
    all sentence-level F1 scores. Higher scores are considered better with a score of at least 0.5 being
    generally considered good.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.

    Returns
    -------
    float
        A corpus-level composite ROUGE score ranging from 0 to 1.

    """
    # ROUGE can also be computed with another package
    # from rouge_score import rouge_scorer

    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # results = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    # for idx, row in mt_df.iterrows():
    #     for metric, score in scorer.score(row["tgt"], row["mt"]).items():
    #         results[metric] += score.fmeasure

    # for key, val in results.items():
    #     results[key] = val / len(mt_df)

    # Using the evaluate package
    import evaluate
    rouge_scorer = evaluate.load('rouge') # Instantiate the ROUGE scorer obj
    results = rouge_scorer.compute(predictions=list(mt_df["mt"].values),
                                 references=[[x] for x in mt_df["tgt"].to_list()])
    return (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3


def compute_corpus_ter_score(mt_df: pd.DataFrame) -> float:
    """
    Computes a corpus level TER (Translation Edit Rate) socre for the input machine translation dataframe
    (mt_df) provided. See https://huggingface.co/spaces/evaluate-metric/ter for details.

    This lexical-based automatic evaluation metric of translation quality measures the number of edit
    operations (insertions, deletions, substitutions, and shifts of word sequences) required to convert the
    machine translation into the reference translation provided. Similar to other matching algorithms such
    as BLEU, this metric penalizes a machine translation for the usage of synonyms or differing word order
    from the provided human reference translation so it is best analyzed at the corpus level.

    The TER metric is computed as the number of edits divided by the number of words in the reference. It
    therefore computes how many edits were required as a percentage of the size of the reference itself i.e.
    more edits are expected for longer sentences (on average) given the same translation quality.

    TER does not directly measure whether the translation is semantically correct e.g. the deletion of just
    1 word (e.g. "not") could make the model translation exactly match the reference, but entirely flip the
    meaning of the sentence. While the TER would be 1 / n (generally small), the meaning could be very much
    impacted by the edit made. Not all edits have the same impact on sentence meaning.

    The TER score per sentence ranges from [0, 100+] and the corpus level score is obtained by averaging over
    all sentence scores. Lower scores (i.e. fewer edits) are considered better and a score of 65 or less
    is generally considered good. TER scores > 100 are theoredically possiable, but uncommon. E.g. if the
    hypothesis contained more words than the reference and none of them matched, then the first n would be
    substituted to match the reference and the rest thereafter would be deleted which would add up to more
    edit operations than there were reference words.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.

    Returns
    -------
    float
        A corpus-level TER score ranging from 0 to 100+.
    """
    import evaluate
    ter_metric = evaluate.load("ter") # Load the TER metric evaluator and run it on the sentences provided
    results = ter_metric.compute(predictions=list(mt_df["mt"].values),
                                 references=[[x] for x in mt_df["tgt"].to_list()])
    return results["score"]


def compute_corpus_bert_score(mt_df: pd.DataFrame, tgt_lang: str = "deu") -> float:
    """
    Computes a corpus level BERT score (Bidirectional Encoder Representations from Transformers) for the input
    machine translation dataframe (mt_df) provided. See: https://github.com/Tiiiger/bert_score for details.

    This embedding model-based automatic evaluation metric of translation quality is an improvement on many
    lexical-based methods by using a large-language model transformer (BERT) to generate deep contextual
    representations of the input sentences provided. BERT is better able to handle synonym similarity, word
    order choice, context, and paraphrasing than other simplier models. Its main limitation is that it takes
    longer to compute given the use of an LLM to generate latent representations of each input word. BERT
    scores generally correlate well with human evaluation scoring.

    BERT score is computed by passing the reference and hypothesis into the BERT model, generating deep latent
    representations of each input word from both texts separately, and then computes the cosine similarities
    between contextual embeddings of the 2 sources with a greedy-matching approach to maximize similarities.

    BERT scores combine recision and recall using F1-score and return a value between [0, 1] with 1 being
    the highest score. A score of 0.7-0.8 is generally considered good, 0.8-0.9 is considered very good, and
    above 0.9 is considered excellent.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.
    tgt_lang : str
        The target language i.e. the language of the references and machine translation model outputs.

    Returns
    -------
    float
        A corpus-level BERT score ranging from 0 to 1.
    """
    from bert_score import BERTScorer
    assert tgt_lang in ["eng", "deu"], "tgt_lang must be either 'eng' or 'deu'"
    lang = "en" if tgt_lang == "eng" else "de"
    bert_model = BERTScorer(model_type='bert-base-multilingual-cased', lang=lang, rescale_with_baseline=True)
    P, R, F1 = bert_model.score(mt_df["mt"].tolist(), mt_df["tgt"].tolist())
    return F1.mean().item() # Average across all sentences and return a float


def compute_corpus_bleurt_score(mt_df: pd.DataFrame) -> float:
    """
    Computes a corpus level BLEURT score (Bilingual Evaluation Understudy with Representations from
    Transformers) for the input machine translation dataframe (mt_df) provided.
    See: https://github.com/google-research/bleurt for details.

    This embedding model-based automatic evaluation metric of translation quality is an improvement on many
    lexical-based methods and simplier model-based methods such as BERTScore. The BLEURT model begins with a
    pre-trained BERT model, which is then trained on synthetic sentence pairs before being fine-tuned on
    data sets of human evaluations of translation quality. This makes BLEURT more correlated with human
    evaluator scorings than non-learned evaluation metrics such as BERTScore.
    See https://research.google/blog/evaluating-natural-language-generation-with-bleurt/ for details.

    BLEURT scores trypically range from -1 to +1 with higher scores being preferred. Scores of 0.7 or greater
    generally reflect strong translation quality.

    Use: pip install git+https://github.com/google-research/bleurt.git to install this repo.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.

    Returns
    -------
    float
        A corpus-level BLEURT score ranging from -1 to +1.
    """
    import evaluate
    bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="BLEURT-20") # Multi-lingual model
    results = bleurt.compute(predictions=mt_df["mt"].tolist(), references=mt_df["tgt"].tolist())
    return sum(results["scores"]) / len(results["scores"])


def compute_corpus_comet_score(mt_df: pd.DataFrame) -> float:
    """
    Computes a corpus level COMET score (Crosslingual Optimized Metric for Evaluation of Translation) for the
    input machine translation dataframe (mt_df) provided. See: https://github.com/Unbabel/COMET for details.

    This embedding, model-based automatic evaluation metric of machine translation quality uses a dual
    cross-lingual modeling technique to evaluate the quality of a hypothesis machine translation in the
    context of both the input source text and reference text. It is a neural-based approach that, unlike
    BERTScore, was trained using a supervised regression-based approach that tries to mimic the quality
    scores assigned by human evaluators by utilizing data sets of human evaluation scores. COMET scores
    therefore generally are more highly correlated with expert human evaluation scores than other methods.
    See https://aclanthology.org/2020.emnlp-main.213.pdf for details.

    This function utilizes the default model as of August 2025 (Unbabel/wmt22-comet-da) which can be found
    here: https://huggingface.co/Unbabel/wmt22-comet-da. COMET quality scores produces range from 0 to 1 with
    higher scores being preferred. Scores of 0.8 and above are generally considered to be good quality
    translations.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A machine translation dataframe containing a column "tgt" with the gold-standard translations and
        another column labeled "mt" with the machine translations from the model.

    Returns
    -------
    float
        A corpus-level COMET score ranging from 0 to 1.
    """
    from comet import download_model, load_from_checkpoint  # Import required dependency
    model_path = download_model("Unbabel/wmt22-comet-da") # Download the evaluation model
    model = load_from_checkpoint(model_path) # Load the model used for mt eval
    data = [{"src": row["src"], "mt": row["mt"], "ref": row["tgt"]} for idx, row in mt_df.iterrows()]
    model_output = model.predict(data, batch_size=32, gpus=1) # Compute evaluations for each sentence
    return model_output.system_score # Return a combined averaged score across all examples [0, 1]

    ## Evaluation can also be done using the evaluate package
    # from evaluate import load
    # comet_metric = load('comet')
    # source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
    # hypothesis = ["They were able to control the fire.", "Schools and kindergartens opened"]
    # reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
    # results = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
    # print([round(v, 1) for v in results["scores"]])


######################################################
### Model Performance Summary Generation Functions ###
######################################################

def build_eval_dataset(data_set_name: str) -> Dict[str, List[Tuple[List[str]]]]:
    """
    Builds an evaluation data set i.e. a dictionary containing the dataset to be used for automatic model
    evaluation and expected by generate_model_summary_table. The keys of the dict are"EngDeu" and "DeuEng"
    for the 2 directions of translation and the values are a list of parallel sentence tuples
    (src_sentence, tgt_sentence) where each sentence is recorded as a list of sub-word tokens.

    Parameters
    ----------
    data_set_name : str
        The name of the data set to read from disk and generate an eval dataset for e.g. "train_1" or "test".

    Returns
    -------
    Dict[str, List[Tuple[List[str]]]]
       A dictionary containing the dataset to be used for automatic model evaluation and expected by
       generate_model_summary_table. The keys of the dict are"EngDeu" and "DeuEng" for the 2 directions of
       translation and the values are a list of parallel sentence tuples (src_sentence, tgt_sentence) where
       each sentence is recorded as a list of sub-word tokens.
    """
    eval_data_dict = {}

    # EngDeu evaluation data set
    src_data = util.read_corpus("eng", data_set_name, is_tgt=False)
    tgt_data = util.read_corpus("deu", data_set_name, is_tgt=True)
    eval_data_dict["EngDeu"] = list(zip(src_data, tgt_data))

    # DeuEng evaluation data set
    src_data = util.read_corpus("deu", data_set_name, is_tgt=False)
    tgt_data = util.read_corpus("eng", data_set_name, is_tgt=True)
    eval_data_dict["DeuEng"] = list(zip(src_data, tgt_data))

    return eval_data_dict


def generate_mt_df(model: NMT, eval_data: List[Tuple[List[str]]]) -> pd.DataFrame:
    """
    Generates a machine translation dataframe for a given model, passes each input source sentence through
    the model's greedy_search method to generate output machine translations (mt).

    The output dataframe records:
        1). the source sentence
        2). the translation provided
        3). the model's output machine translation

    Parameters
    ----------
    model : NMT
        A model object to use for generating translations.
    eval_data : List[Tuple[List[str]]]
        A list of (src_sentence, tgt_sentence) tuples containing source and target sentences stored as lists
        of word-tokens.
    src_lang : str
        The language of the source sentences (e.g. "eng" or "deu").
    tgt_lang : str
        The language of the target sentences (e.g. "eng" or "deu").

    Returns
    -------
    pd.DataFrame
        Returns a DataFrame with columns ["src", "tgt", "mt"] for each input sentence pair.
    """
    # Record the outputs for each translation i.e. the source sentence (src), the target sentence (tgt) i.e.
    # the translation provided in the data set and the model translation (machine translation = mt)
    chunks = [] # Create DataFrame chuncks that will be concatenated at the end
    for src_sents, tgt_sents in util.batch_iter(eval_data, batch_size=32, shuffle=False):
        chunk = pd.DataFrame(columns=["src", "tgt", "mt"])
        chunk["src"] = [util.tokens_to_str(s) for s in src_sents] # Record the input source sentences
        chunk["tgt"] = [util.tokens_to_str(s) for s in tgt_sents] # Record the output target sentences
        # Run the input source sentences through the model and generate machine translations
        mt = model.greedy_search(src_sents)
        chunk["mt"] = [util.tokens_to_str(x[0]) for x in mt] # Record the decoded sentences
        chunks.append(chunk) # Add to the list of dataframe chuncks, one for each batch
    return pd.concat(chunks)  # Concatenate all the df chunks together and return


def print_qualitative_comparison(mt_df: pd.DataFrame) -> None:
    """
    Takes a machine translation (mt) dataframe produced by qualitative_analysis() and prints the output so
    that it is easy to qualitatively compare the model's outputs vs the eval-dataset provided translation
    for a given input source sentence.

    Parameters
    ----------
    mt_df : pd.DataFrame
        A DataFrame with columns ["src", "tgt", "mt"] for each input sentence pair containing a source
        sentence, a reference translation, and a machine translation.
    """
    for idx, row in mt_df.iterrows():  # Iterate over every row and print out a sentence comparison
        print(f"\n{c('Input', 'cyan')}: {row['src']}")
        print(f"{c('Translation', 'magenta')}: {row['tgt']}")
        print(f"{c('Model Output', 'magenta')}: {row['mt']}")


def generate_model_eval_summary(model: NMT, eval_data: List[Tuple[List[str]]]) -> pd.Series:
    """
    This function generates an evaluation summary (a pd.Series of values) for a passed model on a given
    evaluation data set (eval_data). Evaluation metrics include:
        - Perplexity
        - Bi-Lingual Evaluation Understudy (BLEU)
        - National Institute of Standards and Technology (NIST)
        - Metric for Evaluation of Translation with Explicit ORdering (METEOR)
        - Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
        - Translation Edit Rate (TER)
        - Bidirectional Encoder Representations from Transformers Score (BERT)
        - Bilingual Evaluation Understudy with Representations from Transformers (BLEURT)
        - Crosslingual Optimized Metric for Evaluation of Translation (COMET)

    Parameters
    ----------
    model : NMT
        A NMT model instance to evaluate.
    eval_data : List[Tuple[List[str]]]
        A list of (src_sentence, tgt_sentence) tuples containing source and target sentences stored as lists
        of word-tokens.

    Returns
    -------
    eval_summary : pd.Series
        A pd.Series of evaluation metric values.
    """
    # Compute mt_df for this model and use it throughout for various eval metric calculations
    mt_df = generate_mt_df(model, eval_data)

    eval_summary = pd.Series(dtype=float)  # Record in a pd.Series
    if model.name == "Google_API":
        eval_summary.loc["Perplexity"] = 0
    else:
        eval_summary.loc["Perplexity"] = compute_perplexity(model, eval_data)
    eval_summary.loc["BLEU"] = compute_corpus_bleu_score(mt_df, model.vocab.tgt_lang)
    eval_summary.loc["NIST"] = compute_corpus_nist_score(mt_df, model.vocab.tgt_lang)
    eval_summary.loc["METEOR"] = compute_corpus_meteor_score(mt_df, model.vocab.tgt_lang)
    eval_summary.loc["ROUGE"] = compute_corpus_rouge_score(mt_df)
    eval_summary.loc["TER"] = compute_corpus_ter_score(mt_df)
    eval_summary.loc["BERT"] = compute_corpus_bert_score(mt_df, model.vocab.tgt_lang)
    eval_summary.loc["BLEURT"] = compute_corpus_bleurt_score(mt_df)
    eval_summary.loc["COMET"] = compute_corpus_comet_score(mt_df)
    return eval_summary


def generate_model_summary_table(model_classes: List[str],
                                 eval_data_dict: Dict[str, List[Tuple[List[str]]]]) -> pd.DataFrame:
    """
    This function generates a comparative performance summary table across all models in model_classes using
    the same evaluation data for each model contained in eval_data_dict. The output summary table has the
    model class name as the index (rows) and for each it records A). model size in terms of Embed Size,
    Hidden Size and Total (Trainable) Params B). automatic evaluation metrics for DeuEng and C). automatic
    evaluation metrics for EngDeu.

    Parameters
    ----------
    model_classes : List[str]
        A list of model class names e.g. ['Fwd_RNN_3', 'LSTM_Att', 'LSTM_AttNN'].
    eval_data_dict : Dict[str, List[str]]
        A dictionary containing the dataset to be used for automatic model evaluation. The keys of the dict
        should be "EngDeu" and "DeuEng" for the 2 directions of translation and the values should be a list
        of parallel sentence tuples (src_sentence, tgt_sentence) where each sentence is recorded as a list of
        sub-word tokens.

    Returns
    -------
    summary_table : pd.DataFrame
        A comparative performance summary across models.
    """
    # Enumerate all the evaluation metrics used
    metrics = ["Perplexity", "BLEU", "NIST", "METEOR", "ROUGE", "TER", "BERT", "BLEURT", "COMET"]
    cols = pd.MultiIndex.from_tuples([("Model", x) for x in ["Embed Size", "Hidden Size", "Total Params"]] +
                                     [("DeuEng", x) for x in metrics] + [("EngDeu", x) for x in metrics])
    summary_table = pd.DataFrame(index=model_classes, columns=cols)

    for model_class in tqdm(model_classes, ncols=75): # Try to generate data for this model if possible
        for (src_lang, tgt_lang) in [("deu", "eng"), ("eng", "deu")]:
            translation_name = f"{src_lang.capitalize()}{tgt_lang.capitalize()}"
            model_save_dir = util.get_model_save_dir(model_class, src_lang, tgt_lang, False)
            if os.path.exists(f"{model_save_dir}/model.bin"): # Check if there is a model saved in this dir
                model = getattr(all_models, model_class).load(f"{model_save_dir}/model.bin")  # Load model
                summary_table.loc[model_class, ("Model", "Embed Size")] = model.embed_size  # Record e size
                summary_table.loc[model_class, ("Model", "Hidden Size")] = model.hidden_size  # Record h size
                col = ("Model", "Total Params") # Record the number of trainable parameters in the model
                summary_table.loc[model_class, col] = util.count_trainable_parameters(model)
                # Compute automatic performance metrics for this model using the eval data set
                model_smry = generate_model_eval_summary(model, eval_data_dict[translation_name])
                for eval_metric, metric_score in model_smry.items(): # Add each computed metric score to the
                    # summary df using the multi-index
                    summary_table.loc[model_class, (translation_name, eval_metric)] = metric_score

            elif model_class == "Google_API": # Handle this case separately, evaluate the Google Translate API
                # outputs and add them to the summary table using the same eval metrics
                summary_table.loc[model_class, ("Model", "Embed Size")] = np.nan
                summary_table.loc[model_class, ("Model", "Hidden Size")] = np.nan
                summary_table.loc[model_class, ("Model", "Total Params")] = np.nan
                model = all_models.Google_API(src_lang, tgt_lang) # Load in the model
                model_smry = generate_model_eval_summary(model, eval_data_dict[translation_name])
                for eval_metric, metric_score in model_smry.items(): # Add each computed metric score to the
                    # summary df using the multi-index
                    if eval_metric == "Perplexity": # NaN out the perplexity values from the Google API eval
                        summary_table.loc[model_class, (translation_name, eval_metric)] = np.nan
                    else: # If not perplexity, record the eval metric score as-is
                        summary_table.loc[model_class, (translation_name, eval_metric)] = metric_score

            else:  # If this model cannot be located, print a notification and move to the next one
                print(f"No {translation_name} model found for {model_class}")

    return summary_table

############################
### Qualitative Analysis ###
############################

def run_qualitative_analysis(model_class: str, src_lang: str, tgt_lang: str) -> pd.DataFrame:
    """
    Performs a quick qualitative analysis on a given model class specified for a certain translation
    direction and returns the machine translation dataframe produced in the process (mt_df)
    e.g. run_qualitative_analysis("EDTM", "deu", "eng")

    Parameters
    ----------
    model_class : str
        The name of the model class to be evaluated.
    src_lang : str
        The language of the source sentences (e.g. "eng" or "deu").
    tgt_lang : str
        The language of the target sentences (e.g. "eng" or "deu").

    Returns
    -------
    mt_df : pd.DataFrame
        Returns a DataFrame with columns ["src", "tgt", "mt"] for each input sentence pair containing a source
        sentence, a reference translation, and a machine translation.
    """
    # Construct a qualitative assessment data set
    deu_sentences = ['Wo ist die Bank?', 'Was hast du gesagt?', 'Guten Tag.', 'Ich bin neunzehn Jahre alt.',
                     "Bist du ein Doktor?", "Wie geht's es dir?", "Wie viel Uhr ist es?",
                     "Wie kann ich Ihnen helfen?", "Was hast du am Wochenende gemacht?"]

    eng_sentences = ["Where is the bank?", "What did you say?", "Good day.", "I am nineteen years old.",
                     "Are you a doctor?", "How's it going?", "What time is it?", "How can I help you?",
                     "What did you do on the weekend?"]

    src_s, tgt_s = (eng_sentences, deu_sentences) if src_lang == "eng" else (deu_sentences, eng_sentences)
    qual_eval_data = list(zip(util.tokenize_sentences(src_s, src_lang),
                              util.tokenize_sentences(tgt_s, tgt_lang)))

    mt_df = generate_mt_df(all_models.load_model(model_class, src_lang, tgt_lang), qual_eval_data)
    translation_name = f"{src_lang.capitalize()}{tgt_lang.capitalize()}"
    print(f"Qualitative Translation Quality Analysis for {model_class} - {translation_name}")
    print_qualitative_comparison(mt_df)
    return mt_df


## TODO: Get together some easy, medium, and hard sentences based on sentence length

##############################################################################################################

if __name__ == "__main__":
    # Example usage:  python model_eval.py --data-set-name=train_tiny

    import argparse
    import nltk
    # Make sure the download the things we need for model evaluation
    nltk.download('wordnet')  # Used for synonym matching
    nltk.download('omw-1.4')  # For multilingual WordNet, including German and English
    nltk.download('punkt')    # For tokenization in various languages
    nltk.download('punkt_tab')

    parser = argparse.ArgumentParser(description='Run model evaluation pipeline')
    parser.add_argument('--data-set-name', type=str, help='The name of the data set to evaluate on.')
    args = parser.parse_args()
    data_set_name = args.data_set_name

    model_classes = ['Fwd_RNN', 'LSTM_Att', 'EDTM', 'Google_API'] # All the models to evaluate

    # Generate evaluation tables for the data set i.e. one of ["train_debug", "validation", "test"]
    print(f"Running model evaluation for {model_classes} using dataset={data_set_name}")
    start_time = time.time()
    summary_table = generate_model_summary_table(model_classes, build_eval_dataset(data_set_name))
    os.makedirs(os.path.join(BASE_PATH, "eval_tables"), exist_ok=True) # Ensure this folder exists
    summary_table.to_csv(f"eval_tables/{data_set_name}_eval.csv")  # Save the computed results
    print(f"\nSummary Table for Dataset={data_set_name}")
    print(summary_table)
    print(f"Runtime: {time.time() - start_time:.2f}") # Report how long it took to run in total
