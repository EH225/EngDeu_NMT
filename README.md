# EngDeu Neural Machine Translation Project Repo
This repository contains code for the English-German neural machine translation project (NMT). Below is a quick overview of the repo layout:
- `dataset/`: This folder contains the data set used for this project which is too large to maintain in this repo, but can be freely downloaded from Kaggle: https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german. Run `python dataset/split_dset` after saving the paired data into the `dataset/` directory to prepare it for use. See `dataset/split_dset` for details.
- `eval_tables/`: This folder contains evaluation tables that record the performance of each model across a variety of data-subsets using a variety of automatic evaluation metrics. Results are organized within this folder by decode algo (i.e. greedy or beam search).
- `google_api/`: This folder contains a script that is able to generate output translations by sending queries to the Google Translate API. This folder also contained the cached results of doing so on various data-subsets.
- `model_pred/`: This folder contained a module (`cache_predictions.py`) that caches model predictions to this directory for various data-subsets.
- `models/`: This folder contains the definition and implementation of each model class. A module named `all_models.py` imports all of them so it can be used to quickly import all.
- `venv_req/`: This folder contains a `requirements.txt` file specifying the virtual environment requirements for running this repo.
- `vocab/`: This folder contains `vocab.py` which generates the cached sub-word tokenizer vocabs for each language. Those cached tokenizer files are also stored in this directory.
- `model_eval.py`: This module contains code related to evaluating model performance and generating the evaluation summaries found in `eval_tables/`.
- `train.py`: This module contains code related to training the models.
- `util.py`: This module contains general utility functions used throughout the repo.

This project leveraged materials from Stanford University's Natural Language Processing with Deep Learning ([XCS231N](https://web.stanford.edu/class/cs224n/)) course, with many modifications.