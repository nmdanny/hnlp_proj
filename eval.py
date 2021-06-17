# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
from sklearn.utils import resample

# %%
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, auc, accuracy_score, balanced_accuracy_score


from sklearn.linear_model import LogisticRegressionCV
import hnlp_proj.loader as loader
import hnlp_proj.processing as processing
import hnlp_proj.plot_utils as plot_utils
import hnlp_proj.delta as delta
import hnlp_proj.evaluation as evaluation
import hnlp_proj.utils as utils


# %%
DATASET_TO_PATH = {
    "ynet": "data/ynet.pickle.bz2",
    "benyehuda-articles": "data/ben_yehuda/מאמרים ומסות.pickle.bz2",
    "benyehuda-diaries": "data/ben_yehuda/זכרונות ויומנים.pickle.bz2",
    "benyehuda-prose": "data/ben_yehuda/פרוזה.pickle.bz2",
}

DATASET = "benyehuda-prose"

texts = pd.read_pickle(DATASET_TO_PATH[DATASET])
texts.authors = texts.authors.apply(utils.extract_authors)

# find all texts that only have 1 author
one_author_df = texts[texts.authors.str.len() == 1].copy()
one_author_df["authors"] = one_author_df["authors"].apply(lambda ls: ls[0])
one_author_df.rename(columns={"authors": "author"}, inplace=True)


# get rid of authors with only 1 sample
author_counts = one_author_df["author"].value_counts()
only_one_sample = author_counts[author_counts == 1].index
one_author_df = one_author_df[~one_author_df["author"].isin(only_one_sample)]

# get rid of translated texts
if "translators" in one_author_df.columns:
    one_author_df = one_author_df[one_author_df["translators"].isna()]

# %%

ev = evaluation.Evaluation(one_author_df, combine_by_authors=True, tag=DATASET)

ev.set_author_counts([10, 30, 60])
ev.add_pipeline(
    make_pipeline(
        delta.DeltaTransformer(
            processing=processing.FeatureType.HebTokenize,
            vectorizer_options=delta.VectorizerOptions(),
            num_features=50,
        ),
        LogisticRegressionCV(max_iter=10000),
    ),
    "word-unigrams-logistic-regression",
    combine_by_author=False,
)
# ev.add_pipeline(
#     make_pipeline(
#         delta.DeltaTransformer(
#             processing=processing.FeatureType.StanzaPOS,
#             vectorizer_options=delta.VectorizerOptions(),
#             num_features=50,
#         ),
#         KNeighborsClassifier(n_neighbors=1, metric="manhattan"),
#     ),
#     "pos-unigrams-knn-manhattan",
# )
# ev.add_pipeline(
#     make_pipeline(
#         delta.DeltaTransformer(
#             processing=processing.FeatureType.StanzaLemma,
#             vectorizer_options=delta.VectorizerOptions(),
#             num_features=50,
#         ),
#         KNeighborsClassifier(n_neighbors=1, metric="manhattan"),
#     ),
#     "lemma-unigrams-knn-manhattan",
# )
# ev.add_pipeline(
#     make_pipeline(
#         delta.DeltaTransformer(
#             processing=processing.FeatureType.HebTokenize,
#             vectorizer_options=delta.VectorizerOptions(ngram_range=(2, 2)),
#             num_features=50,
#         ),
#         KNeighborsClassifier(n_neighbors=2, metric="manhattan"),
#     ),
#     "word-bigrams-knn-manhattan",
# )
ev.evaluate()


# %%

results = ev.get_result_df()
evaluation.plot_eval_results(results, "f1_micro")
evaluation.plot_eval_results(results, "f1_macro")
results


# %%

ev2 = evaluation.Evaluation(one_author_df, combine_by_authors=True, tag=DATASET)

# # ev2.set_author_counts([15, 30, 45, 60])
ev2.set_author_counts([10, 20, 40, 60])
# ev2.add_pipeline(
#     make_pipeline(
#         delta.DeltaTransformer(
#             processing=processing.FeatureType.StanzaPOS,
#             vectorizer_options=delta.VectorizerOptions(ngram_range=(1, 2)),
#             num_features=None,
#         ),
#         LogisticRegressionCV(),
#     ),
#     "pos-uni-and-bigrams-lr",
#     combine_by_author=False,
# )
ev2.add_pipeline(
    make_pipeline(
        delta.DeltaTransformer(
            processing=processing.FeatureType.SplitTokenize,
            vectorizer_options=delta.VectorizerOptions(ngram_range=(2, 2)),
            num_features=100,
        ),
        KNeighborsClassifier(n_neighbors=1)
        # LogisticRegressionCV(max_iter=10000),
    ),
    "words-k-nn-euclidian",
    combine_by_author=True,
)

ev2.evaluate()

# %%

# res = pd.concat([results, ev2.get_result_df()], ignore_index=True)

# resMany = ev2.get_result_df()

evaluation.plot_eval_results(ev2.get_result_df(), metric="f1_micro")
evaluation.plot_eval_results(ev2.get_result_df(), metric="f1_macro")

# %%

# results.loc[0, "pipeline_desc"] = "word-unigrams-knn-manhattan"
# results.loc[5, "pipeline_desc"] = "words-unigrams-logistic-regression"

# %%
import seaborn as sns

# ev.get_result_df().loc[2, "counts"]

# one_author_df["text_len"].describe()

print(one_author_df.iloc[1337].text)