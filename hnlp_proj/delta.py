from typing import Optional, Iterable
import numpy as np
import pandas as pd
from collections import Counter
from pandas.core.algorithms import isin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
from itertools import chain


def combine_texts_by_author(df: pd.DataFrame) -> pd.DataFrame:
    """Given a set of texts with a "text" and "author" fields,
    combines the texts of each author by two newlines,
    returning the combined texts indexed by author.
    """
    if "text" not in df.columns:
        raise ValueError("text column must be set")
    if "author" not in df.columns:
        raise ValueError("author column must be set")

    return (
        df.groupby("author")["text"]
        .apply(lambda lists: list(chain(*lists)))
        .to_frame(name="text")
    )


def add_count_column(df: pd.DataFrame):
    """Given a set of corpuses (possibly combined or not) with a "text" field,
    counts the number of tokens in each corpus and puts it into a "count" field,
    returning the newly updated data-frame."""
    # add 1 to prevent division by 0 issues.
    return df.assign(count=df["text"].apply(lambda ls: len(ls) + 1))


def create_feature_matrix(
    df: pd.DataFrame,
    combine_by_author: bool = True,
    features: Iterable[str] = None,
) -> pd.DataFrame:
    """Given a dataframe containing texts(under "text" column),
    and possibly authors(under "author" column), converts it to
    a feature matrix for Burrows' Delta algorithm, using the given features.
    """

    if "text" not in df.columns:
        raise ValueError("text column must be set")

    if combine_by_author:
        df = combine_texts_by_author(df)
    else:
        # If we're not combining texts, this is probably the evaluation set
        if "author" in df.columns:
            # The author index will only be used for retrieving Y_test
            df.set_index("author", inplace=True)

    df = add_count_column(df)

    # create feature matrix, a matrix of shape (num_authors, num_features),
    # containing term frequency of each feature word

    # TODO: maybe use different tokenization(change the one used for "count"
    #       calculation too)
    vectorizer = CountVectorizer(vocabulary=features, analyzer=lambda x: x)
    feats = (
        vectorizer.fit_transform(df["text"]).todense()
        / df["count"].to_numpy()[:, np.newaxis]
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    feats = scaler.fit_transform(feats)

    return pd.DataFrame(feats, columns=vectorizer.get_feature_names(), index=df.index)


def pick_most_common_words(df: pd.DataFrame, max_features: int) -> Iterable[str]:
    if "text" not in df.columns:
        raise ValueError("text column must be set")
    return df["text"].explode("text").value_counts()[:max_features].index
