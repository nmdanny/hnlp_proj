from typing import Optional, Iterable, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from itertools import chain
import operator

from hnlp_proj.processing import Processing, process_data


def combine_texts_by_author(df: pd.DataFrame) -> pd.DataFrame:
    """Given a set of texts with a "text", "count" and "author" fields, combines the rows
    by authors, returning the df indexed by author.
    """
    if "text" not in df.columns:
        raise ValueError("text column must be set")
    if "author" not in df.columns:
        raise ValueError("author column must be set")
    if "count" not in df.columns:
        raise ValueError("count column must be set")

    return df.groupby("author")[["text", "count"]].agg(
        {"text": lambda lists: list(chain(*lists)), "count": "sum"}
    )


def create_feature_matrix(
    df: pd.DataFrame,
    combine_by_author: bool = True,
    features: Iterable[str] = None,
    scaler_use_mean: bool = True,
    scaler_use_std: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given a dataframe containing texts(under "text" column),
    and possibly authors(under "author" column), converts it to
    a feature matrix for Burrows' Delta algorithm, using the given features, tupled with
    the counts of each feature.
    """

    if "text" not in df.columns:
        raise ValueError("text column must be set")

    if combine_by_author:
        df = combine_texts_by_author(df)
        print("combine by author")
    else:
        print("not combine by author")
        # If we're not combining texts, this is probably the evaluation set
        if "author" in df.columns:
            # The author index will only be used for retrieving Y_test
            df.set_index("author", inplace=True)
    assert "count" in df.columns, "DF must have count column"

    # create feature matrix, a matrix of shape (num_authors, num_features),
    # containing term frequency of each feature word

    # TODO: maybe use different tokenization(change the one used for "count"
    #       calculation too)
    vectorizer = CountVectorizer(vocabulary=features, analyzer=lambda x: x)
    counts = vectorizer.fit_transform(df["text"]).todense()
    feats = counts / df["count"].to_numpy()[:, np.newaxis]
    scaler = StandardScaler(with_mean=scaler_use_mean, with_std=scaler_use_std)
    feats = scaler.fit_transform(feats)

    return (
        pd.DataFrame(feats, columns=vectorizer.get_feature_names(), index=df.index),
        pd.DataFrame(counts, columns=vectorizer.get_feature_names(), index=df.index),
    )


def pick_most_common_words(
    df: pd.DataFrame, max_features: Optional[int] = None
) -> Sequence[str]:
    if "text" not in df.columns:
        raise ValueError("text column must be set")
    ret = df["text"].explode("text").value_counts()[:max_features]
    if max_features:
        ret = ret[:max_features]
    return ret.index


class DeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: Optional[Sequence[str]] = None,
        num_features: Optional[int] = None,
        processing: Processing = Processing.HebTokenize,
        center_features: bool = True,
        standardize_features: bool = True,
    ) -> None:
        self.features = features
        self.num_features = num_features
        self.processing = processing
        self.center_features = center_features
        self.standardize_features = standardize_features
        self.last_transformed_count: Optional[pd.DataFrame] = None
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        if self.features is not None and self.num_features:
            self.features = self.features[: self.num_features]
        else:
            # learn features
            X = process_data(X, self.processing)
            self.features = pick_most_common_words(X, self.num_features)
        assert (
            self.features is not None and len(self.features) > 0
        ), "Features should be present by now"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "text" not in X.columns:
            raise ValueError("text column missing in X")

        X = process_data(X, self.processing)

        # If 'y' is present, we must be training, so combine corpuses by their authors
        # Otherwise, we're transforming evaluation/prediction data, so treat each text
        # separately
        X_feat, counts = create_feature_matrix(
            X,
            combine_by_author=True,
            features=self.features,
            scaler_use_mean=self.center_features,
            scaler_use_std=self.standardize_features,
        )
        self.last_transformed_count = counts
        return X_feat
