from typing import Any, Callable, Dict, Optional, Iterable, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from itertools import chain
from hnlp_proj.processing import FeatureType, extract_feature_lists


def combine_texts_by_author(df: pd.DataFrame) -> pd.DataFrame:
    """Given a set of texts with an "author", "text" and possibly "stanza_doc" fields,
    combines the rows by authors, returning the df indexed by author.
    """
    if "text" not in df.columns:
        raise ValueError("text column must be set")
    if "author" not in df.columns:
        raise ValueError("author column must be set")

    agg: Dict[str, Any] = {}

    keys = ["text"]
    agg["text"] = "\n\n".join
    if "stanza_doc" in df.columns:
        agg["stanza_doc"] = lambda lists: list(chain(*lists))
        keys.append("stanza_doc")

    return df.groupby(level="author")[keys].agg(agg)


def create_feature_matrix(
    df: pd.DataFrame,
    feature_column: str,
    features: Optional[Iterable[str]] = None,
    scaler_use_mean: bool = True,
    scaler_use_std: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given a dataframe containing texts(under "text" column),
    and possibly authors(under "author" column), converts it to
    a feature matrix for Burrows' Delta algorithm, using the given features, tupled with
    the counts of each feature.
    """

    if feature_column not in df.columns:
        raise ValueError(f"{feature_column} column must be set")

    if "author" in df.columns:
        # The author index will only be used for retrieving Y_test
        df.set_index("author", inplace=True)

    total_count_vector = df[feature_column].apply(len).to_numpy()

    # create feature matrix, a matrix of shape (num_authors, num_features),
    # containing term frequency of each feature(token/lemma/tag etc..)

    vectorizer = CountVectorizer(vocabulary=features, analyzer=lambda x: x)
    counts = vectorizer.fit_transform(df[feature_column]).todense()
    feats = counts / total_count_vector[:, np.newaxis]
    scaler = StandardScaler(with_mean=scaler_use_mean, with_std=scaler_use_std)
    feats = scaler.fit_transform(feats)

    return (
        pd.DataFrame(feats, columns=vectorizer.get_feature_names(), index=df.index),
        pd.DataFrame(counts, columns=vectorizer.get_feature_names(), index=df.index),
    )


def pick_most_common_words(
    df: pd.DataFrame, feature_column: str, max_features: Optional[int] = None
) -> Sequence[str]:
    if feature_column not in df.columns:
        raise ValueError(f"{feature_column} column must be set")
    ret = df[feature_column].explode(feature_column).value_counts()[:max_features]
    if max_features:
        ret = ret[:max_features]
    return ret.index


class DeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: Optional[Sequence[str]] = None,
        num_features: Optional[int] = None,
        processing: FeatureType = FeatureType.HebTokenize,
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

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None
    ) -> "DeltaTransformer":
        if self.features is not None and self.num_features:
            self.features = self.features[: self.num_features]
        elif self.features is not None:
            pass
        else:
            if self.num_features is None:
                raise ValueError("Either features or num_features must be specified")

            # pick the most common features
            X = extract_feature_lists(X, self.processing)
            self.features = pick_most_common_words(
                X,
                feature_column=self.processing.col_name(),
                max_features=self.num_features,
            )
        assert (
            self.features is not None and len(self.features) > 0
        ), "Features should be present by now"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = extract_feature_lists(X, self.processing)
        X_feat, counts = create_feature_matrix(
            X,
            feature_column=self.processing.col_name(),
            features=self.features,
            scaler_use_mean=self.center_features,
            scaler_use_std=self.standardize_features,
        )
        self.last_transformed_count = counts
        return X_feat
