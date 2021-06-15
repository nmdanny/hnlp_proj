from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Iterable,
    Sequence,
    Tuple,
    Union,
)
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


@dataclass
class VectorizerOptions:
    ngram_range: Tuple[int, int] = field(default=(1, 1))
    analyzer: Union[Literal["word"], Literal["char"]] = field(default="word")

    def create_vectorizer(self, **kwargs: Dict[str, Any]) -> CountVectorizer:
        vectorizer = CountVectorizer(
            token_pattern=r"(?u)\S+",
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            **kwargs,
        )
        return vectorizer


def create_feature_matrix(
    df: pd.DataFrame,
    feature_column: str,
    vectorizer: CountVectorizer,
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

    counts = vectorizer.transform(df[feature_column]).todense()
    feats = counts / total_count_vector[:, np.newaxis]
    scaler = StandardScaler(with_mean=scaler_use_mean, with_std=scaler_use_std)
    feats = scaler.fit_transform(feats)

    return (
        pd.DataFrame(feats, columns=vectorizer.get_feature_names(), index=df.index),
        pd.DataFrame(counts, columns=vectorizer.get_feature_names(), index=df.index),
    )


class DeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        vectorizer_options: VectorizerOptions,
        features: Optional[Sequence[str]] = None,
        num_features: Optional[int] = None,
        processing: FeatureType = FeatureType.HebTokenize,
        center_features: bool = True,
        standardize_features: bool = True,
    ) -> None:
        self.vectorizer_options = vectorizer_options
        self.features = features
        self.num_features = num_features
        self.processing = processing
        self.center_features = center_features
        self.standardize_features = standardize_features
        self.last_transformed_count: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[CountVectorizer] = None
        super().__init__()

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None
    ) -> "DeltaTransformer":
        vec_args: Dict[str, Any] = {}
        if self.features:
            self.features = (
                self.features[: self.num_features]
                if self.num_features
                else self.features[:]
            )
            vec_args["vocabulary"] = self.features
        elif self.num_features:
            vec_args["max_features"] = self.num_features
        X = extract_feature_lists(X, self.processing)
        self.vectorizer = self.vectorizer_options.create_vectorizer(**vec_args).fit(
            X[self.processing.col_name()]
        )
        self.features = self.vectorizer.get_feature_names()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = extract_feature_lists(X, self.processing)
        X_feat, counts = create_feature_matrix(
            X,
            feature_column=self.processing.col_name(),
            vectorizer=self.vectorizer,
            scaler_use_mean=self.center_features,
            scaler_use_std=self.standardize_features,
        )
        self.last_transformed_count = counts
        return X_feat
