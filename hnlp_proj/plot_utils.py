from typing import Iterable
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from hnlp_proj.utils import flip_hebrew_text
from hnlp_proj.delta import add_count_column


def plot_hebrew_barchart(values: pd.Series, num_entries: int, title: str):
    counts = (
        values.explode()
        .rename("count")
        .value_counts()
        .iloc[:num_entries]
        .reset_index()
        .rename(columns={"index": "value"})
    )
    counts["value"] = counts["value"].apply(flip_hebrew_text)
    sns.barplot(y=counts["value"], x=counts["count"], orient="horizontal").set_title(
        title
    )
    plt.show()


def plot_corpus_sizes(df: pd.DataFrame, title: str = "Corpus sizes"):
    df = add_count_column(df)

    sns.barplot(y=df.index, x=df["count"], orient="horizontal").set_title(title)
    plt.show()


def plot_feature_freqs(texts: pd.DataFrame, features: Iterable[str], title: str):
    counts = (
        texts["text"].str.split().explode().rename("count").value_counts().loc[features]
    )
    counts = counts.reset_index().rename(columns={"index": "value"})
    counts["value"] = counts["value"].apply(flip_hebrew_text)

    sns.barplot(y=counts["value"], x=counts["count"], orient="horizontal").set_title(
        title
    )
    plt.show()
