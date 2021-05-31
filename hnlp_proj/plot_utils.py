from typing import Iterable
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from hnlp_proj.utils import flip_hebrew_text


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
    y = [flip_hebrew_text(name) for name in df.index]
    sns.barplot(y=y, x=df["count"], orient="horizontal").set_title(title)
    plt.show()


def plot_feature_freqs(
    counts: pd.DataFrame, title: str, count_normalized: bool = False
):
    counts = counts.copy()
    counts.columns = [flip_hebrew_text(col) for col in counts.columns]
    counts.index = [flip_hebrew_text(name) for name in counts.index]
    counts.T.plot(kind="barh", stacked=True, title=title)
    plt.show()
