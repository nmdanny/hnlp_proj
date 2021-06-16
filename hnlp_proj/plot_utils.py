from typing import Iterable
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch import flip
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


def plot_text_length_histogram_per_author(df: pd.DataFrame):
    df = df.set_index("author", drop=False).assign(lengths=df["text"].apply(len))
    df["author"] = [flip_hebrew_text(name) for name in df.author]
    ax = sns.histplot(data=df, hue="author", x="lengths", multiple="stack")
    ax.set_title("Text length histogram per author")
    ax.set(xlabel="Text length(characters)")
    plt.show()


def plot_text_length_histogram_per_category(df: pd.DataFrame, **plot_kwargs):
    df = df.set_index("category", drop=False)
    df["lengths"] = df["text"].str.len()
    df["category"] = [flip_hebrew_text(name) for name in df.category]
    ax = sns.histplot(
        data=df, hue="category", x="lengths", multiple="stack", **plot_kwargs
    )
    ax.set_title("Text length histogram per category")
    ax.set(xlabel="Text length(characters)")
    plt.show()


def plot_total_subcorpus_length_per_author(df: pd.DataFrame):
    df = df.assign(token_lengths=df.text.str.split().apply(len))
    lengths = df.groupby(by="author", level=0)["token_lengths"].sum()
    lengths.index = [flip_hebrew_text(name) for name in lengths.index]
    ax = sns.barplot(y=lengths.index, x=lengths, orient="horizontal")
    ax.set_title("Total subcorpus size per author")
    ax.set(xlabel="Number of tokens(whitespace delimited)")
    plt.show()


def plot_corpus_sizes(df: pd.DataFrame, title: str = "Corpus sizes"):
    y = [flip_hebrew_text(name) for name in df.index]
    df = df.assign(count=df["text"].apply(len))
    sns.barplot(y=y, x=df["count"], orient="horizontal").set_title(title)
    plt.show()


def plot_feature_freqs(counts: pd.DataFrame, title: str):
    counts = counts.copy()
    counts.columns = [flip_hebrew_text(str(col)) for col in counts.columns]
    counts.index = [flip_hebrew_text(str(name)) for name in counts.index]
    counts.T.plot(kind="barh", stacked=True, title=title)
    plt.show()
