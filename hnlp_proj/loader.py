import pandas as pd
from hnlp_proj.utils import clean_texts, combine_texts
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

YNET_PATH = Path(__file__).parent / "../scrape/ynet.jl"

ENG_PATH = Path(__file__).parent / "../data/victorian_large"


def load_ynet(show_html_len_plot: bool = True) -> pd.DataFrame:

    texts = pd.read_json(YNET_PATH, lines=True)
    lens = texts.text.apply(len)
    if show_html_len_plot:
        text_len_count = (
            lens.value_counts()
        )  # .rename("count").reset_index()#.rename({"index": "len"})
        sns.histplot(text_len_count).set_title(
            "Number of parsed HTML text elements in article body"
        )
        plt.show()

    texts.text = texts.text.apply(clean_texts)
    texts.text = texts.text.apply(combine_texts)
    return texts


def load_eng_test() -> pd.DataFrame:
    entries = []
    for path in ENG_PATH.glob("*"):
        [author, title] = path.stem.split("_")
        with open(path, mode="r") as content:
            text = content.read()
            entries.append({"authors": [author], "title": title, "text": text.lower()})

    return pd.DataFrame.from_records(entries)


def load_debug() -> pd.DataFrame:
    return pd.DataFrame(
        {"author": ["A", "A", "B"], "text": ["שלום עולם.", "ביי עולם.", "מה זה"]}
    )
