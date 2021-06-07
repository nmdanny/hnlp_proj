import pandas as pd
from hnlp_proj.utils import clean_texts, combine_texts
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re
from pathlib import Path
import pickle
import bz2

YNET_PATH = Path(__file__).parent / "../data/ynet.jl"

ENG_PATH = Path(__file__).parent / "../data/victorian_large"

BEN_YEHUDA_PATH = Path(__file__).parent / "../data/public_domain_dump-master.zip"

YNET_STANZA_PICKLE = Path(__file__).parent / "../data/ynet.pickle.bz2"

BEN_YEHUDA_STANZA_PICKLE = (
    Path(__file__).parent / "../data/public_domain_dump-master.pickle.bz2"
)


def combine_author_corpora(df: pd.DataFrame) -> pd.DataFrame:
    """Combines all texts(by concatenating along with newlines)
    in of every author in given dataframe, returning the combined data-frame
    """
    if "text" not in df.columns:
        raise ValueError("text column missing")
    if "author" not in df.columns:
        raise ValueError("author column missing")
    return df.groupby("author")["text"].apply("\n\n".join).reset_index()


def load_ynet(show_html_len_plot: bool = False, with_pickle=False) -> pd.DataFrame:

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
    texts.text = texts.text.str.strip()
    texts = texts[texts.text.astype(bool)]

    return texts


def attach_pickle(texts: pd.DataFrame, pickle_path: Path):
    if not pickle_path.exists():
        raise ValueError(f"pickle file {pickle_path} doesn't exist")
    with bz2.open(pickle_path, "rb") as pickle_f:
        docs = pickle.load(pickle_f)
        assert len(docs) == len(
            texts
        ), f"Number of documents in pickle file({len(docs)}) doesn't match DF length({len(texts)}) "
        texts["docs"] = docs


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


BEN_YEHUDA_JUNK_REGEX = re.compile(r"את הטקסט לעיל הפיקו מתנדבי .*", re.UNICODE)


def cleanup_ben_yehuda(txt: str) -> str:
    return BEN_YEHUDA_JUNK_REGEX.sub("", txt)


def load_ben_yehuda() -> pd.DataFrame:
    import zipfile

    ZIP_ROOT_FOLDER = "public_domain_dump-master"
    with zipfile.ZipFile(BEN_YEHUDA_PATH, "r") as zipf:

        def load_text(path: str) -> str:
            with zipf.open(f"{ZIP_ROOT_FOLDER}/txt_stripped{path}.txt", "r") as f:
                txt = io.TextIOWrapper(f, encoding="utf-8").read()
                txt = cleanup_ben_yehuda(txt)
                return txt

        with zipf.open(f"{ZIP_ROOT_FOLDER}/pseudocatalogue.csv") as csv:
            catalog: pd.DataFrame = pd.read_csv(csv)
            catalog["authors"] = catalog.authors.apply(lambda authors: [authors])
            catalog.rename(columns={"genre": "category"}, inplace=True)
            catalog["text"] = catalog.path.apply(load_text)
            catalog.text = catalog.text.str.strip()
            catalog = catalog[catalog.text.astype(bool)]
            return catalog
