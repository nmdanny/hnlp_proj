import pandas as pd
import numpy as np
from enum import Enum
from yap_wrapper import YapApi, HebTokenizer
from typing import Iterable, List

YAP_IP = "localhost:8000"
YAP_API = YapApi(YAP_IP)


class Processing(Enum):
    Raw = 0
    SplitTokenize = 1
    HebTokenize = 2
    HebYap = 3


def process_data(df: pd.DataFrame, option: Processing = Processing.Raw) -> pd.DataFrame:
    if option == Processing.SplitTokenize:
        df["text"] = df["text"].str.split()
    elif option == Processing.HebTokenize:
        df["text"] = df["text"].apply(heb_tokenize)
    elif option == Processing.HebYap:
        df["text"] = df["text"].apply(heb_yap)
    return df


HEB_TOKENIZER = HebTokenizer()


def heb_tokenize(text: str) -> List[str]:
    return [token for _, token in HEB_TOKENIZER.tokenize(text)]


def heb_yap(text: str) -> List[str]:
    results = YAP_API.run(text)
    return list(results.dep_tree["lemma"])
