import pandas as pd
import numpy as np
from enum import IntEnum
from yap_wrapper import YapApi, HebTokenizer
from typing import Iterable, List
from pathlib import Path
import stanza
from functools import lru_cache, partial


YAP_IP = "localhost:8000"
YAP_API = YapApi(YAP_IP)

RESOURCE_DIR = Path(__file__).parent / "stanza_resources"


class Processing(IntEnum):
    Raw = 0
    SplitTokenize = 1
    HebTokenize = 2
    HebYap = 3
    StanzaPOS = 4
    StanzaLemma = 5


@lru_cache(maxsize=None)
def get_pipeline(processing: Processing) -> stanza.Pipeline:
    processors = ""
    if processing == Processing.StanzaPOS:
        processors = "tokenize,mwt,pos"
    elif processing == Processing.StanzaLemma:
        processors = "tokenize,mwt,pos,lemma"
    else:
        raise ValueError("Invalid processing argument", processing)
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    stanza.download("he", model_dir=str(RESOURCE_DIR))
    return stanza.Pipeline(lang="he", dir=str(RESOURCE_DIR), processors=processors)


def process_data(df: pd.DataFrame, option: Processing = Processing.Raw) -> pd.DataFrame:
    if option == Processing.Raw:
        df = df.copy()
    elif option == Processing.SplitTokenize:
        df = df.assign(text=df["text"].str.split())
    elif option == Processing.HebTokenize:
        df = df.assign(text=df["text"].apply(heb_tokenize))
    elif option == Processing.HebYap:
        df = df.assign(text=df["text"].apply(heb_yap))
    else:
        pipeline = get_pipeline(option)
        df = df.assign(docs=df["text"].apply(pipeline))
    df["count"] = df["text"].apply(len)
    return df[df["count"] > 0]


HEB_TOKENIZER = HebTokenizer()


def heb_tokenize(text: str) -> List[str]:
    return [token for _, token in HEB_TOKENIZER.tokenize(text) if token]


def heb_yap(text: str) -> List[str]:
    results = YAP_API.run(text)
    return list(results.dep_tree["lemma"])
