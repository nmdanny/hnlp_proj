import pandas as pd
import numpy as np
from enum import auto, Enum
from yap_wrapper import YapApi, HebTokenizer
from typing import Iterable, List, Mapping, Any, Tuple
from pathlib import Path
import stanza
from stanza.models.common.doc import Document
from functools import lru_cache, partial


YAP_IP = "localhost:8000"
YAP_API = YapApi(YAP_IP)

RESOURCE_DIR = Path(__file__).parent / "stanza_resources"


class FeatureType(Enum):
    """Defines features that can be used in our pipeline"""

    SplitTokenize = auto()
    HebTokenize = auto()
    YapLemmas = auto()
    StanzaPOS = auto()
    StanzaLemma = auto()

    def col_name(self) -> str:
        if self in (FeatureType.SplitTokenize, FeatureType.HebTokenize):
            return "tokens"
        if self in (FeatureType.YapLemmas, FeatureType.StanzaLemma):
            return "lemmas"
        if self == FeatureType.StanzaPOS:
            return "pos"
        raise ValueError("Impossible, invalid processing value", self)


@lru_cache(maxsize=None)
def get_stanza_pipeline(
    feature_type: FeatureType, **kwargs: Mapping[str, Any]
) -> stanza.Pipeline:
    processors = ""
    if feature_type == FeatureType.StanzaPOS:
        processors = "tokenize,mwt,pos"
    elif feature_type == FeatureType.StanzaLemma:
        processors = "tokenize,mwt,pos,lemma"
    else:
        raise ValueError(
            "Invalid processing argument, must be for stanza", feature_type
        )
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    stanza.download("he", model_dir=str(RESOURCE_DIR))
    return stanza.Pipeline(
        lang="he", dir=str(RESOURCE_DIR), processors=processors, **kwargs
    )


def extract_feature_lists(df: pd.DataFrame, feature_type: FeatureType) -> pd.DataFrame:
    """Given a dataframe containing 'text' and possibly 'stanza_doc' columns,
    adds the appropriate feature column(all elements, joined by spaces)
    in a newly returned data-frame."""
    if feature_type == FeatureType.SplitTokenize:
        df = df.assign(tokens=df["text"])
    elif feature_type == FeatureType.HebTokenize:
        df = df.assign(tokens=df["text"].apply(heb_tokenize))
    elif feature_type == FeatureType.YapLemmas:
        df = df.assign(lemmas=df["text"].apply(extract_yap_lemmas))
    elif feature_type == FeatureType.StanzaLemma:
        df = df.assign(lemmas=df["stanza_doc"].apply(extract_stanza_lemmas))
    elif feature_type == FeatureType.StanzaPOS:
        df = df.assign(pos=df["stanza_doc"].apply(extract_stanza_pos))

    return df


HEB_TOKENIZER = HebTokenizer()


def heb_tokenize(text: str) -> str:
    return " ".join(token for _, token in HEB_TOKENIZER.tokenize(text) if token)


def extract_yap_lemmas(text: str) -> str:
    results = YAP_API.run(text)
    return " ".join(results.dep_tree["lemma"])


def extract_stanza_lemmas(sentences: List[Any]) -> str:
    doc = Document(sentences)
    return " ".join(word.lemma for word in doc.iter_words() if word.lemma)


def extract_stanza_pos(sentences: List[Any]) -> str:
    doc = Document(sentences)
    return " ".join(f"{word.upos}_{word.xpos}" for word in doc.iter_words())
