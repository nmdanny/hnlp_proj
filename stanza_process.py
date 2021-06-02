# %%

%load_ext autoreload
%autoreload 2

from YAP_Wrapper.yap_wrapper.hebtokenizer import num
from os import pipe
from hnlp_proj.loader import load_ben_yehuda, load_debug, load_eng_test, load_ynet  # noqa: F401, F403
from hnlp_proj.delta import create_feature_matrix, DeltaTransformer  # noqa: F401, F403
from hnlp_proj.processing import get_stanza_pipeline, process_data, Processing  # noqa: F401, F403
from hnlp_proj.utils import *  # noqa: F401, F403
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
# from sklearn.pipeline import Pipeline, make_pipeline  # noqa: F401
from tqdm import tqdm
import pickle
from pathlib import Path
from stanza_batch import batch
# %%

df = load_ynet()
STANZA_PICKLE_PATH = Path(__file__).parent / "data" / "ynet.pickle"

if (STANZA_PICKLE_PATH.exists()):
    raise ValueError(f"There is already a pickle file at {STANZA_PICKLE_PATH}, please rename it to proceed")
STANZA_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)

# %%
pipeline = get_stanza_pipeline(Processing.StanzaLemma, use_gpu=True)
# %%
%%time
numDocs = 0
docs = []
for doc in tqdm(batch(list(df["text"]), pipeline, batch_size=16, clear_cache=True)):
    print("")
    if numDocs % 100 == 0:
        print(f"Processed {numDocs} documents")
    numDocs += 1
    docs.append(doc)

with open(STANZA_PICKLE_PATH, 'wb') as pickle_f:
    pickle.dump(docs, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)