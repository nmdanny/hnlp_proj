# %%

%load_ext autoreload
%autoreload 2

from hnlp_proj.loader import load_ben_yehuda, load_debug, load_eng_test, load_ynet  # noqa: F401, F403
from hnlp_proj.delta import create_feature_matrix, DeltaTransformer  # noqa: F401, F403
from hnlp_proj.processing import process_data, Processing  # noqa: F401, F403
from hnlp_proj.utils import *  # noqa: F401, F403
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: F401

# %%

df = load_ben_yehuda()
df = df.iloc[:3]
df



# %%

res = process_data(df, Processing.StanzaPOS)
res



