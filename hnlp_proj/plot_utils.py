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
    sns.barplot(y=counts["value"], x=counts["count"]).set_title(title)
    plt.show()
