import pandas as pd
from hnlp_proj.utils import clean_texts, combine_texts
import seaborn as sns
import matplotlib.pyplot as plt


def load_ynet(show_html_len_plot=True) -> pd.DataFrame:

    texts = pd.read_json("./scrape/ynet.jl", lines=True)
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
