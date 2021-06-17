from hnlp_proj.delta import DeltaTransformer, combine_texts_by_author
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluation:
    """Evaluation of several models(sklearn pipelines) over different amounts of authors"""

    def __init__(
        self,
        data: pd.DataFrame,
        combine_by_authors: bool,
        tag: str = "",
        random_state: int = 42,
    ):
        """Initialize the evaluation with train and test sets"""
        self.data = data.set_index("author", drop=False)
        self.authors_counts = self.data.author.value_counts()
        self.combine_by_authors = combine_by_authors

        self.counts: List[int] = []
        self.pipelines: List[Tuple[Pipeline, bool]] = []
        self.pipeline_descriptions: List[Union[str, Dict[str, Any]]] = []
        self.results: List[Dict[str, Any]] = []
        self.random_state = random_state

        self.tag = tag

    def set_author_counts(self, counts: List[int]) -> "Evaluation":
        """Set the author counts to be used for this data-set"""
        self.counts = counts
        return self

    def add_pipeline(
        self,
        pipeline: Pipeline,
        description: Union[str, Dict[str, str]],
        combine_by_author: bool = True,
    ) -> "Evaluation":
        self.pipelines.append((pipeline, combine_by_author))
        self.pipeline_descriptions.append(description)

        return self

    def evaluate(self):

        self.test_time = datetime.now()

        test_num = 1
        for n_authors in tqdm(self.counts, desc="Number of authors"):
            for (pipeline, combine_by_author), description in tqdm(
                zip(self.pipelines, self.pipeline_descriptions),
                desc="Pipelines",
                total=len(self.pipelines),
            ):
                self.evaluate_case(
                    test_num, n_authors, pipeline, combine_by_author, description
                )
                test_num += 1

    def evaluate_case(
        self,
        test_num: int,
        n_authors: int,
        pipeline: Pipeline,
        combine_by_author: bool,
        description: Dict[str, Any],
    ):
        chosen_authors = self.authors_counts[:n_authors].index
        selected_data = self.data[self.data["author"].isin(chosen_authors)]
        selected_train, selected_test = train_test_split(
            selected_data,
            test_size=0.2,
            random_state=self.random_state,
            stratify=selected_data["author"],
        )

        if combine_by_author:
            selected_train = combine_texts_by_author(selected_train)

        fit_start = datetime.now()
        pipeline.fit(selected_train, selected_train.index)

        predict_start = datetime.now()

        y_pred = pipeline.predict(selected_test)

        predict_end = datetime.now()

        y_true = selected_test.index

        scores = self.get_scores(y_true, y_pred)

        result = {
            # metadata
            "test_num": test_num,
            "test_tag": self.tag,
            "test_time": self.test_time,
            # record test configuration
            "n_authors": n_authors,
            "authors": list(chosen_authors),
            "pipeline_desc": description,
            # record scores
            **scores,
            # record running times
            "fit_start": fit_start,
            "predict_start": predict_start,
            "predict_end": predict_end,
        }

        if isinstance(pipeline.steps[0][1], DeltaTransformer):
            # record DeltaTransformer features
            delta = pipeline.steps[0][1]
            freqs = delta.fit_transform(selected_train, selected_train.index)
            features = delta.get_params()["features"]
            counts = delta.last_transformed_count
            result["freqs"] = freqs
            result["features"] = features
            result["counts"] = counts

        self.results.append(result)

    def get_scores(self, y_true, y_pred) -> Dict[str, float]:
        assert len(y_true) == len(y_pred)

        scores = {}
        scores["accuracy"] = accuracy_score(y_true, y_pred)

        for weighting in ["micro", "macro"]:
            scores[f"precision_{weighting}"] = precision_score(
                y_true, y_pred, average=weighting, zero_division=0
            )
            scores[f"recall_{weighting}"] = recall_score(
                y_true, y_pred, average=weighting, zero_division=0
            )
            scores[f"f1_{weighting}"] = f1_score(
                y_true, y_pred, average=weighting, zero_division=0
            )

        return scores

    def get_result_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.results)


def plot_eval_results(df, metric: str = "accuracy"):
    ax = sns.scatterplot(data=df, x="n_authors", y=metric, hue="pipeline_desc")
    ax.set_title(f"Plot of {metric} as function of author count, per model")
    ax.legend(loc="upper right")
    plt.show()
