import numpy as np
import pandas as pd
from hnlp_proj.delta import combine_texts_by_author, create_feature_matrix


def test_combine_texts_by_author():
    df = pd.DataFrame.from_records(
        [
            {"author": "A", "text": ["hello", "world"], "count": 2},
            {"author": "A", "text": ["bye", "world"], "count": 2},
            {"author": "B", "text": ["שלום", "עולם"], "count": 3},
        ]
    )

    combined = combine_texts_by_author(df).reset_index()
    expected = pd.DataFrame.from_records(
        [
            {"author": "A", "text": ["hello", "world", "bye", "world"], "count": 4},
            {"author": "B", "text": ["שלום", "עולם"], "count": 3},
        ]
    )
    assert combined.equals(expected)


def test_create_feature_matrix():
    df = pd.DataFrame.from_records(
        [
            {"author": "A", "text": ["hello", "world"], "count": 2},
            {"author": "A", "text": ["bye", "world"], "count": 2},
            {"author": "B", "text": ["test"], "count": 1},
        ]
    )

    features = ["hello", "bye", "world", "test"]
    matrix, counts = create_feature_matrix(
        df, combine_by_author=True, features=features
    )

    expected_counts = pd.DataFrame.from_records(
        [
            {"author": "A", "hello": 1, "bye": 1, "world": 2, "test": 0},
            {"author": "B", "hello": 0, "bye": 0, "world": 0, "test": 1},
        ]
    ).set_index("author")

    assert counts.equals(expected_counts)
    assert matrix.shape == expected_counts.shape

    # No longer combine by author(treat each text as a separate sample,
    # even if they have same author)
    matrix2, counts2 = create_feature_matrix(
        df, combine_by_author=False, features=features
    )

    expected_counts2 = pd.DataFrame.from_records(
        [
            {"author": "A", "hello": 1, "bye": 0, "world": 1, "test": 0},
            {"author": "A", "hello": 0, "bye": 1, "world": 1, "test": 0},
            {"author": "B", "hello": 0, "bye": 0, "world": 0, "test": 1},
        ]
    ).set_index("author")

    assert counts2.equals(expected_counts2)
    assert matrix2.shape == expected_counts2.shape
