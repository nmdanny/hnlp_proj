from hnlp_proj import __version__
from hnlp_proj.utils import flip_hebrew_text, parse_joined_elements


def test_version():
    assert __version__ == "0.1.0"


def test_parse_joined_elements():
    assert parse_joined_elements("דניאל קרבל, ג'ון סמיט וג'אק סמיט") == [
        "דניאל קרבל",
        "ג'ון סמיט",
        "ג'אק סמיט",
    ]
    assert parse_joined_elements("דניאל קרבל, ג'ון סמיט") == ["דניאל קרבל", "ג'ון סמיט"]
    assert parse_joined_elements("דניאל קרבל וג'ון סמיט") == ["דניאל קרבל", "ג'ון סמיט"]
    assert parse_joined_elements("דניאל קרבל") == ["דניאל קרבל"]


def test_flip_hebrew_text():
    assert flip_hebrew_text("שלום לך") == "ךל םולש"
