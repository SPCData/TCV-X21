import pytest
import numpy as np
from tcvx21.dict_methods_m import (
    summarise_tree_dict,
    recursive_rename_keys,
    recursive_assert_dictionaries_equal,
)


@pytest.fixture
def simple_dict():
    return dict(
        entry1=dict(value1=1.0, value2=2.0, array=np.array([1, 2, 3])),
        entry2=dict(value1=1.0, value2=2.0, array=np.array([1, 2, 3])),
    )


def test_dict_summarise(simple_dict):
    """Tests whether the printing works"""
    summarise_tree_dict(simple_dict)


def test_dict_rename(simple_dict):

    rename = dict(entry1="new_entry", value2="new_value")

    renamed_dict = recursive_rename_keys(simple_dict, rename)

    assert not "entry1" in renamed_dict.keys()
    assert set(renamed_dict.keys()) == set(["new_entry", "entry2"])

    for subdict in renamed_dict.values():
        assert "new_value" in subdict.keys()
        assert "value2" not in subdict.keys()


def test_assert_dict_equals(simple_dict):

    recursive_assert_dictionaries_equal(simple_dict, simple_dict.copy())

    with pytest.raises(AssertionError):
        new_dict = simple_dict.copy()
        new_dict["extra_key"] = "other"
        recursive_assert_dictionaries_equal(simple_dict, new_dict)

    with pytest.raises(AttributeError):
        new_dict = simple_dict.copy()
        new_dict["entry1"] = "other"
        recursive_assert_dictionaries_equal(simple_dict, new_dict)

    with pytest.raises(AssertionError):
        new_dict = simple_dict.copy()
        new_dict["entry1"] = dict(value1=2.0, value2=2.0, array=np.array([1, 2, 3]))
        recursive_assert_dictionaries_equal(simple_dict, new_dict)

    with pytest.raises(AssertionError):
        new_dict = simple_dict.copy()
        new_dict["entry1"] = dict(value2=2.0, array=np.array([1, 2, 3]))
        recursive_assert_dictionaries_equal(simple_dict, new_dict)

    with pytest.raises(ValueError):
        new_dict = simple_dict.copy()
        new_dict["entry1"] = dict(value1=1.0, value2=2.0, array=np.array([1, 3]))
        recursive_assert_dictionaries_equal(simple_dict, new_dict)

    with pytest.raises(AssertionError):
        new_dict = simple_dict.copy()
        new_dict["entry1"] = dict(value1=1.0, value2=2.0, array=np.array([1, 3, 3]))
        recursive_assert_dictionaries_equal(simple_dict, new_dict)
