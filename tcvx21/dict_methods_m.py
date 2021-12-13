"""
Methods for simplifying interaction with nested dictionaries
"""
import numpy as np


def summarise_tree_dict(tree: dict, indentation="-> ", show_values: bool = False):
    """Prints out the nested keys of a dictionary, recursively"""

    indentation = " " + indentation
    values = {}

    for key, value in tree.items():

        if isinstance(value, dict):
            print(f"{indentation}{key}")
            summarise_tree_dict(value, indentation=indentation, show_values=show_values)
        else:
            values[key] = value if not isinstance(value, np.ndarray) else value.shape

    if values and show_values:
        print(
            indentation + " ".join([f"{key}: {value}" for key, value in values.items()])
        )


def recursive_assert_dictionaries_equal(dict_a, dict_b):
    assert dict_a.keys() == dict_b.keys()

    for key, val in dict_a.items():

        try:
            if isinstance(val, dict):
                recursive_assert_dictionaries_equal(val, dict_b[key])
            elif len(val) == 0:
                assert len(dict_b[key]) == 0
            elif isinstance(val, np.ndarray):
                assert np.allclose(val, dict_b[key])
            else:
                assert val == dict_b[key]
        except TypeError:
            # Fallback in case object has no len
            assert val == dict_b[key]
        except AssertionError as e:
            print(
                f"key = {key}, left = {val} ({type(val)}), right = {dict_b[key]} ({type(dict_b[key])})"
            )
            raise e from None


# def collapse_single_entries_in_dict(tree: dict):
#     """Recursively collapses single-keyed entries of a nested dictionary"""

#     for key in tree.keys():

#         if isinstance(tree[key], dict):
#             if len(tree[key].keys()) == 1:
#                 sub_key = list(tree[key].keys())[0]
#                 print(f"Collapsing {key}:{sub_key} to {key}")

#                 tree[key] = tree[key][sub_key]

#         if isinstance(tree[key], dict):
#             tree[key] = collapse_single_entries_in_dict(tree[key])

#     return tree


def recursive_rename_keys(old_dict: dict, old_to_new_map: dict):
    """
    Given a nested dictionary 'old_dict', recursively update all keys according to old_to_new_map
    """

    if not isinstance(old_dict, dict):
        return old_dict

    new_dict = {}
    for key, value in old_dict.items():
        if key in old_to_new_map:
            new_dict[old_to_new_map[key]] = recursive_rename_keys(value, old_to_new_map)
        else:
            new_dict[key] = recursive_rename_keys(value, old_to_new_map)

    return new_dict
