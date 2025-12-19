"""
This module contains unit tests for the `get_value` function in the `cfgan.config` module.
"""
import os
import unittest

from cfgan.config import get_value

PROJ_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


class TestGetValue(unittest.TestCase):
    """Test `get_value` function.
    """

    def setUp(self) -> None:
        pass

    def test_pathlike_type(self):
        result = get_value("data/test", "Union[str, os.PathLike]")
        expected = os.path.join(PROJ_ROOT_DIR, "data/test")
        self.assertEqual(result, expected)

    def test_dict_type_valid(self):
        result = get_value("{'key': 'value'}", "dict")
        self.assertEqual(result, {"key": "value"})

    def test_dict_type_invalid(self):
        result = get_value("invalid_dict", "dict")
        self.assertEqual(result, "invalid_dict")

    def test_bool_type_true(self):
        self.assertTrue(get_value("true", "bool"))
        self.assertTrue(get_value("True", "bool"))
        self.assertTrue(get_value("TRUE", "bool"))

    def test_bool_type_false(self):
        self.assertFalse(get_value("false", "bool"))
        self.assertFalse(get_value("False", "bool"))
        self.assertFalse(get_value("FALSE", "bool"))

    def test_int_conversion(self):
        self.assertEqual(get_value("123", "int"), 123)

    def test_float_conversion(self):
        self.assertEqual(get_value("123.45", "float"), 123.45)

    def test_string_conversion(self):
        self.assertEqual(get_value("test_string", "str"), "test_string")
