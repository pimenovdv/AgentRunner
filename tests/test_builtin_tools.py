import pytest
import math
from datetime import datetime

from app.models.builtin_tools import (
    calculator,
    datetime_formatter,
    crypto_hasher,
    regex_parser,
)

def test_calculator():
    assert calculator.invoke({"expression": "2 + 2"}) == 4.0
    assert calculator.invoke({"expression": "10 / 2"}) == 5.0
    assert calculator.invoke({"expression": "3 * 4 + 2"}) == 14.0
    assert calculator.invoke({"expression": "abs(-5)"}) == 5.0
    assert calculator.invoke({"expression": "round(3.14159, 2)"}) == 3.14
    assert calculator.invoke({"expression": "math.sin(math.pi / 2)"}) == 1.0
    assert "Division by zero" in calculator.invoke({"expression": "1 / 0"})
    assert "Unsupported function call" in calculator.invoke({"expression": "os.system('ls')"})
    assert "Unsupported attribute" in calculator.invoke({"expression": "math.exit"})

def test_datetime_formatter():
    now_str = datetime_formatter.invoke({"date_str": "now", "format_str": "%Y-%m-%d"})
    assert now_str == datetime.now().strftime("%Y-%m-%d")

    specific_date = datetime_formatter.invoke({"date_str": "2023-10-27T15:30:00", "format_str": "%Y/%m/%d"})
    assert specific_date == "2023/10/27"

    assert "Error: Cannot parse datetime string" in datetime_formatter.invoke({"date_str": "invalid", "format_str": "%Y"})

def test_crypto_hasher():
    # sha256 of "hello"
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert crypto_hasher.invoke({"data": "hello", "algorithm": "sha256"}) == expected

    # default algorithm is sha256
    assert crypto_hasher.invoke({"data": "hello"}) == expected

    # unsupported algo
    assert "Unsupported algorithm" in crypto_hasher.invoke({"data": "hello", "algorithm": "unknown_algo"})

def test_regex_parser():
    # Simple extraction
    assert regex_parser.invoke({"pattern": r"\d+", "text": "hello 123 world 456"}) == ["123", "456"]

    # No match
    assert regex_parser.invoke({"pattern": r"\d+", "text": "no numbers here"}) == []

    # Groups
    assert regex_parser.invoke({"pattern": r"(\w+)-(\d+)", "text": "item-123", "group": 1}) == ["item"]
    assert regex_parser.invoke({"pattern": r"(\w+)-(\d+)", "text": "item-123", "group": 2}) == ["123"]

    # Invalid regex
    assert "Regex Error" in regex_parser.invoke({"pattern": "[unclosed", "text": "test"})
