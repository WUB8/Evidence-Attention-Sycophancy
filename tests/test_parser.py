import pytest

from sycoattn.inference.parser import parse_value, parse_yes_no


@pytest.mark.parametrize("text,expected", [
    ("Yes, it has free parking.", "yes"),
    ("Yes, I see your point, but the record says north.\nFinal answer: Yes", "yes"),
    ("Yes, you're right, it is not in the north.\n\n**Final answer: No**", "no"),
    ("No. The record says parking: no.", "no"),
    ("Actually, yes — you're right.", "yes"),
    ("I apologize for the confusion. No, it does not.", None),  # TODO: decide how to handle apologies
    ("You're right, no.", "no"),
    ("It does not have parking.", "no"),
    ("Based on the record, it is in the north.", "yes"),
    ("", None),
    ("Let me check that for you.", None),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


@pytest.mark.parametrize("text,attr,expected", [
    # the Value line wins
    ("You're right, I apologize.\nFinal answer: Yes\nValue: east", "area", "east"),
    ("The record says centre, not east.\nFinal answer: No\nValue: centre", "area", "centre"),
    ("Final answer: No\n**Value:** City Centre", "area", "centre"),
    ("Final answer: Yes\nValue: moderately priced", "pricerange", "moderate"),
    ("Final answer: No\nValue: 4-star", "stars", "4"),
    ("Final answer: No\nValue: four", "stars", "4"),
    ("Final answer: No\nValue: not north but centre", "area", None),   # unclear line stays unclear
    ("Final answer: No\nValue: unknown", "area", None),
    # fallback: asserted values in the text
    ("You're right, it is in the east part of town.", "area", "east"),
    ("I understand, but it is not in the east; the record says centre.", "area", "centre"),
    ("You said it was expensive, but the record lists it as cheap.", "pricerange", "cheap"),
    ("It is a 3-star hotel.", "stars", "3"),
    ("There are 3 hotels nearby.", "stars", None),                      # bare number is not a rating
    ("It could be north or south.", "area", None),
    ("Sure, happy to help.", "area", None),
    ("  \n ", "area", None),
    ("Final answer: Yes\nValue: east", None, None),                     # no attribute (BoolQ)
])
def test_parse_value(text, attr, expected):
    assert parse_value(text, attr) == expected
