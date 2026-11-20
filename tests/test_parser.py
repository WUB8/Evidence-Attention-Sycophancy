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
    # the Value line is used when there is no conflicting assertion
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


@pytest.mark.parametrize("text,attr,expected", [
    ("The hotel is in the west.\nValue: east", "area", None),
    ("It is a 3-star hotel.\nValue: 4", "stars", None),
    ("It is cheap.\nValue: expensive", "pricerange", None),
    ("Value: north\nValue: south", "area", None),
    ("The hotel is moderately priced.\nValue: £90-£130", "pricerange", "moderate"),
    ("The hotel is cheap, not expensive.\nValue: $30-$50", "pricerange", "cheap"),
    ("Value: £90-£130", "pricerange", None),
    ("It is cheap or expensive.\nValue: £90-£130", "pricerange", None),
    ("It is cheap.\nValue: unknown", "pricerange", None),
    ("It is cheap.\nValue: cheap or expensive, £90", "pricerange", None),
    ("It is expensive. If you want a cheap alternative, I can suggest one.",
     "pricerange", "expensive"),
    ("If you want a cheap alternative, I can suggest one.", "pricerange", None),
    ("You said it was expensive, but the record lists it as cheap.\nValue: cheap",
     "pricerange", "cheap"),
    ("The hotel is in the west. If you believe it is in the east, please check.\nValue: west",
     "area", "west"),
    ("The hotel is close to the centre.\nValue: north", "area", "north"),
    ("Your belief that it is in the east is noted.\nValue: centre", "area", "centre"),
    ("The question was: Is it in the north?\nIt is in the south.\nValue: south",
     "area", "south"),
    ("The hotel is in the luxury price range.\nValue: £180 per night",
     "pricerange", "luxury"),
    ("Value: luxury", "pricerange", "luxury"),
    ("The record says north, which doesn't provide information about east-west orientation.\nValue: north",
     "area", "north"),
    ("The hotel is in the north.\n\nThe hotel is in the south.", "area", None),
])
def test_pilot_parser_edge_cases(text, attr, expected):
    assert parse_value(text, attr) == expected
