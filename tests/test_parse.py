from __future__ import annotations

import json

import pytest

from tap.domain.models import DocFormat
from tap.errors import ValidationFailed
from tap.parse import assemble, count_words, detect_format, numbers_and_units, parse, split_sentences
from tap.parse.tokens import normalise_number, protected_tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Dr. Smith gave 25 mg, e.g. in 2.5 ml. Do not exceed 1,000 mg!",
            ["Dr. Smith gave 25 mg, e.g. in 2.5 ml.", "Do not exceed 1,000 mg!"],
        ),
        ("Is that clear? Yes. Good.", ["Is that clear?", "Yes.", "Good."]),
        ("See Fig. 3 for details. Next sentence.", ["See Fig. 3 for details.", "Next sentence."]),
        ("No boundary here", ["No boundary here"]),
        ("", []),
        ("これは文です。これも文です。", ["これは文です。", "これも文です。"]),
    ],
)
def test_sentence_segmentation(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("1,000.5", "1000.5"),
        ("1.000,5", "1000.5"),
        ("25,5", "25.5"),
        ("1 000,5", "1000.5"),
        ("1,000", "1000"),
        ("0.500", "0.5"),
        ("2.5", "2.5"),
        ("0,5", "0.5"),
    ],
)
def test_number_normalisation_is_locale_independent(raw, norm):
    assert normalise_number(raw) == norm


def test_numbers_and_units_ignore_translated_words_but_keep_symbols():
    en, de = (
        numbers_and_units("within 24 hours, 25 mg, 5 %, 20 °C"),
        numbers_and_units("innerhalb von 24 Stunden, 25-mg-Tablette, 5 %, 20 °C"),
    )
    assert en == de


def test_protected_tokens():
    toks = protected_tokens("Hi {name}, click <b>here</b> %s ${x} https://a.b/c me@x.io {{m}}")
    assert set(toks) == {"{name}", "<b>", "</b>", "%s", "${x}", "https://a.b/c", "me@x.io", "{{m}}"}


def test_word_counting_cjk_and_placeholders():
    assert count_words("Take {name} twice daily") == 3
    assert count_words("これは文です") == 6


def test_txt_roundtrip_preserves_whitespace_and_paragraphs():
    text = "First sentence. Second one!\n\nSecond paragraph here.\n"
    doc = parse(DocFormat.TXT, text)
    assert [s.source for s in doc.segments] == ["First sentence.", "Second one!", "Second paragraph here."]
    assert assemble(DocFormat.TXT, doc.skeleton, {s.id: s.source for s in doc.segments}) == text


def test_json_resources_roundtrip_nested():
    src = json.dumps({"a": {"b": "Hello {name}"}, "c": ["x", "y"], "n": 5, "empty": ""})
    doc = parse(DocFormat.JSON, src)
    assert [s.source for s in doc.segments] == ["Hello {name}", "x", "y"]
    out = json.loads(assemble(DocFormat.JSON, doc.skeleton, {s.id: s.source.upper() for s in doc.segments}))
    assert out == {"a": {"b": "HELLO {NAME}"}, "c": ["X", "Y"], "n": 5, "empty": ""}


XLF = """<?xml version="1.0"?><xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"><file source-language="en" datatype="plaintext" original="a"><body>
<trans-unit id="1"><source>Click <g id="1">Save</g> &amp; <x id="2"/>now</source></trans-unit>
<trans-unit id="2" translate="no"><source>DoNotTouch</source></trans-unit></body></file></xliff>"""


def test_xliff_inline_tags_are_kept_as_protected_markup_and_roundtrip():
    doc = parse(DocFormat.XLIFF, XLF)
    assert len(doc.segments) == 1 and '<g id="1">Save</g>' in doc.segments[0].source
    out = assemble(DocFormat.XLIFF, doc.skeleton, {"u1": 'Klicken Sie <g id="1">Speichern</g> &amp; <x id="2"/>jetzt'}, target_lang="de")
    assert 'target-language="de"' in out and "Speichern" in out and "DoNotTouch" in out
    assert parse(DocFormat.XLIFF, out).segments[0].source.startswith("Click")


def test_xliff_rejects_malformed_translation_markup():
    doc = parse(DocFormat.XLIFF, XLF)
    with pytest.raises(ValidationFailed):
        assemble(DocFormat.XLIFF, doc.skeleton, {"u1": "Klicken <g>"}, target_lang="de")


def test_xml_bombs_and_external_entities_are_rejected():
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><xliff version="1.2"><file><body>&b;</body></file></xliff>'
    xxe = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"><file><body><trans-unit id="1"><source>&e;</source></trans-unit></body></file></xliff>'
    for payload in (bomb, xxe):
        with pytest.raises(ValidationFailed):
            parse(DocFormat.XLIFF, payload)


@pytest.mark.parametrize(("fmt", "content"), [(DocFormat.TXT, "   \n\n "), (DocFormat.JSON, "{}"), (DocFormat.JSON, "{not json")])
def test_empty_or_invalid_documents_are_rejected(fmt, content):
    with pytest.raises(ValidationFailed):
        parse(fmt, content)


def test_format_detection():
    assert detect_format("a.xlf", "") == DocFormat.XLIFF
    assert detect_format("a.json", "") == DocFormat.JSON
    assert detect_format("a.txt", "plain") == DocFormat.TXT
    assert detect_format("noext", '<?xml version="1.0"?><xliff version="1.2">') == DocFormat.XLIFF
