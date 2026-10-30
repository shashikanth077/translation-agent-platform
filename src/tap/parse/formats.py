"""File parsing and reassembly for txt, XLIFF 1.2 and flat-JSON string resources.

``parse`` returns segments plus an opaque skeleton; ``assemble`` rebuilds the original format from
translations. Round-trip fidelity (parse -> assemble with the source as target) is covered by tests.
XML is parsed with defusedxml (no entity expansion / external entities).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

import defusedxml.ElementTree as DET

from ..domain.models import DocFormat
from ..errors import ValidationFailed
from .segmentation import split_sentences

MAX_SEGMENTS = 50_000


@dataclass
class RawSegment:
    id: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    format: DocFormat
    segments: list[RawSegment]
    skeleton: dict[str, Any]


def detect_format(filename: str, content: str) -> DocFormat:
    name = filename.lower()
    if name.endswith((".xlf", ".xliff")):
        return DocFormat.XLIFF
    if name.endswith(".json"):
        return DocFormat.JSON
    stripped = content.lstrip()
    if stripped.startswith("<?xml") and "<xliff" in content[:2000]:
        return DocFormat.XLIFF
    return DocFormat.TXT


def parse(fmt: DocFormat, content: str, source_lang: str = "en") -> ParsedDocument:
    parser = {DocFormat.TXT: _parse_txt, DocFormat.JSON: _parse_json, DocFormat.XLIFF: _parse_xliff}[fmt]
    doc = parser(content, source_lang)
    if len(doc.segments) > MAX_SEGMENTS:
        raise ValidationFailed(f"document has {len(doc.segments)} segments; limit is {MAX_SEGMENTS}")
    if not doc.segments:
        raise ValidationFailed("document contains no translatable text")
    return doc


def assemble(fmt: DocFormat, skeleton: dict[str, Any], translations: dict[str, str], *, target_lang: str | None = None) -> str:
    fn = {DocFormat.TXT: _assemble_txt, DocFormat.JSON: _assemble_json, DocFormat.XLIFF: _assemble_xliff}[fmt]
    return fn(skeleton, translations, target_lang)


# -- txt ----------------------------------------------------------------------------------------
_PARA_SPLIT = re.compile(r"(\n\s*\n)")


def _parse_txt(content: str, lang: str) -> ParsedDocument:
    items: list[dict[str, str]] = []
    segments: list[RawSegment] = []
    for part in _PARA_SPLIT.split(content):
        if not part:
            continue
        if _PARA_SPLIT.fullmatch(part) or not part.strip():
            items.append({"raw": part})
            continue
        lead = part[: len(part) - len(part.lstrip())]
        if lead:
            items.append({"raw": lead})
        for sentence in split_sentences(part, lang):
            seg_id = f"s{len(segments) + 1}"
            segments.append(RawSegment(seg_id, sentence))
            items.append({"seg": seg_id})
            items.append({"raw": " "})
        if items and items[-1] == {"raw": " "}:
            items.pop()
        trail = part[len(part.rstrip()) :]
        if trail:
            items.append({"raw": trail})
    return ParsedDocument(DocFormat.TXT, segments, {"items": items})


def _assemble_txt(skel: dict[str, Any], tr: dict[str, str], _lang: str | None) -> str:
    return "".join(tr[i["seg"]] if "seg" in i else i["raw"] for i in skel["items"])


# -- json (flat or nested string resources) ------------------------------------------------------
def _walk_json(node: Any, path: tuple[str, ...], out: list[tuple[tuple[str, ...], str]]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_json(v, (*path, str(k)), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_json(v, (*path, str(i)), out)
    elif isinstance(node, str) and node.strip():
        out.append((path, node))


def _parse_json(content: str, _lang: str) -> ParsedDocument:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValidationFailed(f"invalid JSON resource file: {e}") from e
    found: list[tuple[tuple[str, ...], str]] = []
    _walk_json(data, (), found)
    segments = [RawSegment(f"k{i + 1}", text, {"path": list(path)}) for i, (path, text) in enumerate(found)]
    return ParsedDocument(DocFormat.JSON, segments, {"data": data, "paths": {s.id: s.meta["path"] for s in segments}})


def _assemble_json(skel: dict[str, Any], tr: dict[str, str], _lang: str | None) -> str:
    data = json.loads(json.dumps(skel["data"]))
    for seg_id, path in skel["paths"].items():
        node = data
        for key in path[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        last = path[-1]
        if isinstance(node, list):
            node[int(last)] = tr[seg_id]
        else:
            node[last] = tr[seg_id]
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


# -- xliff 1.2 -----------------------------------------------------------------------------------
_XLIFF_NS = "urn:oasis:names:tc:xliff:document:1.2"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _serialise(el: ET.Element) -> str:
    """Namespace-free serialisation of an inline element, so tags read ``<g id="1">`` in segments."""
    attrs = "".join(f' {_local(k)}="{_attr(v)}"' for k, v in el.attrib.items())
    inner = _escape(el.text or "") + "".join(_serialise(c) + _escape(c.tail or "") for c in el)
    tag = _local(el.tag)
    return f"<{tag}{attrs}>{inner}</{tag}>" if inner else f"<{tag}{attrs}/>"


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attr(s: str) -> str:
    return _escape(s).replace('"', "&quot;")


def _inner_xml(el: ET.Element) -> str:
    """Inner markup of an element as a string (inline tags kept verbatim as protected tokens)."""
    return _escape(el.text or "") + "".join(_serialise(c) + _escape(c.tail or "") for c in el)


def _parse_xliff(content: str, _lang: str) -> ParsedDocument:
    try:
        root = DET.fromstring(content.encode("utf-8"))
    except Exception as e:  # defusedxml raises several exception types for malformed/hostile input
        raise ValidationFailed(f"invalid XLIFF: {e}") from e
    if _local(root.tag) != "xliff":
        raise ValidationFailed("not an XLIFF document")
    segments: list[RawSegment] = []
    for unit in root.iter(f"{{{_XLIFF_NS}}}trans-unit"):
        src = unit.find(f"{{{_XLIFF_NS}}}source")
        if src is None or unit.get("translate") == "no":
            continue
        text = _inner_xml(src).strip()
        if text:
            segments.append(RawSegment(f"u{len(segments) + 1}", text, {"unit_id": unit.get("id")}))
    return ParsedDocument(DocFormat.XLIFF, segments, {"xml": content, "units": {s.id: s.meta["unit_id"] for s in segments}})


def _assemble_xliff(skel: dict[str, Any], tr: dict[str, str], target_lang: str | None) -> str:
    ET.register_namespace("", _XLIFF_NS)
    root = DET.fromstring(skel["xml"].encode("utf-8"))
    by_unit = {unit_id: tr[seg_id] for seg_id, unit_id in skel["units"].items() if seg_id in tr}
    for file_el in root.iter(f"{{{_XLIFF_NS}}}file"):
        if target_lang:
            file_el.set("target-language", target_lang)
    for unit in root.iter(f"{{{_XLIFF_NS}}}trans-unit"):
        text = by_unit.get(unit.get("id") or "")
        if text is None:
            continue
        target = unit.find(f"{{{_XLIFF_NS}}}target")
        if target is None:
            target = ET.SubElement(unit, f"{{{_XLIFF_NS}}}target")
        for child in list(target):
            target.remove(child)
        try:
            fragment = DET.fromstring(f'<t xmlns="{_XLIFF_NS}">{text}</t>'.encode())
        except Exception as e:
            raise ValidationFailed(f"translation for unit {unit.get('id')} is not well-formed XLIFF markup: {e}") from e
        target.text = fragment.text
        for child in fragment:
            target.append(child)
        target.set("state", "translated")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
