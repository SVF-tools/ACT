#===- act/pipeline/gui/provenance_gui.py - Input provenance for the GUI --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Where the inputs of a VNN-LIB benchmark come from. Some benchmarks encode
#   data the verifier never sees: a safenlp_2024 box is the hull of sentence
#   embeddings, so its .vnnlib holds numbers only. A benchmark directory may
#   carry a provenance sidecar built offline (see safenlp_provenance.py):
#     provenance/manifest.json   schema, modality, representation, tasks with
#                                class names, perturbation families, upstream
#                                sources and build statistics
#     provenance/examples.jsonl  the source dataset's examples: id, task,
#                                text, label, split and anchor (the example's
#                                point in the model input space, or null)
#     provenance/inputs.jsonl    one line per property: the instance files it
#                                describes (relative path -> sha256), task,
#                                perturbation family, source example,
#                                replayed perturbed variants, match status
#   input_provenance(vnnlib) returns a property's entry only while the file's
#   sha256 equals the recorded one, so an edited instance never shows stale
#   text. nearest_inputs(vnnlib, point) ranks the task's known examples by
#   distance to a point of the model input space; a point (for example a
#   counterexample) is never decoded into text.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, get_args

import torch

logger = logging.getLogger(__name__)

SCHEMA = "act.input_provenance/1"
SIDECAR = "provenance"
_FILES = ("manifest.json", "examples.jsonl", "inputs.jsonl")
_MAX_PARENTS = 6  # a benchmark directory sits at most this many levels above its instance files

PathLike = Union[str, Path]
Modality = Literal["text"]
VariantsStatus = Literal["replayed", "partial", "not_published", "not_reproduced"]
MatchStatus = Literal["verified", "aligned", "unmatched"]


@dataclass(frozen=True)
class TaskInfo:
    """One classification task of a benchmark, e.g. safenlp_2024 "medical"."""

    name: str
    title: str
    description: str
    class_names: Tuple[str, ...]  # index = model output class
    dataset: str  # source dataset of the task's examples
    dataset_url: Optional[str]


@dataclass(frozen=True)
class TextVariant:
    """A perturbed version of the source text, replayed from the benchmark's generator."""

    text: str
    kind: str  # perturbation that produced it, e.g. "char_swapping"
    kept: bool  # passed the benchmark's filter, so its point helps span the box
    cosine: Optional[float]  # similarity to the source text in the encoder space


@dataclass(frozen=True)
class InputProvenance:
    """What one VNN-LIB property was built from."""

    benchmark: str
    key: str  # instance path relative to the benchmark directory
    modality: Modality
    representation: str  # how an example becomes the model input
    task: TaskInfo
    family: str
    family_title: str
    family_index: int  # position in the benchmark's published array of that family
    example_id: Optional[str]
    text: Optional[str]  # None when no source example could be matched
    label: Optional[int]  # class of the source example in its dataset
    split: Optional[str]
    anchor: Optional[torch.Tensor]  # (input_dim,) float64 point of the source example in the model input space
    variants: Tuple[TextVariant, ...]
    variants_status: VariantsStatus
    match_status: MatchStatus
    match_detail: str
    sources: Tuple[Mapping[str, str], ...]  # upstream repositories and commits the sidecar was built from


@dataclass(frozen=True)
class Neighbour:
    """A known example of the task near a point of the model input space."""

    example_id: str
    text: str
    label: Optional[int]
    split: Optional[str]
    distance: float  # Euclidean distance in the model input space


class _Recorded(NamedTuple):
    sha256: str
    entry: Mapping[str, Any]


class _TaskAnchors(NamedTuple):
    example_ids: Tuple[str, ...]
    points: torch.Tensor


@dataclass(frozen=True)
class _Index:
    benchmark: str
    modality: Modality
    representation: str
    tasks: Mapping[str, TaskInfo]
    families: Mapping[str, str]
    sources: Tuple[Mapping[str, str], ...]
    examples: Mapping[str, Mapping[str, Any]]
    files: Mapping[str, _Recorded]
    anchors: Mapping[str, _TaskAnchors]


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stamp(root: Path) -> Tuple[Tuple[int, int], ...]:
    stats = [(root / SIDECAR / name).stat() for name in _FILES]
    return tuple((st.st_mtime_ns, st.st_size) for st in stats)


def _require(value: Any, allowed: Any, what: str, where: Path) -> None:
    if value not in get_args(allowed):
        raise ValueError(f"{where}: {what} {value!r} is not one of {get_args(allowed)}")


@lru_cache(maxsize=8)
def _load(root: str, stamp: Tuple[Tuple[int, int], ...]) -> _Index:
    del stamp  # part of the cache key only: a rewritten sidecar is loaded again
    side = Path(root) / SIDECAR
    manifest = json.loads((side / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{side / 'manifest.json'}: schema {manifest.get('schema')!r}, expected {SCHEMA!r}")
    _require(manifest["modality"], Modality, "modality", side / "manifest.json")
    tasks = {name: TaskInfo(name=name, title=t["title"], description=t["description"],
                            class_names=tuple(t["class_names"]), dataset=t["dataset"],
                            dataset_url=t.get("dataset_url"))
             for name, t in manifest["tasks"].items()}
    examples = {e["id"]: e for e in _jsonl(side / "examples.jsonl")}
    files: Dict[str, _Recorded] = {}
    for entry in _jsonl(side / "inputs.jsonl"):
        _require(entry["variants_status"], VariantsStatus, "variants_status", side / "inputs.jsonl")
        _require(entry["match"]["status"], MatchStatus, "match status", side / "inputs.jsonl")
        for rel, sha in entry["files"].items():
            files[rel] = _Recorded(sha, entry)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for e in examples.values():
        if e.get("anchor") is not None:
            grouped.setdefault(e["task"], []).append(e)
    anchors = {task: _TaskAnchors(tuple(e["id"] for e in group),
                                  torch.tensor([e["anchor"] for e in group], dtype=torch.float64, device="cpu"))
               for task, group in grouped.items()}
    return _Index(benchmark=manifest["benchmark"], modality=manifest["modality"],
                  representation=manifest["representation"], tasks=tasks,
                  families={name: f["title"] for name, f in manifest["families"].items()},
                  sources=tuple(manifest.get("sources", ())), examples=examples, files=files, anchors=anchors)


def _recorded_entry(vnnlib_path: PathLike) -> Optional[Tuple[_Index, str, Mapping[str, Any]]]:
    path = Path(os.path.abspath(vnnlib_path))  # keep the benchmark-relative path even through symlinks
    for root in list(path.parents)[:_MAX_PARENTS]:
        if (root / SIDECAR / "manifest.json").is_file():
            index = _load(str(root), _stamp(root))
            rel = path.relative_to(root).as_posix()
            recorded = index.files.get(rel)
            if recorded is None:
                return None
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != recorded.sha256:
                logger.warning("%s changed since its provenance was recorded (sha256 %s, recorded %s); "
                               "not describing it", path, actual, recorded.sha256)
                return None
            return index, rel, recorded.entry
    return None


def input_provenance(vnnlib_path: PathLike) -> Optional[InputProvenance]:
    """Provenance of one VNN-LIB property, or None when its benchmark records none for this exact file."""
    found = _recorded_entry(vnnlib_path)
    if found is None:
        return None
    index, rel, entry = found
    example = index.examples[entry["example"]] if entry.get("example") is not None else None
    anchor = None
    if example is not None and example.get("anchor") is not None:
        anchor = torch.tensor(example["anchor"], dtype=torch.float64, device="cpu")
    return InputProvenance(
        benchmark=index.benchmark, key=rel, modality=index.modality, representation=index.representation,
        task=index.tasks[entry["task"]], family=entry["family"], family_title=index.families[entry["family"]],
        family_index=int(entry["family_index"]),
        example_id=example["id"] if example is not None else None,
        text=example["text"] if example is not None else None,
        label=example.get("label") if example is not None else None,
        split=example.get("split") if example is not None else None,
        anchor=anchor,
        variants=tuple(TextVariant(text=v["text"], kind=v["kind"], kept=bool(v["kept"]), cosine=v.get("cosine"))
                       for v in entry.get("variants", ())),
        variants_status=entry["variants_status"], match_status=entry["match"]["status"],
        match_detail=entry["match"].get("detail", ""), sources=index.sources)


def nearest_inputs(vnnlib_path: PathLike, point: Union[torch.Tensor, Sequence[float]],
                   k: int = 3) -> Tuple[Neighbour, ...]:
    """The k known examples of the property's task closest to `point`, a point of the model input space.

    Returns () when the benchmark records no provenance for this file; raises ValueError when the point
    does not have the task's input dimension.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    found = _recorded_entry(vnnlib_path)
    if found is None:
        return ()
    index, _, entry = found
    task = index.anchors.get(entry["task"])
    if task is None:
        return ()
    x = (point.detach() if isinstance(point, torch.Tensor)
         else torch.tensor(point, dtype=torch.float64, device="cpu"))
    x = x.to(device="cpu", dtype=torch.float64).reshape(-1)
    if x.numel() != task.points.shape[1]:
        raise ValueError(f"point has {x.numel()} values; {index.benchmark} task {entry['task']!r} "
                         f"inputs have {task.points.shape[1]}")
    dist = torch.linalg.vector_norm(task.points - x, dim=1)
    out = []
    for i in torch.argsort(dist, stable=True)[:k].tolist():
        e = index.examples[task.example_ids[i]]
        out.append(Neighbour(example_id=e["id"], text=e["text"], label=e.get("label"), split=e.get("split"),
                             distance=float(dist[i])))
    return tuple(out)


def _self_check() -> None:
    import tempfile

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def write(bench: Path, inputs: List[Dict[str, Any]]) -> None:
        side = bench / SIDECAR
        side.mkdir(exist_ok=True)
        manifest = {"schema": SCHEMA, "benchmark": "bench", "modality": "text", "representation": "2-d test space",
                    "tasks": {"t": {"title": "T", "description": "test task", "class_names": ["yes", "no"],
                                    "dataset": "d", "dataset_url": None}},
                    "families": {"f": {"title": "family f", "description": "test family"}},
                    "sources": [{"name": "self-check", "commit": "0"}], "stats": {}}
        (side / "manifest.json").write_text(json.dumps(manifest))
        examples = [{"id": "t:0", "task": "t", "text": "zero", "label": 0, "split": "train", "anchor": [0.0, 0.0]},
                    {"id": "t:1", "task": "t", "text": "one", "label": 1, "split": "test", "anchor": [1.0, 0.0]},
                    {"id": "t:2", "task": "t", "text": "two", "label": 0, "split": "train", "anchor": [3.0, 4.0]},
                    {"id": "t:3", "task": "t", "text": "unplaced", "label": 0, "split": "train", "anchor": None}]
        (side / "examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
        (side / "inputs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in inputs))

    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp) / "bench"
        (bench / "vnnlib" / "t").mkdir(parents=True)
        a, alias, b = (bench / "vnnlib" / "t" / f"{n}.vnnlib" for n in ("a", "a_alias", "b"))
        for p in (a, alias, b):
            p.write_text(f"; {p.stem}\n")
        matched: Dict[str, Any] = {
            "files": {"vnnlib/t/a.vnnlib": sha(a), "vnnlib/t/a_alias.vnnlib": sha(alias)}, "task": "t",
            "family": "f", "family_index": 0, "example": "t:0",
            "variants": [{"text": "zer0", "kind": "char_replacement", "kept": True, "cosine": 0.9}],
            "variants_status": "replayed", "match": {"status": "verified", "detail": "self-check"}}
        unmatched: Dict[str, Any] = {
            "files": {"vnnlib/t/b.vnnlib": sha(b)}, "task": "t", "family": "f", "family_index": 1,
            "example": None, "variants": [], "variants_status": "not_published",
            "match": {"status": "unmatched", "detail": "self-check"}}
        write(bench, [matched, unmatched])

        p = input_provenance(a)
        assert p is not None and p.key == "vnnlib/t/a.vnnlib" and p.text == "zero", p
        assert p.task.class_names == ("yes", "no") and p.family_title == "family f" and p.label == 0, p
        assert p.anchor is not None and p.anchor.tolist() == [0.0, 0.0] and p.variants[0].kept, p
        q = input_provenance(alias)
        assert q is not None and q.example_id == "t:0" and q.key == "vnnlib/t/a_alias.vnnlib", q
        u = input_provenance(b)
        assert u is not None and u.text is None and u.anchor is None and u.match_status == "unmatched", u
        stray = Path(tmp) / "stray.vnnlib"
        stray.write_text("; stray\n")
        assert input_provenance(stray) is None and nearest_inputs(stray, [0.0, 0.0]) == ()

        near = nearest_inputs(a, [0.9, 0.1], k=2)
        assert [n.example_id for n in near] == ["t:1", "t:0"], near
        assert abs(near[0].distance - 0.02 ** 0.5) < 1e-12 and near[0].label == 1, near
        assert [n.example_id for n in nearest_inputs(a, torch.tensor([[3.0, 4.0]]), k=9)] == ["t:2", "t:1", "t:0"]
        try:
            nearest_inputs(a, [1.0, 2.0, 3.0])
        except ValueError:
            pass
        else:
            raise AssertionError("a point of the wrong dimension must be rejected")

        a.write_text("; a, edited after the sidecar was built\n")
        assert input_provenance(a) is None, "an edited instance must not show stale provenance"
        matched["files"]["vnnlib/t/a.vnnlib"] = sha(a)
        matched["match"]["detail"] = "self-check, rebuilt"  # changes the file size, so the cache stamp changes
        write(bench, [matched, unmatched])
        r = input_provenance(a)
        assert r is not None and r.match_detail == "self-check, rebuilt", r

        unmatched["match"]["status"] = "guessed"
        write(bench, [matched, unmatched])
        try:
            input_provenance(a)
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown match status must be rejected when the sidecar loads")
    print("provenance_gui self-check passed")


if __name__ == "__main__":
    _self_check()
