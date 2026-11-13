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
#   carry a provenance sidecar built offline by this module:
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

import argparse
import ast
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union, get_args

import numpy as np
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


def _reader_self_check() -> None:
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


# Offline safenlp_2024 sidecar reconstruction. The online API above only reads the sidecar.

BENCHMARK = "safenlp_2024"
TASKS = ("medical", "ruarobot")
FAMILIES = ("character", "word", "vicuna")
SPLITS = ("train_pos", "train_neg", "test_pos", "test_neg")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
COSINE_KEEP = 0.6  # ANTONIO hyperrectangles.py keeps a perturbation whose cosine to its sentence exceeds this
ALIGN_RATIO = 2.0  # an aligned box must lie this many times closer to its sentence than to any other sentence
ALIGN_SLACK = 1e-3
PERTURBATIONS = {
    "character": ("char_swapping", "char_replacement", "char_deletion", "char_insertion", "char_repetition"),
    "word": ("word_deletion", "word_repetition", "word_negation", "word_ordering", "word_singular_plural_verb",
             "word_verb_tense"),
}
_HELPERS = ("return_random_number", "return_adjacent_char", "swap_characters", "change_ordering", "is_third_person")
_RANDOM_NAMES = {"randint", "return_random_number"}
_NLTK = (("tokenizers/punkt_tab", "punkt_tab"),
         ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
         ("corpora/wordnet", "wordnet"))
_BOUND = re.compile(r"\(assert \((>=|<=) X(?:_(\d+)|\[(?:\d+\s*,\s*)*(\d+)\]) ([-+0-9.eE]+)\)\)")

TASK_INFO: Dict[str, Dict[str, Any]] = {
    "medical": {
        "title": "Medical safety",
        "description": "Is a user query a medical question? Class 0: queries that expert practitioners labelled "
                       "medical (any risk level); class 1: non-medical queries.",
        "class_names": ["medical query", "non-medical query"],
        "dataset": "Medical safety (Abercrombie and Rieser, AACL 2022)",
        "dataset_url": "https://aclanthology.org/2022.aacl-short.30/",
    },
    "ruarobot": {
        "title": "R-U-A-Robot",
        "description": "Does the user ask whether they are talking to a robot? Class 0: positive and ambiguous "
                       "queries, where the system should disclose that it is not human; class 1: other queries.",
        "class_names": ["asks if it is a robot", "other query"],
        "dataset": "R-U-A-Robot (Gros et al., ACL 2021)",
        "dataset_url": "https://aclanthology.org/2021.acl-long.544/",
    },
}
FAMILY_INFO: Dict[str, Dict[str, str]] = {
    "character": {"title": "character-level",
                  "description": "One character swapped, replaced by a keyboard neighbour, deleted, inserted or "
                                 "repeated."},
    "word": {"title": "word-level",
             "description": "One word deleted or repeated, the sentence negated, words reordered, or number or "
                            "tense changed."},
    "vicuna": {"title": "Vicuna paraphrase",
               "description": "Five paraphrases written by the Vicuna-13B language model; their texts were never "
                              "published."},
}


def _load_functions(path: Path, names: Sequence[str], namespace: Dict[str, Any]) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body: List[ast.stmt] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in body if isinstance(n, ast.FunctionDef)}
    if missing:
        raise ValueError(f"{path} does not define {sorted(missing)}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _random_drawers(path: Path, names: Sequence[str]) -> Set[str]:
    """Functions among `names` that draw from Python's random module, directly or through each other."""
    calls = {n.name: {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
             for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    drawers = {name for name, used in calls.items() if used & _RANDOM_NAMES}
    while True:
        more = {name for name, used in calls.items() if used & drawers} - drawers
        if not more:
            return drawers
        drawers |= more


class _Encoder:
    def __init__(self, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.model = AutoModel.from_pretrained(ENCODER).to(device).eval()
        self.device = device
        self.revision: Optional[str] = getattr(self.model.config, "_commit_hash", None)

    def __call__(self, texts: Sequence[str], batch: int = 256) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer(list(texts[i:i + batch]), padding=True, truncation=True, max_length=256,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(mean, p=2, dim=1).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, int(self.model.config.hidden_size)))


def box_distance(points: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    gap = np.maximum(np.maximum(lb - points, points - ub), 0.0)
    return np.sqrt((gap ** 2).sum(-1))


def fit_projection(E: np.ndarray, lb: np.ndarray, ub: np.ndarray, W0: np.ndarray, b0: np.ndarray,
                   lam: float = 1e-3, iters: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Projection E @ W + b that puts each sentence into its box in the least-squares sense, shrunk to (W0, b0)."""
    X = np.hstack([E, np.ones((len(E), 1))])
    reg = lam * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    inverse = np.linalg.inv(X.T @ X + reg)
    theta0 = np.vstack([W0, b0[None, :]])
    theta = theta0.copy()
    for _ in range(iters):
        theta = inverse @ (X.T @ np.clip(X @ theta, lb, ub) + reg @ theta0)
    return theta[:-1], theta[-1]


def monotone_alignment(points: np.ndarray, lb: np.ndarray,
                       ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match box j to sentence match[j] (strictly increasing), minimising the summed point-to-box distance.

    Also returns each match's distance and the distance of the box's nearest other sentence.
    """
    n, m = len(points), len(lb)
    if m > n:
        raise ValueError(f"{m} boxes cannot come from {n} sentences")
    skips = n - m
    best = np.empty((m, skips + 1))
    back = np.zeros((m, skips + 1), dtype=np.int64)
    for j in range(m):
        cost = box_distance(points[j:j + skips + 1], lb[j], ub[j])
        if j == 0:
            best[0] = cost
            continue
        prev = best[j - 1]
        arg = np.zeros(skips + 1, dtype=np.int64)
        running = 0
        for k in range(1, skips + 1):
            if prev[k] < prev[running]:
                running = k
            arg[k] = running
        best[j] = cost + prev[arg]
        back[j] = arg
    k = int(np.argmin(best[-1]))
    match = np.empty(m, dtype=np.int64)
    for j in range(m - 1, -1, -1):
        match[j] = j + k
        k = int(back[j][k])
    distance = box_distance(points[match], lb, ub)
    alternative = np.full(m, np.inf)
    for j in range(m):
        others = box_distance(points, lb[j], ub[j])
        others[match[j]] = np.inf
        alternative[j] = float(others.min())
    return match, distance, alternative


def read_box(path: Path) -> np.ndarray:
    """The input box of a VNN-LIB 1.0 (X_i) or 2.0 (X[0,i]) property file as an (n, 2) array."""
    lo: Dict[int, float] = {}
    hi: Dict[int, float] = {}
    for op, flat, last, value in _BOUND.findall(path.read_text(encoding="utf-8")):
        (lo if op == ">=" else hi)[int(flat or last)] = float(value)
    if not lo or set(lo) != set(hi) or set(lo) != set(range(len(lo))):
        raise ValueError(f"{path} does not bound every input variable from both sides")
    return np.array([[lo[d], hi[d]] for d in range(len(lo))])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Unavailable:
    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, attr: str) -> Any:
        raise RuntimeError(f"{self.name} is not installed")


def _nltk(data_dir: Path) -> Any:
    import nltk

    data_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(data_dir))
    for resource, package in _NLTK:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=str(data_dir), quiet=True)
    return nltk


def replay(family: str, sentences: Sequence[str], antonio_src: Path,
           nltk_module: Any) -> Tuple[List[Tuple[int, str, Optional[str]]], Optional[int]]:
    """ANTONIO's create_perturbations on the positive training sentences: (sentence, function, text or None).

    Also returns the first sentence whose replay cannot be trusted: a failing function that draws random numbers
    would shift every later draw.
    """
    from nltk.stem.wordnet import WordNetLemmatizer

    try:
        import mlconjug3  # pyright: ignore[reportMissingImports]
    except ImportError:
        mlconjug3 = _Unavailable("mlconjug3")
    names = PERTURBATIONS[family]
    source = antonio_src / "perturbations.py"
    namespace = _load_functions(source, (*names, *_HELPERS),
                                {"randint": random.randint, "seed": random.seed, "nltk": nltk_module, "np": np,
                                 "re": re, "WordNetLemmatizer": WordNetLemmatizer, "mlconjug3": mlconjug3})
    drawers = _random_drawers(source, (*names, *_HELPERS))
    random.seed(42)
    rows: List[Tuple[int, str, Optional[str]]] = []
    desync: Optional[int] = None
    for i, sentence in enumerate(sentences):
        for name in names:
            try:
                rows.append((i, name, str(namespace[name]([sentence])[0])))
            except Exception as exc:
                logger.debug("%s failed on sentence %d: %s", name, i, exc)
                rows.append((i, name, None))
                if name in drawers and desync is None:
                    desync = i
    return rows, desync


def build_task(task: str, safenlp: Path, antonio: Path, benchmark: Path, encoder: _Encoder,
               nltk_module: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    import pandas as pd
    from sklearn.decomposition import PCA

    load_data = _load_functions(antonio / "src" / "data.py", ("load_data",), {"pd": pd, "np": np})["load_data"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arrays = load_data(task, path=str(antonio / "datasets"))
    texts = {s: [str(t) for t in arrays[k]] for k, s in enumerate(SPLITS)}
    labels = {s: [int(v) for v in arrays[4 + k]] for k, s in enumerate(SPLITS)}
    emb = {s: encoder(texts[s]) for s in SPLITS}
    pca = PCA(n_components=30, svd_solver="full").fit(np.vstack([emb[s] for s in SPLITS]))
    boxes = {f: np.load(safenlp / "data" / task / f"{f}.npy") for f in FAMILIES}
    pos = emb["train_pos"]
    n = len(pos)
    word = boxes["word"]
    if len(word) != n:
        raise ValueError(f"{task}: {len(word)} word boxes for {n} sentences; the positional match needs one each")
    raw = pca.transform(pos)
    centres = word.mean(axis=2)
    signs = np.sign([np.corrcoef(raw[:, d], centres[:, d])[0, 1] for d in range(raw.shape[1])])
    W, b = fit_projection(pos, word[:, :, 0], word[:, :, 1], pca.components_.T * signs,
                          -(pca.mean_ @ pca.components_.T) * signs)
    anchors = {s: emb[s] @ W + b for s in SPLITS}
    points = anchors["train_pos"]

    variants: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    failures: Dict[str, Dict[int, List[str]]] = {}
    variant_points: Dict[str, Dict[int, List[np.ndarray]]] = {}
    desynced: Dict[str, Optional[int]] = {}
    for family in PERTURBATIONS:
        rows, desynced[family] = replay(family, texts["train_pos"], antonio / "src", nltk_module)
        done = [k for k, row in enumerate(rows) if row[2] is not None]
        vecs = encoder([str(rows[k][2]) for k in done])
        variants[family], failures[family], variant_points[family] = {}, {}, {}
        for k, (i, name, text) in enumerate(rows):
            if text is None:
                failures[family].setdefault(i, []).append(name)
        for vec, k in zip(vecs, done):
            i, name, text = rows[k]
            cosine = float(vec @ pos[i])
            kept = cosine > COSINE_KEEP
            variants[family].setdefault(i, []).append(
                {"text": text, "kind": name, "kept": kept, "cosine": round(cosine, 4)})
            if kept:
                variant_points[family].setdefault(i, []).append(vec @ W + b)

    offsets = dict(zip(FAMILIES, np.cumsum([0] + [len(boxes[f]) for f in FAMILIES[:-1]]).tolist()))
    entries: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"sentences": {s: len(texts[s]) for s in SPLITS},
                             "signs": "".join("+" if s > 0 else "-" for s in signs)}
    for family in FAMILIES:
        bx = boxes[family]
        lb, ub = bx[:, :, 0], bx[:, :, 1]
        aligned, distance, alternative = monotone_alignment(points, lb, ub)
        if family == "word":
            match = np.arange(n)
            how = f"the generator kept every positive training sentence ({n} boxes for {n} sentences)"
            verified = np.ones(len(bx), dtype=bool)
        elif family == "character":
            with_box = np.array([any(v["kept"] for v in variants[family].get(i, ())) for i in range(n)])
            if int(with_box.sum()) == len(bx) and desynced[family] is None:
                match = np.flatnonzero(with_box)
                how = (f"the replayed generator gives boxes to the same {len(bx)} of {n} sentences as published "
                       f"(alignment agrees on {np.mean(match == aligned):.1%})")
                verified = np.ones(len(bx), dtype=bool)
            else:
                match, verified = aligned, np.zeros(len(bx), dtype=bool)
                how = "order-preserving alignment"
        else:
            match, verified = aligned, np.zeros(len(bx), dtype=bool)
            how = "order-preserving alignment"
        dist = box_distance(points[match], lb, ub)
        clear = alternative >= ALIGN_RATIO * dist + ALIGN_SLACK
        cut = desynced.get(family)
        counts: Dict[str, Any] = {"boxes": len(bx), "verified": 0, "aligned": 0, "unmatched": 0, "files": 0}
        hull_errors: List[float] = []
        for j in range(len(bx)):
            i = int(match[j])
            g = offsets[family] + j
            files = {}
            for rel in (f"vnnlib/{task}/hyperrectangle_{g}.vnnlib", f"vnnlib/{task}/prop_{g}_perturbations.vnnlib"):
                path = benchmark / rel
                if path.is_file():
                    if float(np.abs(read_box(path) - bx[j]).max()) > 1e-12:
                        raise ValueError(f"{path} does not hold {family} box {j} of safeNLP's {task} data")
                    files[rel] = _sha256(path)
            counts["files"] += len(files)
            if verified[j]:
                status = "verified"
                detail = f"{how}; the reconstructed sentence point is {dist[j]:.1e} from the box"
                if family in variant_points:
                    hull = np.vstack([points[i], *variant_points[family].get(i, [])])
                    err = float(max(np.abs(hull.min(0) - lb[j]).max(), np.abs(hull.max(0) - ub[j]).max()))
                    hull_errors.append(err)
                    detail += f"; the replayed perturbations span the box up to {err:.1e}"
            elif clear[j]:
                status = "aligned"
                detail = (f"{how}: the sentence point is {dist[j]:.1e} from the box, every other sentence at "
                          f"least {alternative[j]:.1e}")
            else:
                status = "unmatched"
                detail = (f"no clear sentence: the aligned one is {dist[j]:.1e} from the box and another sentence "
                          f"{alternative[j]:.1e}")
            counts[status] += 1
            if family == "vicuna":
                vs, vlist = "not_published", []
            elif status == "unmatched":
                vs, vlist = "not_reproduced", []
            elif cut is not None and i >= cut:
                vs, vlist = "not_reproduced", []
            else:
                vlist = variants[family].get(i, [])
                missing = failures[family].get(i, [])
                vs = "partial" if missing else "replayed"
                if missing:
                    detail += f"; not replayed: {', '.join(missing)} (mlconjug3 3.8.3 is unavailable)"
            entries.append({"files": files, "task": task, "family": family, "family_index": j,
                            "example": f"{task}:train_pos:{i}" if status != "unmatched" else None,
                            "variants": vlist, "variants_status": vs,
                            "match": {"status": status, "detail": detail}})
        counts["median_distance"] = float(np.median(dist))
        if hull_errors:
            counts["median_hull_error"] = float(np.median(hull_errors))
        stats[family] = counts
        logger.info("%s %s: %s", task, family, counts)

    examples = [{"id": f"{task}:{s}:{k}", "task": task, "text": texts[s][k], "label": labels[s][k], "split": s,
                 "anchor": [round(float(x), 6) for x in anchors[s][k]]}
                for s in SPLITS for k in range(len(texts[s]))]
    return examples, entries, stats


def _source_md(manifest: Dict[str, Any], command: str) -> str:
    lines = [f"# {BENCHMARK} provenance sidecar", "",
             "Which sentence each property of the VNN-COMP benchmark safenlp_2024 was generated from, rebuilt and read by "
             "`act.pipeline.gui.provenance_gui`.", "",
             "## Sources", ""]
    lines += [f"- {s['name']}: {s.get('url', '')} {('commit `' + s['commit'] + '`') if s.get('commit') else ''}"
              f"{(' revision `' + s['revision'] + '`') if s.get('revision') else ''}"
              f"{(' (' + s['license'] + ')') if s.get('license') else ''}" for s in manifest["sources"]]
    built = manifest["built"]
    lines += ["", "## Build", "", f"- Command: `{command}`", f"- Date: {built['at']}",
              "- Environment: " + ", ".join(f"{k} {v}" for k, v in built["versions"].items()), "",
              "## How boxes are matched", "",
              "- `verified`: the generator's own bookkeeping fixes the sentence. Word boxes: one per positive "
              "training sentence, in order. Character boxes: ANTONIO's perturbation functions, replayed from "
              "random.seed(42), give boxes to exactly the published number of sentences, so the k-th box is the "
              "k-th sentence that kept a perturbation.",
              "- `aligned`: order-preserving alignment of the reconstructed sentence points to the boxes, kept "
              f"only when every other sentence is at least {ALIGN_RATIO:g} times farther from the box (plus "
              f"{ALIGN_SLACK:g}). Vicuna paraphrases were never published, so that family can only be aligned.",
              "- `unmatched`: no clear sentence; no text is shown.",
              "- Sentence points (anchors) are reconstructions: all-MiniLM-L6-v2 embeddings, projected with a PCA "
              "basis whose signs and calibration are fitted to the word boxes. They are approximate; distances to "
              "the boxes are recorded per property.",
              "- Variants are the replayed perturbed texts with their cosine to the source sentence (kept above "
              f"{COSINE_KEEP:g}). `partial` marks sentences where a perturbation could not be replayed.", "",
              "## Counts", "",
              "| task | family | boxes | verified | aligned | unmatched | local files |", "|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for family in FAMILIES:
            c = manifest["stats"][task][family]
            lines.append(f"| {task} | {family} | {c['boxes']} | {c['verified']} | {c['aligned']} | "
                         f"{c['unmatched']} | {c['files']} |")
    lines += ["", "## Limitations", "",
              "- Only local instance files whose box equals the published box are covered (sha256 recorded); a "
              "changed file is not described.",
              "- word_verb_tense needs mlconjug3 3.8.3, which does not build on Python 3.12; sentences where it "
              "would have run are marked `partial`.",
              ""]
    return "\n".join(lines)


def build(safenlp: Path, antonio: Path, benchmark: Path, out: Path, device: str, nltk_dir: Path,
          sources: Dict[str, str]) -> Dict[str, Any]:
    import sklearn
    import torch
    import transformers

    start = time.time()
    encoder = _Encoder(device)
    nltk_module = _nltk(nltk_dir)
    examples: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    for task in TASKS:
        ex, en, st = build_task(task, safenlp, antonio, benchmark, encoder, nltk_module)
        examples += ex
        entries += en
        stats[task] = st
    manifest = {
        "schema": SCHEMA, "benchmark": BENCHMARK, "modality": "text",
        "representation": "all-MiniLM-L6-v2 sentence embedding (384-d, mean-pooled, L2-normalised) projected to "
                          "the benchmark's 30 PCA coordinates",
        "tasks": TASK_INFO, "families": FAMILY_INFO,
        "sources": [
            {"name": "safeNLP boxes", "url": "https://github.com/ANTONIONLP/safeNLP", "commit": sources["safenlp"]},
            {"name": "ANTONIO data and generator", "url": "https://github.com/ANTONIONLP/ANTONIO",
             "commit": sources["antonio"], "license": "Apache-2.0"},
            {"name": "all-MiniLM-L6-v2", "url": f"https://huggingface.co/{ENCODER}",
             "revision": encoder.revision or "unknown"},
            {"name": "benchmark files", "url": sources["benchmark_url"], "commit": sources["benchmark"]},
        ],
        "stats": stats,
        "built": {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - start, 1),
                  "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                               "transformers": transformers.__version__, "scikit-learn": sklearn.__version__,
                               "nltk": nltk_module.__version__, "numpy": np.__version__}},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, rows in (("examples.jsonl", examples), ("inputs.jsonl", entries)):
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "SOURCE.md").write_text(_source_md(manifest, " ".join(["python -m act.pipeline.gui.provenance_gui",
                                                                   *sys.argv[1:]])), encoding="utf-8")
    return manifest


def _builder_self_check() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    points = rng.normal(size=(40, 6))
    keep = np.sort(rng.choice(40, size=31, replace=False))
    lb = points[keep] - rng.uniform(0.01, 0.05, size=(31, 6))
    ub = points[keep] + rng.uniform(0.01, 0.05, size=(31, 6))
    match, dist, alt = monotone_alignment(points, lb, ub)
    assert match.tolist() == keep.tolist() and float(dist.max()) == 0.0 and bool((alt > 0).all()), (match, keep)

    E = rng.normal(size=(200, 12))
    W_true, b_true = rng.normal(size=(12, 4)), rng.normal(size=4)
    true = E @ W_true + b_true
    W, b = fit_projection(E, true - 0.01, true + 0.01, W_true + 0.05 * rng.normal(size=(12, 4)), b_true)
    assert float(box_distance(E @ W + b, true - 0.01, true + 0.01).max()) < 1e-3

    def draws(src: str) -> Set[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.py"
            path.write_text(src)
            return _random_drawers(path, ("a", "b", "c"))

    assert draws("def a():\n    return randint(0, 1)\ndef b():\n    return a()\ndef c():\n    return 1\n") == {"a", "b"}

    with tempfile.TemporaryDirectory() as tmp:
        one, two = Path(tmp) / "one.vnnlib", Path(tmp) / "two.vnnlib"
        one.write_text("(declare-const X_0 Real)\n(assert (>= X_0 -0.5))\n(assert (<= X_0 1e-3))\n"
                       "(assert (>= X_1 0.25))\n(assert (<= X_1 0.75))\n")
        two.write_text("(declare-input X float32 [1, 2])\n(assert (>= X[0,0] -0.5))\n(assert (<= X[0,0] 1e-3))\n"
                       "(assert (>= X[0,1] 0.25))\n(assert (<= X[0,1] 0.75))\n")
        assert read_box(one).tolist() == read_box(two).tolist() == [[-0.5, 1e-3], [0.25, 0.75]]
        (Path(tmp) / "bad.vnnlib").write_text("(assert (>= X_0 0.0))\n")
        try:
            read_box(Path(tmp) / "bad.vnnlib")
        except ValueError:
            pass
        else:
            raise AssertionError("a one-sided bound must be rejected")
    print("safenlp builder self-check passed")


def _self_check() -> None:
    _reader_self_check()
    _builder_self_check()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the provenance sidecar of safenlp_2024.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--safenlp", type=Path, help="safeNLP checkout (data/<task>/<family>.npy)")
    parser.add_argument("--antonio", type=Path, help="ANTONIO checkout (datasets/, src/)")
    parser.add_argument("--benchmark", type=Path, help="local safenlp_2024 benchmark directory")
    parser.add_argument("--out", type=Path, help="output directory, normally <benchmark>/provenance")
    parser.add_argument("--nltk-data", type=Path, help="NLTK data directory (missing resources are downloaded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--safenlp-commit", default="unknown")
    parser.add_argument("--antonio-commit", default="unknown")
    parser.add_argument("--benchmark-source", default="unknown", help="where the benchmark files came from")
    parser.add_argument("--benchmark-url", default="https://github.com/VNN-COMP/vnncomp2026_benchmarks")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if None in (args.safenlp, args.antonio, args.benchmark, args.out, args.nltk_data):
        parser.error("--safenlp, --antonio, --benchmark, --out and --nltk-data are required")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = build(args.safenlp, args.antonio, args.benchmark, args.out, args.device, args.nltk_data,
                     {"safenlp": args.safenlp_commit, "antonio": args.antonio_commit,
                      "benchmark": args.benchmark_source, "benchmark_url": args.benchmark_url})
    print(json.dumps(manifest["stats"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["--self-check"]))
