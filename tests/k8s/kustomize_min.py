# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# ==============================================================================
"""Minimal, dependency-free Kustomize renderer used by the manifest tests.

The suite must run on a machine with no cluster and no `kubectl`, because the
project's own CI image installs nothing but Python dependencies
(`.github/workflows/owl/ci.yaml`). This module therefore reproduces only the
Kustomize features this repository actually uses:

* ``resources`` (files and directories, attached with their own directory so
  relative generator paths keep working), ``components``
* ``namespace`` and ``commonLabels``
* ``images`` (``newName`` / ``newTag`` / ``digest``)
* ``configMapGenerator`` / ``secretGenerator``: ``literals``, ``envs`` and
  ``files`` (including the ``key=path`` rename form)
* ``patches``: file-based and inline JSON6902 (``add`` / ``replace`` /
  ``remove``, with ``~0`` / ``~1`` pointer escapes), ``$patch: delete``
* strategic-merge for merge-keyed lists (matched by ``name``)

Anything outside that subset raises :class:`UnsupportedKustomizeFeature`
instead of being silently skipped, so a manifest that stops rendering correctly
fails the test rather than passing on a partially applied overlay.

This is deliberately *not* a general Kustomize implementation. `kubectl
kustomize` remains the source of truth for deployment; the renderer exists so
that the contract tests can execute in CI.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "RenderedObject",
    "UnsupportedKustomizeFeature",
    "render",
    "render_with_kubectl",
    "repo_root",
]


class UnsupportedKustomizeFeature(RuntimeError):
    """A kustomization uses a feature this renderer does not model."""


def repo_root() -> Path:
    """Absolute path of the repository root (the parent of ``tests/``)."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Small YAML helpers
# --------------------------------------------------------------------------- #
def _load_documents(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [doc for doc in yaml.safe_load_all(handle) if doc is not None]


# --------------------------------------------------------------------------- #
# JSON pointers (RFC 6901) and JSON6902 patches
# --------------------------------------------------------------------------- #
def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _parse_pointer(pointer: str) -> list[str]:
    if pointer in ("", "/"):
        return []
    if not pointer.startswith("/"):
        raise UnsupportedKustomizeFeature(f"invalid JSON pointer: {pointer!r}")
    return [_unescape(token) for token in pointer.lstrip("/").split("/")]


def _resolve(document: Any, tokens: list[str]) -> Any:
    node = document
    for token in tokens:
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def _apply_json6902(document: Any, operations: list[dict]) -> Any:
    for operation in operations:
        op = operation.get("op")
        tokens = _parse_pointer(operation.get("path", ""))

        if not tokens:
            if op in ("add", "replace"):
                document = copy.deepcopy(operation.get("value"))
                continue
            raise UnsupportedKustomizeFeature(f"unsupported whole-document op {op!r}")

        key = tokens[-1]
        parent = _resolve(document, tokens[:-1])

        if op == "remove":
            if isinstance(parent, list):
                del parent[int(key)]
            else:
                parent.pop(key, None)
        elif op in ("add", "replace"):
            value = copy.deepcopy(operation.get("value"))
            if isinstance(parent, list):
                if key == "-":
                    parent.append(value)
                elif key.isdigit():
                    parent.insert(int(key), value)
                else:
                    raise UnsupportedKustomizeFeature(
                        f"non-numeric list index {key!r} in {operation.get('path')!r}"
                    )
            else:
                parent[key] = value
        else:
            raise UnsupportedKustomizeFeature(f"unsupported JSON6902 op {op!r}")
    return document


# --------------------------------------------------------------------------- #
# Strategic merge (only the merge-keyed lists this repository patches)
# --------------------------------------------------------------------------- #
_MERGE_KEYS = {
    "containers": "name",
    "initContainers": "name",
    "volumes": "name",
    "volumeMounts": "name",
    "env": "name",
    "ports": "name",
    "hostAliases": "ip",
}


def _strategic_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        result = copy.deepcopy(base)
        for key, value in patch.items():
            if key == "$patch":
                continue
            if isinstance(value, dict) and value.get("$patch") == "delete":
                result.pop(key, None)
                continue
            if key in result:
                merged = _strategic_merge(result[key], value)
                if merged is None:
                    result.pop(key, None)
                else:
                    result[key] = merged
            else:
                result[key] = copy.deepcopy(value)
        return result

    if isinstance(base, list) and isinstance(patch, list):
        # An explicit whole-list deletion (`- $patch: delete`) matched by name.
        if patch and all(
            isinstance(item, dict) and item.get("$patch") == "delete" for item in patch
        ):
            names = {item["name"] for item in patch if "name" in item}
            return [item for item in base if item.get("name") not in names]
        # Merge-keyed lists (containers, volumes, env, ...) are the only place
        # strategic merge differs from a plain replacement.
        for merge_key in _MERGE_KEYS.values():
            if patch and all(
                isinstance(item, dict) and merge_key in item for item in patch
            ):
                return _merge_keyed_list(base, patch, merge_key)
        return copy.deepcopy(patch)

    return copy.deepcopy(patch)


def _merge_keyed_list(base: list, patch: list, key: str) -> list:
    result = copy.deepcopy(base)
    for item in patch:
        name = item.get(key)
        position = next(
            (index for index, entry in enumerate(result) if entry.get(key) == name), None
        )
        if position is None:
            result.append(copy.deepcopy({k: v for k, v in item.items() if k != "$patch"}))
        else:
            result[position] = _strategic_merge(result[position], item)
    return result


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
@dataclass
class RenderedObject:
    """One rendered Kubernetes object plus the overlay file it came from."""

    document: dict
    source: str

    @property
    def kind(self) -> str:
        return self.document.get("kind", "")

    @property
    def name(self) -> str:
        return (self.document.get("metadata") or {}).get("name", "")

    @property
    def namespace(self) -> str:
        return (self.document.get("metadata") or {}).get("namespace", "")

    @property
    def identifier(self) -> tuple[str, str, str, str]:
        return (self.document.get("apiVersion", ""), self.kind, self.namespace, self.name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.kind} {self.namespace}/{self.name} from {self.source}>"


@dataclass
class _Context:
    """Inherited kustomization settings, threaded top-down."""

    directory: Path = field(default_factory=Path)
    namespace: str = ""
    common_labels: dict = field(default_factory=dict)
    generator_labels: dict = field(default_factory=dict)
    disable_name_suffix_hash: bool = False
    images: list = field(default_factory=list)

    def child(self, kustomization: dict, directory: Path) -> "_Context":
        options = kustomization.get("generatorOptions") or {}
        return replace(
            self,
            directory=directory,
            namespace=kustomization.get("namespace", self.namespace),
            common_labels={
                **self.common_labels,
                **(kustomization.get("commonLabels") or {}),
            },
            generator_labels={
                **self.generator_labels,
                **(options.get("labels") or {}),
            },
            disable_name_suffix_hash=options.get(
                "disableNameSuffixHash", self.disable_name_suffix_hash
            ),
            images=self.images + list(kustomization.get("images") or []),
        )


def _apply_image_overrides(document: Any, overrides: list[dict]) -> None:
    """Rewrite every ``image:`` leaf in place, matching on the original name."""
    if not overrides:
        return
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "image" and isinstance(value, str):
                document[key] = _rewrite_image(value, overrides)
            else:
                _apply_image_overrides(value, overrides)
    elif isinstance(document, list):
        for item in document:
            _apply_image_overrides(item, overrides)


def _rewrite_image(image: str, overrides: list[dict]) -> str:
    for override in overrides:
        if override.get("name") != image:
            continue
        new_name = override.get("newName", image)
        if override.get("digest"):
            return f"{new_name.split('@')[0].split(':')[0]}@{override['digest']}"
        if override.get("newTag"):
            # Strip any existing tag or digest from the repository part.
            repository = new_name.split("@")[0]
            last = repository.rsplit("/", 1)[-1]
            if ":" in last:
                repository = repository.rsplit(":", 1)[0]
            return f"{repository}:{override['newTag']}"
    return image


def _apply_common_labels(document: dict, labels: dict) -> None:
    if not labels:
        return
    metadata = document.setdefault("metadata", {})
    metadata["labels"] = {**(metadata.get("labels") or {}), **labels}
    if document.get("kind") == "Deployment" or document.get("kind") == "StatefulSet":
        spec = document.get("spec") or {}
        if isinstance(spec.get("selector"), dict):
            selector = spec["selector"]
            selector["matchLabels"] = {**(selector.get("matchLabels") or {}), **labels}
        template = spec.get("template") or {}
        template_meta = template.setdefault("metadata", {})
        template_meta["labels"] = {**(template_meta.get("labels") or {}), **labels}


def _generator_data(entry: dict, context: _Context) -> dict[str, str]:
    data: dict[str, str] = {}
    for literal in entry.get("literals") or []:
        key, _, value = str(literal).partition("=")
        if not _:
            raise UnsupportedKustomizeFeature(f"literal without '=': {literal!r}")
        data[key] = value
    for env_file in entry.get("envs") or []:
        env_path = (context.directory / env_file).resolve()
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip()
    for file_entry in entry.get("files") or []:
        if "=" in file_entry:
            key, _, source = file_entry.partition("=")
        else:
            key = Path(file_entry).name
            source = file_entry
        data[key] = (context.directory / source).resolve().read_text(encoding="utf-8")
    return data


def _build_generated(entry: dict, context: _Context, kind: str) -> dict:
    if entry.get("behavior") in ("merge", "replace"):
        raise UnsupportedKustomizeFeature(
            f"generator behavior {entry['behavior']!r} is not modelled"
        )
    if kind == "Secret" and entry.get("type"):
        raise UnsupportedKustomizeFeature("secretGenerator 'type' is not modelled")

    data = _generator_data(entry, context)
    metadata: dict[str, Any] = {
        "name": entry["name"],
        "labels": dict(context.generator_labels),
    }
    if context.namespace:
        metadata["namespace"] = context.namespace
    if kind == "ConfigMap":
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata,
            "data": data,
        }
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": metadata,
        "type": "Opaque",
        "stringData": data,
    }


def _selector_matches(target: dict, document: dict) -> bool:
    if target.get("kind") and target["kind"] != document.get("kind"):
        return False
    api_version = document.get("apiVersion", "")
    if target.get("group") and target["group"] != api_version.split("/")[0]:
        return False
    if target.get("version") and target["version"] != api_version.split("/")[-1]:
        return False
    metadata = document.get("metadata") or {}
    if target.get("name") and target["name"] != metadata.get("name"):
        return False
    if target.get("namespace") and target["namespace"] != metadata.get("namespace"):
        return False
    return True


def _apply_patch(entry: dict, documents: list[dict], context: _Context) -> list[dict]:
    """Apply one `patches:` entry.

    Handles every shape this repository uses:

    * inline ``patch: |-`` with an explicit ``target:`` (JSON6902 text);
    * a patch file whose single document is the bare patch body, self-targeted
      by its own ``kind``/``metadata.name``;
    * a patch file holding several such self-targeted documents;
    * a patch file whose document carries the structured ``patch:`` + ``target:``
      form.
    """
    if "path" in entry and "patch" not in entry:
        loaded = _load_documents((context.directory / entry["path"]).resolve())
        if len(loaded) == 1 and isinstance(loaded[0], list):
            # A bare JSON6902 operation list; its target must be supplied by the
            # kustomization entry itself.
            return _apply_one_patch(
                entry.get("target"), loaded[0], documents, source=entry["path"]
            )
        if len(loaded) == 1 and isinstance(loaded[0], dict) and "patch" in loaded[0]:
            document = loaded[0]
            target = entry.get("target") or document.get("target")
            if not target:
                # `patch:` present but no explicit target in either place, so
                # the document is itself the (self-describing) object.
                target = {
                    "kind": document.get("kind"),
                    "name": (document.get("metadata") or {}).get("name"),
                }
            return _apply_one_patch(
                target, document["patch"], documents, source=entry["path"]
            )
        # Multi-document (or single-document) strategic-merge file: every
        # document is its own patch and carries its own identity.
        for document in loaded:
            metadata = document.get("metadata") or {}
            target: dict[str, Any] = {
                "kind": document.get("kind"),
                "name": metadata.get("name"),
            }
            if metadata.get("namespace"):
                target["namespace"] = metadata["namespace"]
            documents = _apply_one_patch(target, document, documents, source=entry["path"])
        return documents

    target = entry.get("target")
    return _apply_one_patch(target, entry["patch"], documents, source="inline patch")


def _apply_one_patch(
    target: dict | None,
    operations: Any,
    documents: list[dict],
    source: str = "patch",
) -> list[dict]:
    if not target or not (target.get("kind") or target.get("name")):
        # A targetless patch is legal in Kustomize but ambiguous for a reader;
        # this repository always targets explicitly.
        raise UnsupportedKustomizeFeature(f"{source}: patch without a kind/name target")

    # An inline `patch: |-` block scalar arrives as a string and must be parsed
    # before it can be interpreted as JSON6902 or a strategic merge.
    if isinstance(operations, str):
        operations = yaml.safe_load(operations)

    result: list[dict] = []
    for document in documents:
        if "$patch" in document and len(document) == 1:
            result.append(document)
            continue
        if not _selector_matches(target, document):
            result.append(document)
            continue
        if isinstance(operations, list) and operations and all(
            isinstance(item, dict) and "op" in item for item in operations
        ):
            patched = _apply_json6902(copy.deepcopy(document), operations)
        else:
            patched = _strategic_merge(document, operations)
        if patched is None or patched.get("$patch") == "delete":
            continue
        result.append(patched)
    return result


def _resolve_entry(entry: str | dict, context: _Context) -> list[dict]:
    """Resolve one `resources:` entry (or `patch.path`) into documents."""
    if isinstance(entry, dict):
        raise UnsupportedKustomizeFeature("inline resource objects are not modelled")
    path = (context.directory / entry).resolve()
    if path.is_dir():
        documents: list[dict] = []
        kustomization = path / "kustomization.yaml"
        if kustomization.is_file():
            documents.extend(_render_directory(path, context))
        return documents
    if not path.is_file():
        raise UnsupportedKustomizeFeature(f"resource {entry!r} does not exist")
    return _load_documents(path)


def _render_directory(
    directory: Path,
    parent: _Context | None,
    accumulated: list[dict] | None = None,
) -> list[dict]:
    kustomization_path = directory / "kustomization.yaml"
    if not kustomization_path.is_file():
        return list(accumulated or [])
    kustomization = _load_documents(kustomization_path)[0]
    if kustomization.get("kind") not in ("Kustomization", "Component"):
        raise UnsupportedKustomizeFeature(
            f"{kustomization_path} is not a Kustomization/Component"
        )

    context = (
        _Context(
            directory=directory,
            namespace=kustomization.get("namespace", ""),
            common_labels=dict(kustomization.get("commonLabels") or {}),
            generator_labels=dict(
                (kustomization.get("generatorOptions") or {}).get("labels") or {}
            ),
            disable_name_suffix_hash=bool(
                (kustomization.get("generatorOptions") or {}).get(
                    "disableNameSuffixHash", False
                )
            ),
            images=list(kustomization.get("images") or []),
        )
        if parent is None
        else parent.child(kustomization, directory)
    )

    for unsupported_key in (
        "bases",
        "transformers",
        "vars",
        "replacements",
        "configurations",
        "crds",
        "helmCharts",
        "openapi",
    ):
        if kustomization.get(unsupported_key):
            raise UnsupportedKustomizeFeature(
                f"{kustomization_path}: {unsupported_key!r} is not modelled"
            )

    # Resources, components and this directory's own patches share one
    # accumulator: Kustomize lets a component patch an object contributed by the
    # kustomization that includes it (the monitoring component adds ingress
    # paths to the base Ingress, for example), which only works if the component
    # sees what has already accumulated.
    documents: list[dict] = list(accumulated or [])
    for entry in kustomization.get("resources") or []:
        documents.extend(_resolve_entry(entry, context))
    for entry in kustomization.get("components") or []:
        component_directory = (directory / entry).resolve()
        documents = _render_directory(component_directory, context, documents)

    options = kustomization.get("generatorOptions") or {}
    if options.get("annotations") or options.get("immutable"):
        raise UnsupportedKustomizeFeature("generatorOptions annotations/immutable")
    for section, kind in (("configMapGenerator", "ConfigMap"), ("secretGenerator", "Secret")):
        for entry in kustomization.get(section) or []:
            documents.append(_build_generated(entry, context, kind))

    for entry in kustomization.get("patches") or []:
        documents = _apply_patch(entry, documents, context)

    for document in documents:
        if not isinstance(document, dict):
            continue
        _apply_common_labels(document, context.common_labels)
        _apply_image_overrides(document, context.images)
        metadata = document.setdefault("metadata", {})
        if context.namespace and not metadata.get("namespace"):
            if document.get("kind") != "Namespace":
                metadata["namespace"] = context.namespace
    return documents


def render(overlay: str | Path) -> list[RenderedObject]:
    """Render an overlay (or component) directory into Kubernetes objects."""
    root = repo_root()
    directory = (root / overlay).resolve() if not Path(overlay).is_absolute() else Path(overlay)
    if not directory.is_dir():
        raise FileNotFoundError(f"overlay directory not found: {directory}")
    documents = _render_directory(directory, None)
    return [RenderedObject(document=document, source=str(directory)) for document in documents]


# --------------------------------------------------------------------------- #
# Optional cross-check against the real Kustomize
# --------------------------------------------------------------------------- #
def render_with_kubectl(overlay: str | Path) -> list[dict] | None:
    """Render with `kubectl kustomize`, or ``None`` when kubectl is absent.

    Used by the test suite as a differential check: when a developer has
    kubectl installed, the hand-written renderer's output is compared against the
    authoritative one.
    """
    import shutil
    import subprocess  # noqa: S404 - test-only, fixed argument vector

    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return None
    root = repo_root()
    directory = (root / overlay).resolve() if not Path(overlay).is_absolute() else Path(overlay)
    completed = subprocess.run(  # noqa: S603
        [kubectl, "kustomize", str(directory)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise UnsupportedKustomizeFeature(
            f"kubectl kustomize failed for {overlay}:\n{completed.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(completed.stdout) if doc]
