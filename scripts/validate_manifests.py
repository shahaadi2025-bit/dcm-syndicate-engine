"""Structural validation for Kubernetes manifests (no cluster required).

Checks every document in deploy/*.yaml for:
  * required fields (apiVersion, kind, metadata.name)
  * Deployment selector matching template labels
  * container `args` referencing modules that actually exist in the repo
  * env var names matching what dcm_engine actually reads at runtime

This catches the two classes of "manifest looks fine but the pod would
boot broken" bugs: pointing at phantom modules and setting env vars the
code never reads.

    python scripts/validate_manifests.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "deploy"

# Env vars the codebase actually consumes (keep in sync with dcm_engine).
KNOWN_ENV_VARS = {
    "DCM_KAFKA_BOOTSTRAP",  # consumer/producer/bridge bootstrap
    "DCM_CONSUMER_MAX_MESSAGES",  # consumer debug cap
    "DCM_PRODUCER_RATE",  # producer CLI knobs
    "DCM_PRODUCER_DURATION",
    "DCM_LLM_PROVIDER",  # agent provider selection
    "DCM_LLM_MODEL",
    "OPENAI_API_KEY",  # langchain provider credentials
    "DCM_MODEL_DIR",  # ML artifact location
    "POD_NAME",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
}


def check_doc(doc: dict[str, Any], source: str, errors: list[str]) -> None:
    """Run structural checks on one manifest document."""
    for field in ("apiVersion", "kind", "metadata"):
        if field not in doc:
            errors.append(f"{source}: missing required field '{field}'")
            return
    name = str(doc.get("metadata", {}).get("name", ""))
    if not name:
        errors.append(f"{source}: missing metadata.name")
    kind = str(doc.get("kind"))

    if kind == "Deployment":
        spec = doc.get("spec", {})
        selector = spec.get("selector", {}).get("matchLabels", {})
        template_labels = spec.get("template", {}).get("metadata", {}).get("labels", {})
        for k, v in selector.items():
            if template_labels.get(k) != v:
                errors.append(
                    f"{source}: selector '{k}={v}' does not match template labels"
                )
        for container in spec.get("template", {}).get("spec", {}).get("containers", []):
            args = list(container.get("args") or [])
            module = args[2] if len(args) > 2 and args[0] == "python" and args[1] == "-m" else None
            if module:
                mod_path = REPO_ROOT / (module.replace(".", "/") + ".py")
                if not mod_path.exists():
                    errors.append(
                        f"{source}: container '{container.get('name')}' runs "
                        f"nonexistent module {module}"
                    )
            for env in container.get("env", []):
                key = str(env.get("name", ""))
                if key and key not in KNOWN_ENV_VARS:
                    errors.append(
                        f"{source}: container '{container.get('name')}' sets "
                        f"'{key}' which dcm_engine never reads"
                    )


def main() -> int:
    errors: list[str] = []
    files = sorted(MANIFEST_DIR.glob("*.yaml")) + sorted(MANIFEST_DIR.glob("*.yml"))
    if not files:
        print(f"FAIL: no manifests found in {MANIFEST_DIR}")
        return 1
    for path in files:
        docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict):
                errors.append(f"{path.name}[{i}]: not a mapping")
                continue
            check_doc(doc, f"{path.name}[{i}]", errors)
        print(f"checked {path.name}: {len(docs)} document(s)")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("MANIFESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
