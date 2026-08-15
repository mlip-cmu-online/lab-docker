#!/usr/bin/env python3
"""Generate the Lab 5 Docker submission report and manifest without running Docker."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit


REPORT_NAME = "docker-report.html"
MANIFEST_NAME = "docker-manifest.json"
CHECKER_VERSION = "1.0"
SOURCE_PATHS = (
    "docker/training/Dockerfile",
    "docker/training/requirements.txt",
    "docker/training/train.py",
    "docker/inference/Dockerfile",
    "docker/inference/requirements.txt",
    "docker/inference/server.py",
    "docker-compose.yml",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
    re.compile(r"\b(?:ghp|github_pat|sk|xox[baprs])_[A-Za-z0-9_-]{16,}\b"),
)


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def check(name: str, present: bool, location: str, identifier: str, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "present" if present else "missing",
        "location": location,
        "identifier": identifier,
        "detail": detail,
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def sanitize_repository_url(value: str) -> tuple[str, bool]:
    """Return a credential-free HTTPS repository URL and whether it is valid."""
    value = value.strip()
    scp_match = re.fullmatch(r"git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?", value)
    if scp_match:
        owner, repository = scp_match.groups()
        repository = repository.removesuffix(".git")
        return f"https://github.com/{owner}/{repository}", True

    try:
        parsed = urlsplit(value)
    except ValueError:
        return "invalid-url-redacted", False

    host = parsed.hostname or ""
    safe = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    path_parts = [part for part in parsed.path.removesuffix(".git").split("/") if part]
    valid = (
        parsed.scheme == "https"
        and host.lower() == "github.com"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and len(path_parts) == 2
    )
    if not valid:
        return safe or "invalid-url-redacted", False
    return f"https://github.com/{path_parts[0]}/{path_parts[1]}", True


def parse_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(read_text(path))
    except (json.JSONDecodeError, OSError):
        return None


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def has_all(text: str, patterns: tuple[str, ...]) -> bool:
    return all(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def has_executable_ellipsis(text: str) -> bool:
    return bool(
        re.search(r"^\s*[A-Za-z_]\w*\s*=\s*\.\.\.\s*(?:#.*)?$", text, flags=re.MULTILINE)
        or re.search(r"\(\s*\.\.\.\s*\)", text)
    )


def completed_training_files(files: dict[str, str]) -> tuple[bool, str]:
    dockerfile = files.get("docker/training/Dockerfile", "")
    requirements = files.get("docker/training/requirements.txt", "")
    train = files.get("docker/training/train.py", "")
    dockerfile_ok = has_all(
        dockerfile,
        (
            r"^\s*FROM\s+python:3\.11-slim\s*$",
            r"^\s*WORKDIR\s+/app\s*$",
            r"pip\s+install[^\n]*requirements\.txt",
            r"COPY\s+docker/training/train\.py\s+(?:\./train\.py|/app/train\.py)",
            r"^\s*(?:CMD|ENTRYPOINT)\s+\[?[^\n]*python(?:3)?[^\n]*train\.py",
        ),
    ) and "CMD []" not in dockerfile
    train_ok = has_all(
        train,
        (
            r"RandomForestClassifier\s*\(",
            r"\.fit\s*\(",
            r"joblib\.dump\s*\(",
            r"wine_model\.pkl",
        ),
    ) and not has_executable_ellipsis(train)
    requirements_ok = all(name in requirements.lower() for name in ("scikit-learn", "joblib", "numpy"))
    ok = dockerfile_ok and train_ok and requirements_ok
    return ok, "Training Dockerfile, requirements, and train.py contain the required executable structure." if ok else "Complete the training Dockerfile, requirements, and executable train.py placeholders."


def completed_inference_files(files: dict[str, str]) -> tuple[bool, str]:
    dockerfile = files.get("docker/inference/Dockerfile", "")
    requirements = files.get("docker/inference/requirements.txt", "")
    server = files.get("docker/inference/server.py", "")
    dockerfile_ok = has_all(
        dockerfile,
        (
            r"^\s*FROM\s+python:3\.11-slim\s*$",
            r"^\s*WORKDIR\s+/app\s*$",
            r"pip\s+install[^\n]*requirements\.txt",
            r"COPY\s+docker/inference/server\.py\s+(?:\./server\.py|/app/server\.py)",
            r"^\s*EXPOSE\s+808[01]\s*$",
            r"^\s*(?:CMD|ENTRYPOINT)\s+\[?[^\n]*python(?:3)?[^\n]*server\.py",
        ),
    )
    requirements_ok = all(name in requirements.lower() for name in ("flask", "scikit-learn", "joblib", "numpy"))
    server_ok = has_all(
        server,
        (
            r"joblib\.load\s*\(",
            r"request\.get_json\s*\(",
            r"[\[\(]['\"]input['\"][\]\)]",
            r"\.predict\s*\(",
            r"open\s*\([^\n]*LOG_PATH[^\n]*(?:['\"]a['\"]|mode\s*=\s*['\"]a['\"])",
            r"@app\.route\s*\(\s*['\"]/health['\"]",
        ),
    ) and not has_executable_ellipsis(server)
    ok = dockerfile_ok and requirements_ok and server_ok
    return ok, "Inference Dockerfile, requirements, server routes, prediction, and append-only logging structure are present." if ok else "Complete the inference Dockerfile, requirements, model loading, prediction, logging, and health route."


def completed_compose(text: str) -> tuple[bool, str]:
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    ok = has_all(
        uncommented,
        (
            r"^\s*services\s*:\s*$",
            r"^\s+training\s*:\s*$",
            r"^\s+inference\s*:\s*$",
            r"docker/training/Dockerfile",
            r"docker/inference/Dockerfile",
            r"(?:^|[\s'\"])(?:\./)?logs\s*:\s*/app/logs(?:$|[\s'\"])",
            r"8081\s*:\s*8080",
            r"^volumes\s*:\s*$",
        ),
    ) and len(re.findall(r"wine_model_storage\s*:\s*/app/models", uncommented)) >= 2
    ok = ok and len(re.findall(r"wine_model_storage\s*:", uncommented)) >= 3
    return ok, "Compose defines both builds, the shared named volume, host log bind mount, port mapping, and top-level volume." if ok else "Complete both Compose services, both model mounts, the log bind mount, 8081:8080 mapping, and named-volume definition."


def volume_record(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return None


def health_record(value: Any) -> Optional[dict[str, Any]]:
    return value if isinstance(value, dict) and "status" in value and "model_loaded" in value else None


def evidence_section(title: str, location: str, content: str, withheld: bool) -> str:
    body = "Content withheld because an obvious credential pattern was detected." if withheld else content
    return (
        f'<section><h3>{html.escape(title)}</h3><p><code>{html.escape(location)}</code></p>'
        f'<pre>{html.escape(body)}</pre></section>'
    )


def render_report(
    learner: str,
    generated_at: str,
    repository_url: str,
    repository_valid: bool,
    commit: str,
    checks: list[dict[str, str]],
    source_files: dict[str, str],
    evidence_files: list[tuple[str, str, str]],
    secret_locations: set[str],
) -> str:
    overall = "complete" if all(item["status"] == "present" for item in checks) else "incomplete"
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td class=\"{item['status']}\">{item['status']}</td>"
        f"<td>{html.escape(item['identifier'])}</td>"
        f"<td>{html.escape(item['detail'])}</td>"
        "</tr>"
        for item in checks
    )
    commit_url = f"{repository_url}/commit/{commit}" if repository_valid and commit else repository_url
    source_sections = "\n".join(
        evidence_section(path, path, content, path in secret_locations)
        for path, content in source_files.items()
    )
    saved_sections = "\n".join(
        evidence_section(label, path, content, path in secret_locations)
        for label, path, content in evidence_files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lab 5 Docker Submission Report</title>
  <style>
    body {{ color:#17202a; font:16px/1.45 system-ui,sans-serif; margin:2rem auto; max-width:76rem; padding:0 1rem; }}
    h1,h2,h3 {{ color:#102a43; }} table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid #bcccdc; padding:.55rem; text-align:left; vertical-align:top; }} th {{ background:#f0f4f8; }}
    pre {{ background:#f5f7fa; border:1px solid #bcccdc; overflow:auto; padding:1rem; white-space:pre-wrap; }}
    code {{ overflow-wrap:anywhere; }} .present,.complete {{ color:#176b3a; font-weight:700; }} .missing,.incomplete {{ color:#a61b1b; font-weight:700; }}
    .note {{ background:#fffbea; border-left:.3rem solid #d69e2e; padding:.7rem 1rem; }}
  </style>
</head>
<body>
  <h1>Lab 5: Docker Submission Report</h1>
  <table>
    <tr><th>Learner</th><td>{html.escape(learner)}</td></tr>
    <tr><th>Generated</th><td>{html.escape(generated_at)}</td></tr>
    <tr><th>Overall completeness</th><td class="{overall}">{overall}</td></tr>
    <tr><th>Immutable repository revision</th><td><a href="{html.escape(commit_url)}">{html.escape(commit_url)}</a></td></tr>
    <tr><th>Commit SHA</th><td><code>{html.escape(commit)}</code></td></tr>
    <tr><th>Checker version</th><td>{CHECKER_VERSION}</td></tr>
  </table>

  <h2>Completeness</h2>
  <table><thead><tr><th>Required evidence</th><th>Status</th><th>Identifier</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table>

  <h2>Saved execution evidence</h2>
  {saved_sections}

  <h2 id="completed-container-files-at-the-reported-revision">Completed container files at the reported revision</h2>
  {source_sections}

  <h2>Staff spot checks</h2>
  <p class="note">This checker confirms objective structure and matching saved evidence only. Answer these separately in Canvas; it does not judge either answer.</p>
  <ol>
    <li><strong>Persistence:</strong> explain what persisted after named-volume removal and distinguish the named volume from the bind mount.</li>
    <li><strong>Reproducibility:</strong> identify one reproducible image choice and one remaining portability limit.</li>
  </ol>

  <h2>Safety check</h2>
  <p>The checker scans included text for common plaintext credential patterns. Review images and the finished report yourself before uploading.</p>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Lab 5 Docker HTML report and JSON manifest using saved evidence only.")
    parser.add_argument("--learner", required=True)
    parser.add_argument("--repository-url", required=True, help="HTTPS or git@github.com repository URL")
    parser.add_argument("--training-output", required=True, type=Path)
    parser.add_argument("--service-output", required=True, type=Path)
    parser.add_argument("--volume-inspection", required=True, type=Path)
    parser.add_argument("--prediction-response", required=True, type=Path)
    parser.add_argument("--prediction-log", required=True, type=Path)
    parser.add_argument("--health-before", required=True, type=Path)
    parser.add_argument("--health-after", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("submission"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_result = run_git(Path.cwd(), "rev-parse", "--show-toplevel")
    if root_result.returncode != 0:
        print("Run this command from inside the lab-docker repository.", file=sys.stderr)
        return 2
    root = Path(root_result.stdout.strip())

    inputs = {
        "training build/run output": args.training_output,
        "inference build/service output": args.service_output,
        "named-volume inspection": args.volume_inspection,
        "prediction response": args.prediction_response,
        "bind-mounted prediction log": args.prediction_log,
        "health before volume removal": args.health_before,
        "health after volume removal": args.health_after,
    }
    missing_inputs = [f"{label}: {path}" for label, path in inputs.items() if not path.is_file()]
    if missing_inputs:
        print("Missing required saved evidence:\n  " + "\n  ".join(missing_inputs), file=sys.stderr)
        return 2

    head_result = run_git(root, "rev-parse", "HEAD")
    if head_result.returncode != 0:
        print("The repository has no commit to use as an immutable revision.", file=sys.stderr)
        return 2
    head = head_result.stdout.strip()
    repository_url, repository_valid = sanitize_repository_url(args.repository_url)
    origin_result = run_git(root, "remote", "get-url", "origin")
    origin_url, origin_valid = sanitize_repository_url(origin_result.stdout) if origin_result.returncode == 0 else ("origin-not-found", False)
    repository_matches_origin = repository_valid and origin_valid and repository_url.lower() == origin_url.lower()

    source_files = {path: read_text(root / path) if (root / path).is_file() else "" for path in SOURCE_PATHS}
    tracked = set(run_git(root, "ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines())
    diff_result = run_git(root, "diff", "--quiet", "HEAD", "--", *SOURCE_PATHS)
    immutable = all(path in tracked for path in SOURCE_PATHS) and diff_result.returncode == 0 and repository_matches_origin

    training_ok, training_detail = completed_training_files(source_files)
    inference_ok, inference_detail = completed_inference_files(source_files)
    compose_ok, compose_detail = completed_compose(source_files.get("docker-compose.yml", ""))

    evidence_text = {label: read_text(path) for label, path in inputs.items()}
    training_text = evidence_text["training build/run output"]
    service_text = evidence_text["inference build/service output"]
    volume = volume_record(parse_json(args.volume_inspection))
    prediction = parse_json(args.prediction_response)
    log_text = evidence_text["bind-mounted prediction log"]
    health_before = health_record(parse_json(args.health_before))
    health_after = health_record(parse_json(args.health_after))

    build_marker = r"successfully built|exporting to image|load build definition|\bbuilt\b|\bbuilding\b"
    training_evidence_ok = has_all(training_text, (build_marker, r"training complete", r"model saved"))
    service_evidence_ok = has_all(service_text, (build_marker, r"inference", r"running on|started|healthy"))
    volume_name = str(volume.get("Name", "")) if volume else ""
    volume_mount = str(volume.get("Mountpoint", "")) if volume else ""
    volume_ok = bool(volume and volume_name.endswith("wine_model_storage") and volume_mount)
    prediction_label = str(prediction.get("prediction", "")) if isinstance(prediction, dict) else ""
    prediction_ok = bool(prediction_label and prediction_label in log_text and re.search(r"input\s*:", log_text, re.IGNORECASE))
    before_ok = bool(health_before and health_before.get("model_loaded") is True and str(health_before.get("status", "")).lower() == "healthy")
    after_status = str(health_after.get("status", "")).lower() if health_after else ""
    after_ok = bool(health_after and health_after.get("model_loaded") is False and "not found" in after_status)

    displayed_paths: dict[str, str] = {}
    for label, path in inputs.items():
        try:
            displayed_paths[label] = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            displayed_paths[label] = path.name

    secret_locations: set[str] = set()
    for path, content in source_files.items():
        if contains_secret(content):
            secret_locations.add(path)
    for label, content in evidence_text.items():
        if contains_secret(content):
            secret_locations.add(displayed_paths[label])
    secrets_ok = not secret_locations

    report_location = REPORT_NAME
    checks = [
        check("immutable_repository_revision", immutable, report_location, head, "All required container files are tracked and unchanged at HEAD, and the credential-free repository URL matches origin." if immutable else "Commit all required container files and provide the credential-free GitHub URL configured as origin."),
        check("completed_training_files", training_ok, f"{report_location}#completed-container-files-at-the-reported-revision", hashlib.sha256("\n".join(source_files[path] for path in SOURCE_PATHS[:3]).encode()).hexdigest()[:16], training_detail),
        check("completed_inference_files", inference_ok, f"{report_location}#completed-container-files-at-the-reported-revision", hashlib.sha256("\n".join(source_files[path] for path in SOURCE_PATHS[3:6]).encode()).hexdigest()[:16], inference_detail),
        check("compose_configuration", compose_ok, f"{report_location}#completed-container-files-at-the-reported-revision", "wine_model_storage", compose_detail),
        check("training_build_and_run_output", training_evidence_ok, displayed_paths["training build/run output"], "image built; training complete; model saved" if training_evidence_ok else "markers missing", "Saved output contains image-build, completed-training, and model-save markers." if training_evidence_ok else "Save the training build/run output containing image-build, completion, and model-save markers."),
        check("inference_build_and_service_output", service_evidence_ok, displayed_paths["inference build/service output"], "image built; inference service started" if service_evidence_ok else "markers missing", "Saved output identifies the image build, inference service, and a startup/health marker." if service_evidence_ok else "Save inference build/service output containing image-build, service-name, and startup or health markers."),
        check("named_volume_inspection", volume_ok, displayed_paths["named-volume inspection"], volume_name or "not parsed", "Parsed Docker volume name and non-empty mountpoint matching the Compose volume." if volume_ok else "Save raw JSON from docker volume inspect for the wine_model_storage volume."),
        check("prediction_and_bind_mounted_log", prediction_ok, displayed_paths["bind-mounted prediction log"], prediction_label or "not parsed", "The prediction label appears in a saved log entry that includes the request input." if prediction_ok else "Save a JSON prediction response and the corresponding host-side log entry with the same label and an input field."),
        check("health_before_volume_removal", before_ok, displayed_paths["health before volume removal"], compact_json(health_before) if health_before else "not parsed", "Saved response reports healthy with the model loaded." if before_ok else "Save the pre-removal JSON health response showing healthy and model_loaded true."),
        check("health_after_volume_removal", after_ok, displayed_paths["health after volume removal"], compact_json(health_after) if health_after else "not parsed", "Saved response reports model not found and not loaded." if after_ok else "Save the post-removal JSON health response showing model not found and model_loaded false."),
        check("obvious_credential_leakage", secrets_ok, report_location, "none detected" if secrets_ok else "content withheld", "No common plaintext credential pattern was found." if secrets_ok else "Remove and revoke plaintext credentials before regenerating; affected content was withheld."),
    ]

    evidence_files = [(label, displayed_paths[label], evidence_text[label]) for label in inputs]
    generated_at = datetime.now(timezone.utc).isoformat()
    report = render_report(args.learner, generated_at, repository_url, repository_valid, head, checks, source_files, evidence_files, secret_locations)
    complete = all(item["status"] == "present" for item in checks)
    manifest = {
        "schema_version": "1.0",
        "lab": "Lab 5: Docker",
        "learner": args.learner,
        "generated_at": generated_at,
        "checker_version": CHECKER_VERSION,
        "complete": complete,
        "report": REPORT_NAME,
        "identifiers": {"repository_url": repository_url, "commit_sha": head, "volume_name": volume_name, "prediction": prediction_label},
        "checks": checks,
        "manual_spot_checks": ["persistence", "reproducibility"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    (args.output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / REPORT_NAME}")
    print(f"Wrote {args.output_dir / MANIFEST_NAME}")
    print("Submission evidence is complete." if complete else "Submission evidence is incomplete; open the report for details.")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
