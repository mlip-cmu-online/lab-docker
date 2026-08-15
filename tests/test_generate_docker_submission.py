import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate-docker-submission.py"


def run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class GenerateDockerSubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        run("git", "init", "-b", "main", cwd=self.root)
        run("git", "config", "user.name", "Test Learner", cwd=self.root)
        run("git", "config", "user.email", "learner@example.com", cwd=self.root)
        run(
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/test-learner/lab-docker.git",
            cwd=self.root,
        )
        self._write_complete_source()
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        (self.evidence / "training.txt").write_text("#1 exporting to image\nTraining complete. Test accuracy: 0.9722\nModel saved to /app/models/wine_model.pkl\n", encoding="utf-8")
        (self.evidence / "service.txt").write_text("inference  Built\ninference  | Running on http://127.0.0.1:8080\n", encoding="utf-8")
        (self.evidence / "volume.json").write_text(json.dumps([{"Name": "wine_model_storage", "Mountpoint": "/var/lib/docker/volumes/wine_model_storage/_data"}]), encoding="utf-8")
        (self.evidence / "prediction.json").write_text('{"prediction":"class_0"}\n', encoding="utf-8")
        (self.evidence / "predictions.log").write_text("2026-08-15 12:00:00 | input: [13.2, 1.78] | prediction: class_0\n", encoding="utf-8")
        (self.evidence / "health-before.json").write_text('{"model_loaded":true,"status":"healthy"}\n', encoding="utf-8")
        (self.evidence / "health-after.json").write_text('{"model_loaded":false,"status":"model not found"}\n', encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

    def _write_complete_source(self) -> None:
        self._write("docker/training/Dockerfile", """
            FROM python:3.11-slim
            WORKDIR /app
            COPY docker/training/requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt
            COPY docker/training/train.py ./train.py
            VOLUME ["/app/models"]
            CMD ["python", "train.py"]
        """)
        self._write("docker/training/requirements.txt", "scikit-learn==1.4.2\njoblib==1.4.2\nnumpy==1.26.4\n")
        self._write("docker/training/train.py", """
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            clf = RandomForestClassifier(random_state=42)
            clf.fit([[0], [1]], [0, 1])
            joblib.dump(clf, "/app/models/wine_model.pkl")
        """)
        self._write("docker/inference/Dockerfile", """
            FROM python:3.11-slim
            WORKDIR /app
            COPY docker/inference/requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt
            COPY docker/inference/server.py ./server.py
            EXPOSE 8080
            CMD ["python", "server.py"]
        """)
        self._write("docker/inference/requirements.txt", "Flask==3.0.3\nscikit-learn==1.4.2\njoblib==1.4.2\nnumpy==1.26.4\n")
        self._write("docker/inference/server.py", """
            from flask import Flask, request
            import joblib
            app = Flask(__name__)
            MODEL_PATH = "/app/models/wine_model.pkl"
            LOG_PATH = "/app/logs/predictions.log"
            # Example input excerpt: [13.2, 1.78, ...]
            model = joblib.load(MODEL_PATH)
            @app.route('/predict', methods=['POST'])
            def predict():
                features = request.get_json()["input"]
                prediction = model.predict([features])
                with open(LOG_PATH, "a") as stream:
                    stream.write(f"input: {features} | prediction: {prediction[0]}")
            @app.route('/health')
            def health():
                return {"status": "healthy", "model_loaded": model is not None}
        """)
        self._write("docker-compose.yml", """
            services:
              training:
                build:
                  context: .
                  dockerfile: docker/training/Dockerfile
                volumes:
                  - wine_model_storage:/app/models
              inference:
                build:
                  context: .
                  dockerfile: docker/inference/Dockerfile
                volumes:
                  - wine_model_storage:/app/models
                  - ./logs:/app/logs
                ports:
                  - "8081:8080"
            volumes:
              wine_model_storage:
        """)
        run("git", "add", ".", cwd=self.root)
        run("git", "commit", "-m", "complete lab", cwd=self.root)

    def generate(self, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return run(
            "python3", str(SCRIPT), "--learner", "Test Learner",
            "--repository-url", "https://github.com/test-learner/lab-docker",
            "--training-output", str(self.evidence / "training.txt"),
            "--service-output", str(self.evidence / "service.txt"),
            "--volume-inspection", str(self.evidence / "volume.json"),
            "--prediction-response", str(self.evidence / "prediction.json"),
            "--prediction-log", str(self.evidence / "predictions.log"),
            "--health-before", str(self.evidence / "health-before.json"),
            "--health-after", str(self.evidence / "health-after.json"),
            cwd=self.root, env=env,
        )

    def test_generates_complete_report_and_matching_manifest_without_docker(self) -> None:
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        marker = self.root / "docker-was-called"
        docker = fake_bin / "docker"
        docker.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n", encoding="utf-8")
        docker.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"

        result = self.generate(env)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
        report = (self.root / "submission" / "docker-report.html").read_text(encoding="utf-8")
        manifest = json.loads((self.root / "submission" / "docker-manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        self.assertTrue(all(item["status"] == "present" for item in manifest["checks"]))
        self.assertIn("class_0", report)
        self.assertIn("wine_model_storage", report)
        self.assertEqual(manifest["manual_spot_checks"], ["persistence", "reproducibility"])

    def test_marks_dirty_container_source_and_mismatched_log_incomplete(self) -> None:
        server = self.root / "docker/inference/server.py"
        server.write_text(server.read_text(encoding="utf-8") + "\n# uncommitted\n", encoding="utf-8")
        (self.evidence / "predictions.log").write_text("input: [13.2] | prediction: class_2\n", encoding="utf-8")

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        manifest = json.loads((self.root / "submission" / "docker-manifest.json").read_text(encoding="utf-8"))
        checks = {item["name"]: item["status"] for item in manifest["checks"]}
        self.assertEqual(checks["immutable_repository_revision"], "missing")
        self.assertEqual(checks["prediction_and_bind_mounted_log"], "missing")

    def test_marks_committed_server_placeholder_incomplete(self) -> None:
        server = self.root / "docker/inference/server.py"
        server.write_text(
            server.read_text(encoding="utf-8").replace(
                "model = joblib.load(MODEL_PATH)", "model = ...  # TODO"
            ),
            encoding="utf-8",
        )
        run("git", "commit", "-am", "leave server incomplete", cwd=self.root)

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        manifest = json.loads(
            (self.root / "submission" / "docker-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        checks = {item["name"]: item["status"] for item in manifest["checks"]}
        self.assertEqual(checks["immutable_repository_revision"], "present")
        self.assertEqual(checks["completed_inference_files"], "missing")

    def test_flags_and_withholds_plaintext_credentials(self) -> None:
        secret = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        service = self.evidence / "service.txt"
        service.write_text(service.read_text(encoding="utf-8") + f"api_key={secret}\n", encoding="utf-8")

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        combined = (self.root / "submission" / "docker-report.html").read_text(encoding="utf-8") + (self.root / "submission" / "docker-manifest.json").read_text(encoding="utf-8")
        self.assertNotIn(secret, combined)
        self.assertIn("Content withheld", combined)

    def test_rejects_missing_evidence_without_creating_outputs(self) -> None:
        (self.evidence / "health-after.json").unlink()

        result = self.generate()

        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing required saved evidence", result.stderr)
        self.assertFalse((self.root / "submission").exists())


if __name__ == "__main__":
    unittest.main()
