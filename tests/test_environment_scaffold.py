import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PREFLIGHT = ROOT / "scripts" / "preflight.sh"


class EnvironmentScaffoldTest(unittest.TestCase):
    def test_codespace_enables_docker_and_forwards_inference_port(self) -> None:
        config = json.loads(
            (ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")
        )

        self.assertIn("ghcr.io/devcontainers/features/docker-in-docker:2", config["features"])
        self.assertIn(8081, config["forwardPorts"])
        self.assertEqual(config["postCreateCommand"], "bash scripts/preflight.sh")

    def test_preflight_shell_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(PREFLIGHT)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preflight_checks_named_and_bind_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_bin = root / "bin"
            state = root / "state"
            fake_bin.mkdir()
            state.mkdir()
            fake_docker = fake_bin / "docker"
            fake_docker.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/python3
                    import os
                    import pathlib
                    import sys

                    args = sys.argv[1:]
                    state = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
                    if args[:2] in (["compose", "version"], ["volume", "rm"]):
                        raise SystemExit(0)
                    if args == ["info"] or args[:1] == ["ps"]:
                        raise SystemExit(0)
                    if args[:2] == ["volume", "create"]:
                        (state / "volume-check").write_text("", encoding="utf-8")
                        print(args[2])
                        raise SystemExit(0)
                    if args[:1] == ["run"]:
                        mount = args[args.index("--mount") + 1]
                        fields = dict(field.split("=", 1) for field in mount.split(",") if "=" in field)
                        if fields["type"] == "volume":
                            check = state / "volume-check"
                            if "cat" in args:
                                print(check.read_text(encoding="utf-8"), end="")
                            else:
                                check.write_text("named-volume-ok", encoding="utf-8")
                        else:
                            pathlib.Path(fields["source"], "from-container").write_text(
                                "bind-mount-ok", encoding="utf-8"
                            )
                        raise SystemExit(0)
                    raise SystemExit(f"unexpected fake docker call: {args}")
                    """
                ),
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            env = dict(os.environ)
            env["FAKE_DOCKER_STATE"] = str(state)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(PREFLIGHT)],
                cwd=ROOT,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Docker preflight: PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
