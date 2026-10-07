#!/usr/bin/env python3
"""Reproduce the required Linux registry profile with disposable infrastructure."""
import argparse
import json
import shlex
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent


def main():
    from process_runner import run_process
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("check", "test"))
    parser.add_argument("--source")
    parser.add_argument("--version")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    pins = json.loads((ROOT / "scripts/toolchain.json").read_text())
    registry = json.loads((ROOT / "scripts/check-registry.json").read_text())
    name = "ragy-check-" + uuid.uuid4().hex[:12]
    image = "ragy-check-linux:go" + pins["go"] + "-lint" + pins["golangci_lint"]
    env = dict(os.environ)
    def run(argv, timeout=900):
        result = run_process(argv, ROOT, env, timeout)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv)
    with tempfile.TemporaryDirectory(prefix="ragy-linux-") as temporary:
        directory = Path(temporary)
        checkout = directory / "checkout"
        run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(checkout)])
        selected = args.source or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", selected], check=True)
        dockerfile = directory / "Dockerfile"
        dockerfile.write_text(f'''FROM {registry["linux_image"]}
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv docker.io git gcc ca-certificates && rm -rf /var/lib/apt/lists/*
ENV GOTOOLCHAIN=go{pins["go"]} GOWORK=off
RUN GOBIN=/usr/local/bin go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v{pins["golangci_lint"]}
RUN python3 -m venv /opt/pdf && /opt/pdf/bin/pip install pdfplumber=={registry["pdf"]["pdfplumber"]} pypdf=={registry["pdf"]["pypdf"]}
ENV GOLANGCI_LINT=/usr/local/bin/golangci-lint RAGY_PDF_PYTHON=/opt/pdf/bin/python
WORKDIR /work
''')
        run(["docker", "build", "-t", image, str(directory)], timeout=1800)
        try:
            run(["docker", "run", "-d", "--name", name, "--label", "ragy.task20=T09", "-e", "POSTGRES_PASSWORD=isolated-test-only", "-e", "POSTGRES_DB=ragy", registry["postgres_image"]])
            run(["docker", "exec", name, "sh", "-c", "until psql -U postgres -d ragy -tAc 'SELECT 1' >/dev/null 2>&1; do sleep 1; done"], timeout=60)
            run(["docker", "exec", name, "psql", "-U", "postgres", "-d", "ragy", "-c", "CREATE EXTENSION vector"] )
            command = ["docker", "run", "--rm", "-v", str(checkout) + ":/source:ro", "-v", "/var/run/docker.sock:/var/run/docker.sock",
                       "-e", "RAGY_PG_TEST_CONTAINER=" + name]
            for key in ("RAGY_CANDIDATE_VERSION",):
                if env.get(key):
                    command.extend(["-e", key + "=" + env[key]])
            runner = ["python3", "scripts/verify.py", args.profile]
            for key in ("source", "version"):
                if getattr(args, key):
                    runner.extend(["--" + key, getattr(args, key)])
            if args.candidate:
                runner.append("--candidate")
            if args.output:
                output = Path(args.output).resolve()
                output.mkdir(parents=True, exist_ok=True)
                command.extend(["-v", str(output) + ":/results"])
                runner.extend(["--output", "/results"])
            command.extend([image, "sh", "-c", "cp -a /source/. /work/ && git config --global --add safe.directory /work && " + shlex.join(runner)])
            run(command, timeout=7200)
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
