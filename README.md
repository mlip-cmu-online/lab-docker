# Lab 5: Containerizing ML Models with Docker

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/mlip-cmu-online/lab-docker?quickstart=1)

## Overview

In this lab, you will containerize a machine learning training pipeline and inference server using Docker. You will train a Wine classifier, serve it via Flask, and orchestrate both containers with Docker Compose. A key focus of this lab is **Docker volume management** — you will use both named volumes and bind mounts to share data between containers and persist artifacts on the host.

### Deliverables

- [ ] **Deliverable 1**: The training script has been run in a container and the resulting model is saved to a shared volume. Able to explain why Docker is useful for reproducibility and portability in ML training scenarios.

- [ ] **Deliverable 2**: Containerize the inference service to serve predictions on a specific port and save the host-side `./logs/predictions.log` file as evidence.
       Explain what the Dockerfile is and how it helps containerize the inference service.

- [ ] **Deliverable 3**: Call the inference service health endpoint before and after destroying the named volume to demonstrate how model availability changes. Explain the difference between named volumes and bind mounts in Docker.

## Generate the Submission Report

Complete the lab at a committed repository revision, then save the raw terminal and HTTP evidence while Docker is running. Use plain-text or JSON files with these contents:

- the training image build/run output, including `Training complete` and `Model saved`;
- the inference image build/service output, including the inference service startup;
- raw JSON from `docker volume inspect wine_model_storage`;
- one JSON prediction response and the corresponding host-side `./logs/predictions.log`;
- raw JSON health responses from before and after `docker compose down -v`.

For example, use `tee` on the build, run, Compose, and `curl` commands as you complete the steps below, and copy the host-side prediction log into your evidence directory. Do not paste credentials into any evidence file. Then run this command from the repository root, replacing the example paths:

```shell
python3 scripts/generate-docker-submission.py \
  --learner "Your name" \
  --repository-url "https://github.com/your-account/lab-docker" \
  --training-output evidence/training-build-run.txt \
  --service-output evidence/inference-build-service.txt \
  --volume-inspection evidence/volume-inspect.json \
  --prediction-response evidence/prediction.json \
  --prediction-log evidence/predictions.log \
  --health-before evidence/health-before.json \
  --health-after evidence/health-after.json
```

Open `submission/docker-report.html` and correct every item marked `missing` before uploading it to Canvas. Keep `submission/docker-manifest.json` with the raw evidence. Submit a link to the exact commit shown in the report, not merely a branch URL.

The command reads local files and Git metadata only. It does not build an image, start a container, contact GitHub, or call the service. It checks that the required source files are committed and unchanged, checks the expected Dockerfile and Compose structure, and matches the saved volume, prediction/log, and before/after health evidence. Answer the persistence and reproducibility interpretation questions separately in Canvas; the checker does not decide whether those answers are correct.

## Step 0: Open the Lab Environment

### Recommended: GitHub Codespaces

1. Select **Open in GitHub Codespaces** above and create a Codespace from the course starter.
2. Wait for the `postCreateCommand` to finish. The repository DevContainer starts a private Docker daemon, installs Docker Compose, and forwards port 8081.
3. In a terminal at the repository root, rerun the disposable readiness check:

```bash
bash scripts/preflight.sh
```

The check must end with `Docker preflight: PASS`. It verifies the daemon, Compose v2, named-volume persistence, a host bind mount, and port 8081 without retaining its test volume or file.

In this lab, **host** means the machine that runs the lab's Docker daemon. In Codespaces that is the hosted Codespace/DevContainer environment, not your physical laptop. Therefore:

- `wine_model_storage:/app/models` is a named volume managed under the Codespace Docker daemon's storage;
- `./logs:/app/logs` is a bind mount of the repository's `logs/` directory in the Codespace workspace; and
- a file written to `/app/logs` in the container appears at `./logs` in the Codespace terminal and file explorer.

These are still different persistence mechanisms even though both live in the hosted environment. Do not describe a Codespace path or metric as coming from your physical laptop.

Keep forwarded port 8081 **Private** in the Codespaces **Ports** tab. The terminal `curl` commands below use `localhost:8081`; the Ports tab also provides a private browser URL for the service.

### Narrow fallback: native local Docker

Use native local Docker only if your GitHub account cannot create a Codespace or the hosted Docker daemon does not pass the preflight. Install [Docker Desktop or Docker Engine](https://docs.docker.com/get-docker/), clone the repository onto that host, and run `bash scripts/preflight.sh` in a native host terminal. Do not open this fallback inside a DevContainer that lacks its own Docker daemon or a host-socket mount: that would make `$(pwd)` refer to a path the Docker host might not share and obscure the bind-mount exercise.

Whichever route you use, stop here until the preflight passes.

## Step 1: Containerize the Training Pipeline

Create a Docker container for the training code. When launched, the container should train a Wine classifier and save the model file to a shared volume. You can use the partially completed Dockerfile and code in `docker/training/`.

### 1a. Complete `train.py`

Create a `RandomForestClassifier` and train it on the Wine dataset

- Save the trained model using `joblib.dump()` to `/app/models/wine_model.pkl`

### 1b. Complete the Training Dockerfile

Open `docker/training/Dockerfile`. It is nearly complete. Fill in the TODO to set the command that runs the training script.

### 1c. Build and Run with a Named Volume

Build the training image and run it, mounting a **named volume** for model storage:

```bash
mkdir -p ./evidence
docker build -t mlip-training -f docker/training/Dockerfile . \
  2>&1 | tee evidence/training-build-run.txt
docker run --rm -v wine_model_storage:/app/models mlip-training \
  2>&1 | tee -a evidence/training-build-run.txt
```

You should see output showing the test accuracy and a message that the model was saved.

**What is a named volume?** When you use `-v wine_model_storage:/app/models`, Docker creates a named volume called `wine_model_storage` that is managed by Docker. The data in this volume persists even after the container exits.

## Step 2: Containerize the Inference Server

Create a Docker container that loads the trained model from the shared volume and serves predictions via a Flask API. The server also logs predictions to a bind-mounted directory so you can inspect them from the host.

### 2a. Complete `server.py`

Open `docker/inference/server.py`. You need to:

Load the trained model from the shared volume when the model file exists, extract features from the incoming JSON request, run inference, and log each prediction to a host-mounted log file (`/app/logs/predictions.log`). If the model file does not exist, leave `model` as `None` so the server can still start and the health endpoint can report the missing model.

The server includes a `/health` endpoint that reports whether the model file exists — this is useful for debugging volume issues.

### 2b. Create the Inference Dockerfile

Create a new file `docker/inference/Dockerfile` from scratch. It should:

- Use `python:3.11-slim` as the base image
- Set the working directory to `/app`
- Copy and install dependencies from a requirements file
- Copy `server.py` to the working directory
- Expose the container's port 8080
- Set the command to run `server.py`

**HINT**: Look at the training Dockerfile for reference. Note that the build context is the project root, so paths should be `docker/inference/...`.

You will also need to create `docker/inference/requirements.txt` with the necessary packages (Flask, scikit-learn, joblib, numpy).

### 2c. Build and Run with Both Volume Types

Create a local directory for logs, then run the inference container with both a named volume (for the model) and a bind mount (for logs):

```bash
mkdir -p ./logs
docker build -t mlip-inference -f docker/inference/Dockerfile .
docker run --rm -p 8081:8080 \
  -v wine_model_storage:/app/models \
  -v "$(pwd)/logs:/app/logs" \
  mlip-inference
```

Leave that terminal running and use a second terminal for Step 2d. Press **Ctrl+C** in the server terminal when the checks are complete.

Notice the two `-v` flags:

- `wine_model_storage:/app/models` — **named volume** (Docker-managed, shared with training)
- `$(pwd)/logs:/app/logs` — **bind mount** (maps your local `./logs/` directory into the container)

### 2d. Test the Inference Server

Check the health endpoint:

```bash
curl http://localhost:8081/health
```

Send a prediction request (13 Wine features):

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"input": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]}' \
  | tee evidence/prediction.json
```

Test error handling with a bad request:

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"bad_key": [1,2,3]}'
```

After sending predictions, check your local `./logs/` directory — you should see a `predictions.log` file with timestamped entries. This is the bind mount in action: the container writes to `/app/logs/` and the file appears on your host filesystem.
Copy it to `evidence/predictions.log` after the successful prediction so the report generator reads the same log state you inspected.

```bash
cp ./logs/predictions.log evidence/predictions.log
```

## Step 3: Docker Compose

Docker Compose allows you to define and manage multi-container applications without long command-line parameters. Complete the `docker-compose.yml` file to set up both services.

You need to fill in:

- Build context and Dockerfile path for each service
- **Named volume** `wine_model_storage` mounted to `/app/models` on both services (for sharing the model)
- **Bind mount** `./logs` mapped to `/app/logs` on the inference service (for prediction logs)
- Port mapping for the inference service
- Named volume definition in the `volumes:` section at the bottom

Give the top-level volume the explicit name `wine_model_storage`, as well as using that key in both service mounts. This keeps the standalone `docker run` commands, Compose, `docker volume inspect wine_model_storage`, and volume-removal exercise focused on the same Docker-managed volume.

Then run Compose in the background so you can issue the evidence commands in the same terminal:

```bash
mkdir -p ./logs ./evidence
docker compose up --build -d \
  2>&1 | tee evidence/inference-build-service.txt
docker compose logs training inference \
  2>&1 | tee -a evidence/inference-build-service.txt
```

After both services start, test with the same curl commands from Step 2d. Verify that:

- The prediction endpoint returns a valid wine class
- The `./logs/predictions.log` file on your host is being written to

Shut it down:

```bash
docker compose down
```

## Step 4: Volume Lifecycle

This step demonstrates the differences in how Docker manages data persistence and host-container sharing.

### 4a. Inspect the Named Volume

Check where Docker physically stores your model on the host:

```bash
docker volume ls
docker volume inspect wine_model_storage | tee evidence/volume-inspect.json
```

Observe the `Mountpoint` field. This is a Docker-managed path on the lab's Docker host: the Codespace/DevContainer environment on the recommended route, or your local machine on the fallback. It is not the bind-mounted repository `./logs` directory.

### 4b. Persistence Verification

Verify that the trained model persists across training container destruction:

Start both containers to run training and inference:

```bash
docker compose up --build -d
until curl --silent --output /dev/null http://localhost:8081/health; do sleep 1; done
curl --silent --show-error http://localhost:8081/health | tee evidence/health-before.json
```

The response must report `"status":"healthy"` and `"model_loaded":true`.

Stop and remove containers while retaining the named volume:

```bash
docker compose down
```

Restart only the inference container:

```bash
docker compose up -d inference --no-deps
```

Run the health check again and confirm it still returns healthy, demonstrating that the model was successfully persisted in the named volume.

### 4c. Removing Volumes

To fully reset the environment and delete the model:

```bash
docker compose down -v
```

The `-v` flag deletes the named volume. Start inference without training so Compose creates a new, empty named volume, then save the changed health response:

```bash
docker compose up -d inference --no-deps
until curl --silent --output /dev/null http://localhost:8081/health; do sleep 1; done
curl --silent --show-error http://localhost:8081/health | tee evidence/health-after.json
test -f ./logs/predictions.log && tail ./logs/predictions.log
docker compose down -v
```

The response must report `"status":"model not found"` and `"model_loaded":false`. The host-side `./logs/predictions.log` must still exist because `docker compose down -v` removes the named volume, not the bind-mounted host file.

## Additional Resources

1. [Docker For Beginners](https://docker-curriculum.com/)
2. [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
3. [Docker Bind Mounts Documentation](https://docs.docker.com/storage/bind-mounts/)

## Troubleshooting

If you encounter issues:

- Rerun `bash scripts/preflight.sh` before debugging learner code
- In Codespaces, wait for creation to finish or use **Codespaces: Rebuild Container** if the Docker daemon or Compose is missing
- Verify port availability (is port 8081 already in use?)
- Review service logs with `docker compose logs`
- Use `docker compose ps -a` to confirm training completed successfully before inference started
- Use `docker compose exec` to inspect container file systems
- If the model file is missing, check that `wine_model_storage` is explicitly named and mounted by both services with `docker volume inspect wine_model_storage`
- If logs are not appearing on the host, verify your bind mount path
- If the Codespaces browser URL does not open, keep port 8081 private, verify it in the Ports tab, and first test `curl http://localhost:8081/health` in the terminal
