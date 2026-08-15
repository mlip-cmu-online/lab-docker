# Lab 5: Containerizing ML Models with Docker

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


## Step 0: Setup Docker

Install Docker on your system and verify your installation:

```bash
docker run hello-world
```

Follow the instructions for your operating system: https://docs.docker.com/get-docker/


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
docker build -t mlip-training -f docker/training/Dockerfile .
docker run --rm -v wine_model_storage:/app/models mlip-training
```

You should see output showing the test accuracy and a message that the model was saved.

**What is a named volume?** When you use `-v wine_model_storage:/app/models`, Docker creates a named volume called `wine_model_storage` that is managed by Docker. The data in this volume persists even after the container exits.


## Step 2: Containerize the Inference Server

Create a Docker container that loads the trained model from the shared volume and serves predictions via a Flask API. The server also logs predictions to a bind-mounted directory so you can inspect them from the host.

### 2a. Complete `server.py`

Open `docker/inference/server.py`. You need to:

Load the trained model from the shared volume, extract features from the incoming JSON request, run inference, and log each prediction to a host-mounted log file (`/app/logs/predictions.log`)

The server includes a `/health` endpoint that reports whether the model file exists — this is useful for debugging volume issues.

### 2b. Create the Inference Dockerfile

Create a new file `docker/inference/Dockerfile` from scratch. It should:

- Use `python:3.11-slim` as the base image
- Set the working directory to `/app`
- Copy and install dependencies from a requirements file
- Copy `server.py` to the working directory
- Expose port 8081
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
  -v $(pwd)/logs:/app/logs \
  mlip-inference
```

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
  -d '{"input": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]}'
```

Test error handling with a bad request:

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"bad_key": [1,2,3]}'
```

After sending predictions, check your local `./logs/` directory — you should see a `predictions.log` file with timestamped entries. This is the bind mount in action: the container writes to `/app/logs/` and the file appears on your host filesystem.


## Step 3: Docker Compose

Docker Compose allows you to define and manage multi-container applications without long command-line parameters. Complete the `docker-compose.yml` file to set up both services.

You need to fill in:
- Build context and Dockerfile path for each service
- **Named volume** `wine_model_storage` mounted to `/app/models` on both services (for sharing the model)
- **Bind mount** `./logs` mapped to `/app/logs` on the inference service (for prediction logs)
- Port mapping for the inference service
- Named volume definition in the `volumes:` section at the bottom

Then run:

```bash
docker compose up --build
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
docker volume inspect wine_model_storage
```
Observe the Mountpoint field; this is the Docker-managed path on your machine.


### 4b. Persistence Verification
Verify that the trained model persists across training container destruction:

Start both containers to run training and inference:
```docker compose up --build```
(Confirm health: curl http://localhost:8081/health)

Stop and remove containers while retaining the named volume:
```docker compose down```

Restart only the inference container:
```docker compose up inference --no-deps```

Run the health check again and confirm it still returns healthy, demonstrating that the model was successfully persisted in the named volume.

### 4c. Removing Volumes
To fully reset the environment and delete the model:

```bash
docker compose down -v
```
The `-v` flag deletes the named volume.
Start the inference container again, repeat the health check, and record what changed and why.


## Additional Resources

1. [Docker For Beginners](https://docker-curriculum.com/)
2. [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
3. [Docker Bind Mounts Documentation](https://docs.docker.com/storage/bind-mounts/)

## Troubleshooting
If you encounter issues:
- Check Docker daemon status
- Verify port availability (is port 8081 already in use?)
- Review service logs with `docker compose logs`
- Ensure the training service completes before the inference service starts
- Use `docker compose exec` to inspect container file systems
- If the model file is missing, check that your named volume is correctly mounted with `docker volume ls`
- If logs are not appearing on the host, verify your bind mount path
