# Docker Basics Reference

## Containers vs Images

### What Is a Container?

**Container**: Lightweight, isolated environment that runs an application with all its dependencies.

**Like a virtual machine, but**:
- Shares host OS kernel (more efficient)
- Starts in seconds (not minutes)
- Uses less disk space and memory
- Same application behavior across different machines

**Analogy**: Shipping container for software
- Package everything needed to run
- Works the same everywhere (dev laptop, production server)
- Isolated from other containers

### What Is an Image?

**Image**: Blueprint/template for creating containers.

**Contains**:
- Base operating system files (Ubuntu, Alpine, etc.)
- Application code
- Dependencies (Python packages, libraries)
- Configuration files
- Commands to run

**Relationship**: Image → Container
- Image is the recipe
- Container is the meal cooked from that recipe
- One image can create many containers

### Key Difference

|  | Image | Container |
|---|---|---|
| **What** | Blueprint | Running instance |
| **Analogy** | Class | Object |
| **File** | Dockerfile → image.tar | Ephemeral (deleted on stop) |
| **State** | Static, immutable | Dynamic, running process |

**Example**:
```bash
# Build image from Dockerfile
docker build -t my-app .

# Create and run container from image
docker run my-app

# Can create multiple containers from same image
docker run my-app  # Container 1
docker run my-app  # Container 2
```

## Why Containerization Matters

### Problem: "Works on My Machine"

**Scenario**: Code works on your laptop but fails in production.

**Causes**:
- Different Python versions (3.9 vs 3.11)
- Missing dependencies (forgot to document library)
- Different OS (Windows dev, Linux prod)
- Environment variables not set

### Solution: Containerization

**Benefits**:

1. **Reproducibility**: Same container runs identically everywhere
2. **Isolation**: Dependencies don't conflict with host or other containers
3. **Portability**: Move containers between dev/test/prod seamlessly
4. **Consistency**: Team members use exact same environment

### In This Project

**What we containerize**:
- Python 3.11 runtime
- PyTorch and FastAPI dependencies
- Application code (backend + frontend)
- Trained model (models/latest_model.pth)

**Result**: Anyone can run `docker run` and get a working ML system - no manual setup.

## Dockerfile Commands

### FROM (Base Image)

**Purpose**: Specify starting point for your image.

```dockerfile
FROM python:3.11-slim
```

**Meaning**: Start with official Python 3.11 image (slim variant = smaller size).

**Common bases**:
- `python:3.11-slim` - Python without extra packages
- `ubuntu:22.04` - Ubuntu base OS
- `alpine:latest` - Minimal Linux (very small)

**Why needed**: Don't build OS from scratch - start with tested base.

### WORKDIR (Set Working Directory)

**Purpose**: Set current directory for subsequent commands.

```dockerfile
WORKDIR /app
```

**Meaning**: All future commands run in `/app` directory. Creates directory if doesn't exist.

**Why needed**: Organize files in predictable location. Like `cd /app` but persistent.

### COPY (Copy Files from Host to Container)

**Purpose**: Copy files from your machine into the image.

```dockerfile
COPY requirements.txt .
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/
```

**Syntax**: `COPY <source-on-host> <destination-in-container>`

**Why needed**: Container needs your application code and dependencies list.

**Best practice**: Copy `requirements.txt` first, install dependencies, then copy code
- Reason: Docker caches layers - if code changes but requirements don't, dependencies don't reinstall

### RUN (Execute Commands During Build)

**Purpose**: Run shell commands to set up the environment.

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
```

**When it runs**: During `docker build` (image creation), not when container starts.

**Common uses**:
- Install packages: `RUN apt-get update && apt-get install -y curl`
- Install Python dependencies: `RUN pip install -r requirements.txt`
- Create directories: `RUN mkdir /data`

**Multiple commands**: Chain with `&&` to reduce layers
```dockerfile
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*
```

### EXPOSE (Document Port)

**Purpose**: Document which port the container listens on.

```dockerfile
EXPOSE 8000
```

**Meaning**: Application inside container uses port 8000 (FastAPI default).

**Important**: `EXPOSE` is documentation only - doesn't actually publish port.
- To publish: Use `-p` flag when running container

```bash
docker run -p 8000:8000 my-app
#          ^^^^^^^^^^^^
#          host:container
```

### CMD (Default Command to Run)

**Purpose**: Specify what command runs when container starts.

```dockerfile
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Syntax**: JSON array format (exec form) - preferred

**Alternative syntax**: Shell form
```dockerfile
CMD uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**When it runs**: When container starts (`docker run`), not during build.

**Can be overridden**: `docker run my-app python -m pytest` replaces CMD.

### Complete Example

```dockerfile
# Start from Python 3.11 slim image
FROM python:3.11-slim

# Set working directory to /app
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Document that app uses port 8000
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Single Container Approach

### What We're Doing

**One Dockerfile** that bundles:
- FastAPI backend (Python)
- Static frontend files (HTML/CSS/JS)
- Trained ML model (.pth file)

FastAPI serves both API endpoints and static files.

### Why Single Container?

**Simplicity**: One build, one run command - easy to understand.

**Sufficient**: Frontend is static files - no need for separate web server.

**Learning focus**: Understand containerization basics without orchestration complexity.

### What Production Might Do

**Multi-container setup** (docker-compose):
- Container 1: Backend API
- Container 2: Frontend (Nginx)
- Container 3: Database
- Container 4: Redis cache
- Container 5: Model serving (separate from API)

**Why**: Scalability (scale backend independently from frontend), specialized optimizations, team separation.

**Trade-off**: Much more complex - need container orchestration (Kubernetes, docker-compose).

### In This Project

```dockerfile
# One container has everything
FROM python:3.11-slim
WORKDIR /app

# Install backend dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy everything
COPY backend/ ./backend/
COPY frontend/ ./frontend/  # Static files served by FastAPI
COPY models/ ./models/

# Run FastAPI (serves both API and static files)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**FastAPI setup** (backend/main.py):
```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files at /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

## Environment Variables

### What Are Environment Variables?

**Environment variables**: Configuration values passed to container at runtime.

**Common uses**:
- Database URLs
- API keys
- Feature flags
- Deployment environment (dev/staging/prod)

### Using .env File

**Create `.env` file**:
```
MODEL_PATH=models/latest_model.pth
API_PORT=8000
LOG_LEVEL=INFO
```

**Load in Python** (backend/config/settings.py):
```python
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/latest_model.pth")
API_PORT = int(os.getenv("API_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

**Pass to Docker**:
```bash
# Option 1: --env-file
docker run --env-file .env my-app

# Option 2: -e flag
docker run -e MODEL_PATH=models/best_model.pth my-app
```

### Why Not Hardcode?

**Problem with hardcoded values**:
```python
MODEL_PATH = "models/latest_model.pth"  # Can't change without rebuilding image
```

**Solution with environment variables**:
```python
MODEL_PATH = os.getenv("MODEL_PATH", "models/latest_model.pth")  # Can override at runtime
```

**Benefits**:
- Different configs for dev/prod without code changes
- Secrets not in code (security)
- Easy testing with different configurations

## Building Images

### docker build Command

**Purpose**: Create image from Dockerfile.

```bash
docker build -t my-app .
```

**Flags**:
- `-t my-app`: Tag (name) the image "my-app"
- `.`: Build context (current directory) - Docker looks for Dockerfile here

**What happens**:
1. Docker reads Dockerfile
2. Executes each instruction sequentially
3. Creates image layer for each instruction
4. Final image is sum of all layers
5. Tags image with specified name

**Layer caching**: Docker caches each layer
- If instruction unchanged, reuses cached layer
- If instruction changes, rebuilds that layer and all subsequent layers
- **Optimization**: Put frequent changes (code) after infrequent changes (dependencies)

### Build Context

**Build context**: Directory Docker can access during build (specified by `.` in command).

**Files included**: Everything in build context directory (recursively).

**Problem**: Large build context slows build.

**Solution**: Use `.dockerignore`:
```
# .dockerignore file
.git/
__pycache__/
*.pyc
.pytest_cache/
.venv/
node_modules/
.planning/
```

## Running Containers

### docker run Command

**Purpose**: Create and start container from image.

```bash
docker run -p 8000:8000 my-app
```

**Flags**:
- `-p 8000:8000`: Publish port (map host port 8000 to container port 8000)
- `-d`: Detach (run in background)
- `--name my-container`: Give container a name
- `-e KEY=VALUE`: Set environment variable
- `--env-file .env`: Load variables from file
- `-v /host/path:/container/path`: Mount volume

**Example with multiple flags**:
```bash
docker run -d \
  --name financial-risk-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  my-app
```

### Port Mapping

**Syntax**: `-p <host-port>:<container-port>`

**Example**: `-p 8000:8000`
- Container listens on port 8000
- Host port 8000 forwards to container port 8000
- Access application at `http://localhost:8000`

**Different ports**: `-p 3000:8000`
- Container listens on port 8000
- Host port 3000 forwards to container port 8000
- Access application at `http://localhost:3000`

## Common Docker Commands

### List Images
```bash
docker images
```

### List Containers
```bash
docker ps        # Running only
docker ps -a     # All (including stopped)
```

### Stop Container
```bash
docker stop <container-id-or-name>
```

### Remove Container
```bash
docker rm <container-id-or-name>
```

### Remove Image
```bash
docker rmi <image-id-or-name>
```

### View Logs
```bash
docker logs <container-id-or-name>
docker logs -f <container-id-or-name>  # Follow (stream) logs
```

### Execute Command in Running Container
```bash
docker exec -it <container-id-or-name> bash  # Open shell
docker exec <container-id-or-name> python -m pytest  # Run command
```

## Dockerfile Best Practices

### 1. Use Specific Base Image Tags
```dockerfile
# Good: Specific version
FROM python:3.11-slim

# Bad: Latest (can break unexpectedly)
FROM python:latest
```

### 2. Minimize Layer Count
```dockerfile
# Good: Combined into one RUN
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Bad: Multiple layers
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*
```

### 3. Order Instructions by Change Frequency
```dockerfile
# Dependencies change infrequently - install first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code changes frequently - copy last
COPY backend/ ./backend/
```

### 4. Use .dockerignore
```
.git/
*.pyc
__pycache__/
.pytest_cache/
```

### 5. Don't Run as Root (Security)
```dockerfile
RUN useradd -m appuser
USER appuser
```

## Key Takeaways

1. **Container**: Running instance of an image (process)
2. **Image**: Blueprint for creating containers (template)
3. **Dockerfile**: Instructions to build image
4. **FROM**: Base image to start from
5. **WORKDIR**: Set working directory
6. **COPY**: Copy files from host to image
7. **RUN**: Execute commands during build
8. **EXPOSE**: Document port (doesn't publish)
9. **CMD**: Default command when container starts
10. **Environment variables**: Configuration passed at runtime
11. **Build**: `docker build -t name .`
12. **Run**: `docker run -p 8000:8000 name`
13. **Single container**: Sufficient for learning, simpler than multi-container
14. **Why containerization**: Reproducibility, isolation, portability
