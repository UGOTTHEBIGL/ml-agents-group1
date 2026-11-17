# Docker setup
Versions in the docker

| Package | Version |
|---|---|
| Python | 3.10.12 |
| mlagents / mlagents-envs | 1.0.0 |
| torch | 2.8.0+cpu |
| numpy | 1.23.5 |
| protobuf | 3.20.1 |
| grpcio | 1.48.2 |

---

## 1. Install Docker Desktop

- **Windows:** from docker.com
- **Mac:** `brew install --cask docker`.

Check that it works:
```bash
docker run --rm hello-world
```

---

## 2. Build the image (once)

Everyone builds the image themselves using the Dockerfile in the repo root.
```bash
docker build --platform linux/amd64 -t mlagents-group1 .
```
first build will take quite long, after that it is faster

Check the versions:
```bash
docker run --rm mlagents-group1 pip list
```

Not all the files from the mlagents are in the docker file so the dockerfile does not need to be rebuild everytime.

Only rebuild when the Dockerfile or ml-agents/ or ml-agents-envs/ change.

---

## 3. Build the game for Linux (Unity)

The container runs Linux, so the game has to be a Linux build as well. I tried using mono and IL2CPP, both dont work for mac.

### One-time Unity setup
1. **Unity Hub → Installs → 2022.3.4f1 → ⚙ → Add modules →
   Linux Build Support (Mono)**. 
2. **Edit → Project Settings → Player →** tab with the **monitor icon** (PC, Mac & Linux Standalone)
   **→ Other Settings → Configuration → Scripting Backend: Mono**.

### Building a game
Same as normally, but make sure target platform is linux

in builds add another folder called **linux** where you save the builds.

---

## 4. Run a training

First create the folders and a config (once):
```bash
mkdir -p aiml1/builds/linux aiml1/configs aiml1/data
cp config/ppo/3DBall.yaml aiml1/configs/3DBall_ppo_conf.yaml
```

this is the same as the normal version but an extra linux folder inside builds.

Always run from the **repo root**, and use a **new run name** every time.

**Windows (PowerShell):**
```powershell
docker run --rm -v "${PWD}:/workspace" mlagents-group1 `
  python aiml1/data_collection_scripts/run_once.py aiml1/configs/3DBall_ppo_conf.yaml aiml1/builds/linux/3DBall.x86_64 run_001
```

**macOS (Intel) / Linux:**
```bash
docker run --rm -v "$PWD":/workspace mlagents-group1 \
  python aiml1/data_collection_scripts/run_once.py aiml1/configs/3DBall_ppo_conf.yaml aiml1/builds/linux/3DBall.x86_64 run_001
```


### Results
Written straight into your repo:
- `aiml1/data/runs.csv`: one row per run (created automatically on the first run)


