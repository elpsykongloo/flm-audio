# FLM-Audio

[[Hugging Face]](https://huggingface.co/CofeAI/flm-audio)

FLM-Audio is a audio-language subversion of [RoboEgo/FLM-Ego](https://arxiv.org/abs/2506.01934v1) -- an omnimodal model with native full duplexity. It simultaneously listens, speaks, and composes internal monologue, delivering low‑latency, duplex conversational responses in both English and Chinese. FLM‑Audio is robust to noise and user interruptions, prioritizing responsiveness and naturalness.

## Model Card

- **Language(s):** Chinese; English;

## Technical Report
Motivation & Survey: [Toward Embodied AGI: A Review of Embodied AI and the Road Ahead](https://arxiv.org/abs/2505.14235)

FLM-Audio Research Paper: [FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training](https://arxiv.org/abs/2509.02521)

Omnimodal System Card: [RoboEgo System Card: An Omnimodal Model with Native Full Duplexity](https://arxiv.org/abs/2506.01934v1)


## Bias, Risks, and Limitations

Despite extensive data cleaning, FLM‑Audio may still produce undesired content (e.g., biased or offensive language). Users should not disseminate unsafe outputs. Project authors are not responsible for misuse or harmful consequences.


## Quick Start

### Recommended: Run the Server via Docker (Production/Deployment)

We recommend using the official Docker images published under the `cofe-ai` organization on GitHub Container Registry:

> `ghcr.io/cofe-ai/flm-audio`

Image variants:

- `ghcr.io/cofe-ai/flm-audio:server-1.0.0-model-v202507` — **includes the pre‑downloaded model** (ideal for offline or fast startup environments).
- `ghcr.io/cofe-ai/flm-audio:server-1.0.0` — **downloads the model from Hugging Face at runtime** (requires internet access).

Example startup commands:

```bash
# Using the image with pre-downloaded model (recommended for offline/fast startup)
docker run -dit --gpus '"device=1"' -p 8990:8990 --restart always --name flm-audio-server ghcr.io/cofe-ai/flm-audio:server-1.0.0-model-v202507

# Or: using the image that downloads the model at runtime (requires network access)
docker run -dit --gpus '"device=1"' -p 8990:8990 --restart always --name flm-audio-server ghcr.io/cofe-ai/flm-audio:server-1.0.0
```

**Notes / Tips:**
- `--gpus '"device=1"'`: the example binds GPU device `1`. Adjust as needed (e.g., `--gpus all` or `--gpus '"device=0,1"'`).
- Port `8990` is the default server port; adjust host mapping if necessary with `-p HOST_PORT:8990`.
- For private images, you may need to `docker login ghcr.io` with a GitHub personal access token (PAT).
- When using the non-preloaded image (`server-1.0.0`), the container will download the model on first startup. Download time depends on your network.

### Local Development (Optional)

```bash
# install dependencies
pip install -r requirements-server.txt
python -m flmaudio.server --port 8990
```

### Start the Web UI

```bash
# install dependencies
pip install -r requirements-clientgui.txt
python -m flmaudio.client_gradio --url http://localhost:8990
```

### Start the CLI

```bash
# install dependencies
pip install -r requirements-clientcli.txt
python -m flmaudio.client --url http://localhost:8990
```

## Recommended Environment

- **OS:** Linux (preferred for production).
- **GPU:** NVIDIA GPU with **at least 20 GB VRAM** recommended for larger models and stable inference.
- **Software:** Docker, NVIDIA Container Toolkit (`nvidia-docker`/`--gpus` support), and appropriate NVIDIA driver.
- **Storage:** Ensure sufficient disk space for models and logs (model files can require multiple GBs).
- **Network:** Required only if using the image without pre-downloaded model; the preloaded image does not need internet to start.

## FAQ (Brief)

- **Which image should I choose?**
  - If your server can access the internet and you don’t mind first-run download: use `server-1.0.0`.
  - If your server cannot access the internet or you prefer fast startup: use `server-1.0.0-model-v202507`.

- **How to specify different GPUs?**
  - Adjust the `--gpus` parameter, e.g., `--gpus '"device=0"'` or `--gpus all`, depending on your host configuration.

## License
FLM-Audio is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), except for python code under `third_party/moshi`, which is licensed under the [MIT License](https://opensource.org/license/mit/).
This project is intended for research use only in compliance with applicable laws. For commercial use, please contact us.
