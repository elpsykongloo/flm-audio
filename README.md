# FLM-Audio

[[Hugging Face]](https://huggingface.co/CofeAI/flm-audio)

FLM-Audio is a audio-language subversion of [RoboEgo/FLM-Ego](https://arxiv.org/abs/2506.01934v1) -- an omnimodal model with native full duplexity. It simultaneously listens, speaks, and composes internal monologue, delivering low‑latency, duplex conversational responses in both English and Chinese. FLM‑Audio is robust to noise and user interruptions, prioritizing responsiveness and naturalness.

## Model Card

- **Language(s):** Chinese; English;

## Technical Report
Motivation & Survey: [Toward Embodied AGI: A Review of Embodied AI and the Road Ahead](https://arxiv.org/abs/2505.14235)

System Card: [RoboEgo System Card: An Omnimodal Model with Native Full Duplexity](https://arxiv.org/abs/2506.01934v1)


## Bias, Risks, and Limitations

Despite extensive data cleaning, FLM‑Audio may still produce undesired content (e.g., biased or offensive language). Users should not disseminate unsafe outputs. Project authors are not responsible for misuse or harmful consequences.


## Quick Start

### Run the Server

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

## License
FLM-Audio is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), except for python code under `third_party/moshi`, which is licensed under the [MIT License](https://opensource.org/license/mit/).
This project is intended for research use only in compliance with applicable laws. For commercial use, please contact us.