<p align="left">
    中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp 
</p>
<br><br>

# FLM-Audio

<p align="center">
        🤗 <a href="https://huggingface.co/CofeAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/FLM">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2509.02521">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/FLM/FLM-Audio-Demo/summary">Demo</a>
</p>

FLM-Audio 是一个原生全双工语音模型，其技术来源于 [RoboEgo/FLM-Ego](https://arxiv.org/abs/2506.01934v1)，一个原生全双工（native full‑duplexity）的全模态模型（omnimodal）。FLM-Audio能够同时听、说并生成内部独白，以低延迟在中英文两种语言中提供双向对话。FLM‑Audio 对噪声与用户打断具有很好的鲁棒性，响应性与自然度均得到了很好的保证。

## 模型信息

- **支持语言：** 汉语、英语

## 技术报告

动机与综述： [Toward Embodied AGI: A Review of Embodied AI and the Road Ahead](https://arxiv.org/abs/2505.14235)

FLM-Audio 论文： [FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training](https://arxiv.org/abs/2509.02521)

RoboEgo 论文： [RoboEgo System Card: An Omnimodal Model with Native Full Duplexity](https://arxiv.org/abs/2506.01934v1)

## 偏见、风险与限制

尽管经过大量数据清洗，FLM‑Audio 仍可能产生不期望的内容（例如带有偏见或冒犯性的语言）。请勿将可能不安全的输出传播或用于有害目的。项目作者对误用或由此产生的有害后果不承担责任。

## 快速开始

### 推荐：通过 Docker 运行 Server（生产/部署首选）

推荐使用发布在 GitHub Container Registry（ghcr.io）上 `cofe-ai` 组织的官方 Docker 镜像来运行服务：

> `ghcr.io/cofe-ai/flm-audio`

镜像变体说明：

- `ghcr.io/cofe-ai/flm-audio:server-1.0.0-model-v202507` — **包含已预下载的模型**（适用于无法联网或需要快速启动的场景）。
- `ghcr.io/cofe-ai/flm-audio:server-1.0.0` — **不包含模型，容器启动后会在运行时从 Hugging Face 下载模型**（需要网络）。

启动示例（推荐在无法联网或希望避免首次下载延迟时使用内置模型镜像）：

```bash
# 使用内置已下载模型的镜像（推荐：无需从 Hugging Face 下载）
docker run -dit --gpus '"device=1"' -p 8990:8990 --restart always --name flm-audio-server ghcr.io/cofe-ai/flm-audio:server-1.0.0-model-v202507

# 或者：使用会在运行时从 Hugging Face 自动下载模型的镜像（需要网络）
docker run -dit --gpus '"device=1"' -p 8990:8990 --restart always --name flm-audio-server ghcr.io/cofe-ai/flm-audio:server-1.0.0
```

**说明**：
- `--gpus '"device=1"'`：示例中指定使用编号为 `1` 的 GPU。请根据实际机器调整（例如 `--gpus all` 或 `--gpus '"device=0,1"'`）。
- 端口 `8990` 是 Server 的默认端口；如需更改对外端口，可调整为 `-p 主机端口:8990`。
- 若镜像托管为私有仓库，可能需要先执行 `docker login ghcr.io` 并使用 GitHub Personal Access Token（PAT）进行认证，具体取决于仓库访问权限。
- 使用不带预置模型的镜像（`server-1.0.0`）时，容器首次启动会联网下载模型，下载时间取决于网络和模型大小；使用带预置模型的镜像则无需网络即可启动。

请注意，首次启动容器的时候，为了加速推理，模型加载后需要编译一段时间（约2分钟，由服务器性能决定），当看到日志信息包含如下内容时，表明已经完全启动成功：
```
[Info] model loaded
[Info] warming up the model
[Info] Access the API directly at http://0.0.0.0:8990/api/chat
======== Running on http://0.0.0.0:8990 ========
(Press CTRL+C to quit)
```

### 本地启动服务（可选）

若用于本地开发或调试，可直接以 Python 方式运行：

```bash
# 安装依赖
pip install -r requirements-server.txt
# 启动 server
python -m flmaudio.server --port 8990
```

> 注意：本地启动同样需要模型文件存在；若未提前下载，程序将尝试从 Hugging Face 获取对应模型权重。


### 启动 Web UI（连接到已运行的 Server）

```bash
# 启动 Web UI（Gradio），连接到本地或远程 server
pip install -r requirements-clientgui.txt
python -m flmaudio.client_gradio --url http://localhost:8990
```

然后就可以在浏览器打开 http://localhost:50000 进行体验。

### 启动 CLI（连接到已运行的 Server）

```bash
# 启动 CLI 客户端
pip install -r requirements-clientcli.txt
python -m flmaudio.client --url http://localhost:8990
```

**说明**：
- 无论是Web UI 还是 CLI方式，均需要将 url 替换为你的服务所在服务器的ip和端口，记得防火墙放行；
- 使用 Web UI 时，由于gradio和现代浏览器的安全措施，建议你在调试时候，执行python命令的机器和浏览器在同一台机器上，这样可以在浏览器上使用localhost

## 推荐运行环境

- **操作系统：** Linux（推荐）。
- **GPU：** NVIDIA GPU，**建议显存不少于 20 GB**，以保证大模型推理的稳定性与性能。
- **软件：** Docker、NVIDIA Container Toolkit（用于容器内 GPU 支持，亦称 `nvidia-docker`），以及匹配的 NVIDIA 驱动程序。
- **存储：** 为模型文件与日志预留充足磁盘空间（模型文件通常需要 16GB）。
- **网络：** 仅在使用不含模型的镜像或选择在线下载模型时需要；使用包含模型的镜像则无需联网即可启动。

## 常见问题（简要）

- **我应该选择哪个镜像？**
  - 如果服务器可以访问互联网且不介意首次下载：可使用 `server-1.0.0`。
  - 如果服务器无法联网，或希望开箱即用、快速启动：请使用 `server-1.0.0-model-v202507`（已预置模型）。

- **如何指定不同的 GPU？**
  - 调整 `--gpus` 参数，例如 `--gpus '"device=0"'` 或 `--gpus all`（具体支持和语法取决于主机上的 Docker 与 NVIDIA 容器工具配置）。

## 致谢

本工作受到新一代人工智能国家科技重大专项 (No. 2022ZD0116314)的支持，特此感谢。

## 引用

如果你觉得我们的工作对你有帮助，欢迎引用！

```
@article{flm-audio,
  title={Flm-audio: Natural monologues improves native full-duplex chatbots via dual training},
  author={Yao, Yiqun and Li, Xiang and Jiang, Xin and Fang, Xuezhi and Yu, Naitong and Wenjia, Ma and Sun, Aixin and Wang, Yequan},
  journal={arXiv preprint arXiv:2509.02521},
  year={2025}
}
@article{embodied-agi,
  title={Toward embodied agi: A review of embodied ai and the road ahead},
  author={Wang, Yequan and Sun, Aixin},
  journal={arXiv preprint arXiv:2505.14235},
  year={2025}
}
@article{roboego,
  title={RoboEgo System Card: An Omnimodal Model with Native Full Duplexity},
  author={Yao, Yiqun and Li, Xiang and Jiang, Xin and Fang, Xuezhi and Yu, Naitong and Sun, Aixin and Wang, Yequan},
  journal={arXiv preprint arXiv:2506.01934},
  year={2025}
}
```

## 使用协议

FLM-Audio 使用 Apache License 2.0 授权，`third_party/moshi` 目录下的部分 Python 代码采用 MIT 许可，本模型默认的音色版权由原音色持有人保留。本项目仅供研究用途，须遵守适用法律；如需商业用途，请联系我们。
