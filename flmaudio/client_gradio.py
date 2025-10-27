# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse  # 解析命令行参数
import asyncio  # 异步协程支持
from pathlib import Path  # 用于处理文件路径
from typing import Literal, cast  # 类型注解辅助

import os  # 文件系统读写
import numpy as np  # 数组计算
import requests  # 发送 HTTP 请求更新服务端设置
import sphn  # Opus 编解码
from numpy.typing import NDArray  # ndarray 的类型别名

try:
    import gradio as gr  # 构建 WebUI 的库
    from websockets.asyncio.client import connect  # WebSocket 客户端
    from fastrtc import AdditionalOutputs, AsyncStreamHandler, WebRTC, wait_for_item  # FastRTC 的核心组件
except ImportError:
    raise ImportError("Please install fastrtc>=0.0.21 to run this script.")

# See https://fastrtc.org/deployment/ for instructions on how to set the rtc_configuration
# variable for deployment on cloud platforms like Heroku, Spaces, etc.
"""
Usage:
python client_gradio.py --url=http://10.3.3.117:8990 --time-limit 300 --use-turn-server --port 50000
"""


rtc_configuration = None  # WebRTC 的 ICE 配置，默认不使用 TURN 服务器

# ---------------- Visit count persistence ------------
VISIT_COUNT_FILE = "flmaudio/visit_count.txt"  # 访问量统计文件路径


def load_visit_count():
    """读取累计访客数量，读取失败时返回 0。"""

    if os.path.exists(VISIT_COUNT_FILE):
        try:
            with open(VISIT_COUNT_FILE, "r") as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def save_visit_count(count: int):
    """持久化存储最新访客数量。"""

    with open(VISIT_COUNT_FILE, "w") as f:
        f.write(str(count))
# ---------------------------------------------------

class EchoHandler(AsyncStreamHandler):
    """FastRTC 的流式处理器：负责音频与文本的双向传输。"""

    def __init__(
        self,
        url: str,
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int = 480,
    ) -> None:
        self.url = url  # 用户输入的服务器地址
        proto, without_proto = self.url.split("://", 1)  # 拆分协议与域名
        if proto in ["ws", "http"]:
            proto = "ws"  # 将 http/ ws 统一转换为 ws
        elif proto in ["wss", "https"]:
            proto = "wss"  # 将 https/wss 统一成 wss

        self.output_chunk_size = 1920  # 每次发送给浏览器的音频帧长度
        self.ws = None  # WebSocket 连接对象占位
        self.ws_url = f"{proto}://{without_proto}/api/chat"  # 真实的 WebSocket API 地址
        self.stream_reader = sphn.OpusStreamReader(output_sample_rate)  # 解码服务器返回的音频
        self.stream_writer = sphn.OpusStreamWriter(output_sample_rate)  # 编码浏览器麦克风输入
        self.output_queue = asyncio.Queue()  # 音频/文本输出队列
        self.output_buffer = None  # 累积音频数据，确保输出分片长度一致

        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=24000,
        )

    async def receive(self, frame: tuple[int, NDArray]) -> None:
        """接收来自浏览器的 PCM 音频，将其编码并发送到服务器。"""

        if not self.ws:
            return
        _, array = frame
        array = array.squeeze().astype(np.float32) / 32768.0
        self.stream_writer.append_pcm(array)
        bytes = b"\x01" + self.stream_writer.read_bytes()
        await self.ws.send(bytes)

    async def emit(self) -> tuple[int, NDArray] | AdditionalOutputs | None:
        """从输出队列取出一条消息返回给浏览器。"""
        return await wait_for_item(self.output_queue)

    def copy(self) -> AsyncStreamHandler:
        """按照 FastRTC 要求，返回同配置的新处理器副本。"""
        return EchoHandler(
            self.url,
            self.expected_layout,  # type: ignore
            self.output_sample_rate,
            self.output_frame_size,
        )

    async def start_up(self):
        """在 WebRTC 会话开始时建立到服务器的 WebSocket 连接并处理返回数据。"""

        self.ws = await connect(self.ws_url, proxy=None)
        async for message in self.ws:
            if len(message) == 0:
                continue
            kind = message[0]
            if kind == 1: # audio
                payload = message[1:]
                self.stream_reader.append_bytes(payload)
                pcm = self.stream_reader.read_pcm()
                if self.output_buffer is None:
                    self.output_buffer = pcm
                else:
                    self.output_buffer = np.concatenate((self.output_buffer, pcm))
                while self.output_buffer.shape[-1] >= self.output_chunk_size:
                    self.output_queue.put_nowait((
                        self.output_sample_rate,
                        self.output_buffer[: self.output_chunk_size].reshape(1, -1),
                    ))
                    self.output_buffer = np.array(
                        self.output_buffer[self.output_chunk_size :]
                    )
            elif kind == 2: # text
                payload = cast(bytes, message[1:])
                payload = payload.decode()
                # print(payload, flush=True, end='')
                self.output_queue.put_nowait(AdditionalOutputs(payload))

    async def shutdown(self) -> None:
        """在会话结束时清理 WebSocket 连接。"""
        if self.ws:
            await self.ws.close()
            self.ws = None


def main():
    parser = argparse.ArgumentParser("client_gradio")  # 构建命令行解析器
    parser.add_argument("--url", type=str, help="URL to flmaudio server.")  # 服务端地址
    parser.add_argument(
        "--use-turn-server", action="store_true", help="whether to use turn server."
    )
    parser.add_argument(
        "--turn-url", type=str, default="turn:115.190.107.191:3478", help="TURN server URL"
    )
    parser.add_argument(
        "--turn-username", type=str, default="cofe", help="TURN server username"
    )
    parser.add_argument(
        "--turn-credential", type=str, default="EAw0rZ1GW0bQUj6m", help="TURN server credential"
    )
    parser.add_argument(
        "--time-limit", type=int, default=180, help="conversation time limit"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="serving host"
    )
    parser.add_argument(
        "--port", type=int, default=50000, help="serving port"
    )
    parser.add_argument(
        "--ssl_certfile", type=str, default='./ssl/public.crt', help="ssl certfile path"
    )
    parser.add_argument(
        "--ssl_keyfile", type=str, default='./ssl/privatekey.pem', help="ssl keyfile path"
    )
    args = parser.parse_args()  # 解析命令行参数

    global rtc_configuration
    if args.use_turn_server:
        rtc_configuration = {
            "iceServers": [
                {
                    "urls": args.turn_url,
                    "username": args.turn_username,
                    "credential": args.turn_credential,
                }
            ]
        }

    http_url = args.url  # Gradio 端访问的 HTTP 地址
    if http_url.startswith("ws"):
        http_url = http_url.replace("ws", "http", 1)

    with gr.Blocks() as demo:
        gr.HTML(
            f"""
        <div style='text-align: center'>
            <h1>
                Talk To FLM-Audio (Powered by FastRTC ⚡️)
            </h1>
            <p>
                Each conversation is limited to {args.time_limit} seconds. Once the time limit is up you can rejoin the conversation.
            </p>
        </div>
        """
        )

        # ---------------- Display visit statistics ------------
        visit_counter = gr.Markdown(value="", elem_id="visit-counter")  # 展示访客统计

        def update_visit_count():
            """页面加载时增加访问量并返回最新统计文案。"""

            count = load_visit_count() + 1
            save_visit_count(count)
            return f"👥 Total visitors: **{count}**"

        demo.load(fn=update_visit_count, inputs=None, outputs=visit_counter)
        # ---------------------------------------------------

        webrtc = WebRTC(
            label="Audio Chat",
            modality="audio",
            mode="send-receive",
            rtc_configuration=rtc_configuration,
        )

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                use_sampling = gr.Checkbox(label="Use Sampling", value=True)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Audio Settings")
                    temp_audio = gr.Slider(
                        minimum=0, maximum=2, value=0.1, label="Temperature"
                    )
                    top_k_audio = gr.Slider(
                        minimum=0, maximum=1000, step=1, value=0, label="Top K"
                    )
                    top_p_audio = gr.Slider(
                        minimum=0, maximum=1, value=0.9, label="Top P"
                    )
                    repetition_penalty_audio = gr.Slider(
                        minimum=0, maximum=2, value=1.2, label="Repetition Penalty"
                    )
                    repetition_penalty_window_audio = gr.Slider(
                        minimum=-1, maximum=100, step=1, value=50, label="Repetition Penalty Window"
                    )

                with gr.Column():
                    gr.Markdown("### Text Settings")
                    temp_text = gr.Slider(
                        minimum=0, maximum=2, value=0.01, label="Temperature"
                    )
                    top_k_text = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0, label="Top K"
                    )
                    top_p_text = gr.Slider(
                        minimum=0, maximum=1, value=1.0, label="Top P"
                    )
                    repetition_penalty_text = gr.Slider(
                        minimum=0, maximum=2, value=1.1, label="Repetition Penalty"
                    )
                    repetition_penalty_window_text = gr.Slider(
                        minimum=-1, maximum=100, step=1, value=50, label="Repetition Penalty Window"
                    )

        def update_settings(*args):
            """收集所有控件的值并组成字典。"""
            return {
                "use_sampling": args[0],
                "temp_audio": args[1],
                "temp_text": args[2],
                "top_k_audio": args[3],
                "top_k_text": args[4],
                "top_p_audio": args[5],
                "top_p_text": args[6],
                "repetition_penalty_audio": args[7],
                "repetition_penalty_text": args[8],
                "repetition_penalty_window_audio": args[9],
                "repetition_penalty_window_text": args[10],
            }

        def on_settings_change(settings):
            """在参数变化时调用后端接口同步设置。"""
            try:
                response = requests.post(f"{http_url}/api/settings", json=settings)
                if response.status_code == 200:
                    print("设置已更新为:", settings)
                else:
                    print(f"设置更新失败：{response.text}")
            except Exception as e:
                print(f"设置更新出错：{str(e)}")
            return None

        settings = gr.State({})  # 存储最新的设置字典，方便触发 change 回调

        settings.change(
            fn=on_settings_change,
            inputs=[settings],
            outputs=None,
        )

        all_inputs = [
            use_sampling,
            temp_audio,
            temp_text,
            top_k_audio,
            top_k_text,
            top_p_audio,
            top_p_text,
            repetition_penalty_audio,
            repetition_penalty_text,
            repetition_penalty_window_audio,
            repetition_penalty_window_text,
        ]

        demo.load(
            fn=update_settings,
            inputs=all_inputs,
            outputs=settings,
        )

        for input_component in all_inputs:
            input_component.change(
                fn=update_settings,
                inputs=all_inputs,
                outputs=settings,
            )

        webrtc.stream(
            EchoHandler(args.url),
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=args.time_limit,
        )


        if Path(args.ssl_certfile).exists() and Path(args.ssl_keyfile).exists():
            demo.launch(
                server_name=args.host,
                server_port=args.port,
                ssl_certfile=args.ssl_certfile,
                ssl_keyfile=args.ssl_keyfile,
                ssl_verify=False,
            )
        else:
            demo.launch(
                server_name=args.host,
                server_port=args.port,
            )


if __name__ == "__main__":
    main()
