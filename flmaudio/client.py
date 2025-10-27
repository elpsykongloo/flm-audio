import argparse  # 解析命令行参数，用于自定义服务器地址等
import asyncio  # 异步事件循环，驱动整体客户端流程
from pathlib import Path  # 处理输入/输出目录
import sys  # 退出程序时使用
from typing import Iterable, List

import numpy as np  # 处理音频数据的数组计算库
import soundfile as sf  # 负责读取/写入本地音频文件
import sphn  # Opus 编解码库，负责压缩与解码音频

from websockets.asyncio.client import connect  # WebSocket 客户端，用于和服务器通信
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from .utils import log  # 自定义日志输出函数


class Connection:
    """封装文件模式下的 WebSocket 连接与音频帧收发流程。"""

    def __init__(
        self,
        ws_uri: str,
        input_frames: Iterable[np.ndarray],
        sample_rate: float = 24000,
        frame_size: int = 1920,
        response_timeout: float = 3.0,
    ) -> None:
        self.ws_uri = ws_uri  # WebSocket 服务端地址
        self.sample_rate = sample_rate  # 采样率（服务器与客户端需一致）
        self.frame_size = frame_size  # 每次处理的帧长度
        self._input_frames = list(input_frames)  # 预先准备好的 PCM 帧序列
        self._response_timeout = max(response_timeout, 0.5)  # 等待服务器静默的超时阈值

        # Opus 编码器与解码器：负责上传音频的压缩与服务器返回音频的解码
        self._opus_writer = sphn.OpusStreamWriter(self.sample_rate)
        self._opus_reader = sphn.OpusStreamReader(self.sample_rate)

        self._output_pcm_chunks: List[np.ndarray] = []  # 存放服务器返回的 PCM 片段
        self._stop_event: asyncio.Event | None = None  # 控制各协程退出
        self._input_finished_event: asyncio.Event | None = None  # 标记输入帧是否已经全部写入编码器
        self._response_complete_event: asyncio.Event | None = None  # 标记服务器音频是否已经收全
        self._recv_done_event: asyncio.Event | None = None  # 标记接收协程是否结束

        self._last_decoder_activity: float = 0.0  # 记录最近一次成功解码的时间戳
        self._received_audio_messages = 0  # 统计服务器返回的音频消息数量

    async def _produce_input_frames(self) -> None:
        """逐帧将输入 PCM 写入 Opus 编码器，模拟实时推流。"""
        assert self._input_finished_event is not None
        assert self._stop_event is not None
        try:
            for idx, frame in enumerate(self._input_frames, start=1):
                if self._stop_event.is_set():
                    log("warning", f"在第 {idx} 帧写入前检测到停止信号，提前终止上传。")
                    break
                self._opus_writer.append_pcm(frame.astype(np.float32))
                await asyncio.sleep(0)  # 把控制权交回事件循环，避免阻塞其它协程
            log("info", f"全部 {len(self._input_frames)} 帧音频已写入编码器，等待发送完毕。")
        except Exception as exc:  # 捕获编码器写入阶段的所有异常
            log("error", f"推送输入音频帧时出现异常：{exc!r}")
            self._stop_event.set()
            raise
        finally:
            self._input_finished_event.set()

    async def _send_loop(self, ws) -> None:
        """循环读取编码后的音频并通过 WebSocket 发送给服务器。"""
        assert self._input_finished_event is not None
        assert self._stop_event is not None
        try:
            while True:
                if self._stop_event.is_set() and self._input_finished_event.is_set():
                    break
                await asyncio.sleep(0.001)
                msg = self._opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send(b"\x01" + msg)
                elif self._input_finished_event.is_set() and self._stop_event.is_set():
                    break
        except asyncio.CancelledError:
            # 协程被主流程主动取消时，直接退出即可
            raise
        except ConnectionClosedOK:
            # 连接已由双方协商关闭，属于正常流程
            log("info", "服务器已关闭连接，停止继续发送音频数据。")
            self._stop_event.set()
            return
        except ConnectionClosedError as exc:
            log("warning", f"发送音频时检测到非正常关闭：{exc!r}")
            self._stop_event.set()
            return
        except Exception as exc:
            log("error", f"发送音频数据时出现异常：{exc!r}")
            self._stop_event.set()
            raise

    async def _decoder_loop(self) -> None:
        """持续从解码器取出 PCM 数据并缓存在列表中，等待最终合并写入文件。"""
        assert self._input_finished_event is not None
        assert self._response_complete_event is not None
        assert self._stop_event is not None

        all_pcm_data = None  # 累积缓冲区，确保输出帧长度固定
        loop = asyncio.get_running_loop()
        self._last_decoder_activity = loop.time()

        try:
            while True:
                if self._stop_event.is_set() and self._recv_done_event and self._recv_done_event.is_set():
                    break
                await asyncio.sleep(0.001)
                pcm = self._opus_reader.read_pcm()
                if pcm is None or pcm.size == 0:
                    if (
                        self._input_finished_event.is_set()
                        and not self._response_complete_event.is_set()
                        and (loop.time() - self._last_decoder_activity) > self._response_timeout
                    ):
                        log(
                            "info",
                            "在超时时间内未再收到新的音频数据，视为服务器回复结束。",
                        )
                        self._response_complete_event.set()
                        break
                    continue

                self._last_decoder_activity = loop.time()
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = np.array(all_pcm_data[self.frame_size :])
                    self._output_pcm_chunks.append(chunk.astype(np.float32))
        except asyncio.CancelledError:
            # 收尾阶段若被取消，跳过报错，改为执行 finally 中的缓存写回
            raise
        except Exception as exc:
            log("error", f"解码服务器音频时出现异常：{exc!r}")
            self._stop_event.set()
            raise
        finally:
            if all_pcm_data is not None and all_pcm_data.size > 0:
                self._output_pcm_chunks.append(all_pcm_data.astype(np.float32))
            if not self._response_complete_event.is_set():
                self._response_complete_event.set()

    async def _recv_loop(self, ws) -> None:
        """接收服务器消息，按类型处理音频或文本。"""
        assert self._stop_event is not None
        assert self._response_complete_event is not None
        try:
            async for message in ws:
                if not isinstance(message, bytes):
                    log("warning", f"忽略非二进制消息：{type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "收到长度为 0 的消息，跳过。")
                    continue
                kind = message[0]
                if kind == 0:
                    # 目前服务端尚未定义类型 0，但部分部署会发送保活占位符
                    log("info", "收到服务器保活消息，忽略不处理。")
                    continue
                if kind == 1:  # audio
                    payload = message[1:]
                    self._opus_reader.append_bytes(payload)
                    self._received_audio_messages += 1
                elif kind == 2:  # text
                    payload = message[1:]
                    try:
                        text = payload.decode()
                    except UnicodeDecodeError as exc:
                        log("warning", f"文本消息解码失败：{exc!r}")
                        continue
                    log("info", f"服务器文本片段：{text!r}")
                else:
                    log("warning", f"未知的消息类型 {kind}，已忽略。")
        except asyncio.CancelledError:
            raise
        except ConnectionClosedOK:
            log("info", "服务器已结束 WebSocket 会话。")
        except ConnectionClosedError as exc:
            log("warning", f"WebSocket 连接异常关闭：{exc!r}")
            self._stop_event.set()
            pending_ex = exc
        except Exception as exc:
            log("error", f"接收服务器数据时出现异常：{exc!r}")
            self._stop_event.set()
            pending_ex = exc
        finally:
            if self._recv_done_event:
                self._recv_done_event.set()
            self._response_complete_event.set()
            self._stop_event.set()
            if "pending_ex" in locals():
                raise pending_ex

    def _collect_output(self) -> np.ndarray:
        """把收集到的 PCM 片段拼接成一维数组并返回。"""
        if not self._output_pcm_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._output_pcm_chunks, axis=0)

    async def run(self) -> np.ndarray:
        """并发运行推流、发送、接收、解码协程，返回服务器生成的 PCM。"""
        log("info", f"连接至 {self.ws_uri}，准备推送音频文件。")

        try:
            async with connect(self.ws_uri, proxy=None) as ws:
                self._stop_event = asyncio.Event()
                self._input_finished_event = asyncio.Event()
                self._response_complete_event = asyncio.Event()
                self._recv_done_event = asyncio.Event()

                producer_task = asyncio.create_task(self._produce_input_frames())
                send_task = asyncio.create_task(self._send_loop(ws))
                recv_task = asyncio.create_task(self._recv_loop(ws))
                decoder_task = asyncio.create_task(self._decoder_loop())

                pending_exceptions: List[BaseException] = []
                try:
                    await producer_task
                    await self._input_finished_event.wait()
                    log("info", "输入音频帧已经全部推送完毕，等待服务器回复。")

                    await self._response_complete_event.wait()
                    log(
                        "info",
                        f"服务器音频接收完成，共收到 {self._received_audio_messages} 条音频消息。",
                    )
                except Exception as exc:
                    pending_exceptions.append(exc)
                finally:
                    self._stop_event.set()

                    close_coro = getattr(ws, "close", None)
                    if close_coro is not None:
                        try:
                            await asyncio.wait_for(close_coro(), timeout=3.0)
                        except asyncio.TimeoutError:
                            log("warning", "主动关闭 WebSocket 超时，尝试强制断开。")
                            transport = getattr(ws, "transport", None)
                            if transport is not None:
                                transport.close()
                        except Exception as exc:
                            log("warning", f"主动关闭 WebSocket 失败：{exc!r}")

                    pending_tasks = [send_task, recv_task, decoder_task]
                    done, pending = await asyncio.wait(
                        pending_tasks,
                        timeout=3.0,
                        return_when=asyncio.ALL_COMPLETED,
                    )

                    if pending:
                        for task in pending:
                            task.cancel()
                        done_cancelled, still_pending = await asyncio.wait(
                            pending,
                            timeout=2.0,
                            return_when=asyncio.ALL_COMPLETED,
                        )
                        done |= done_cancelled
                        pending = still_pending

                    results: List[BaseException] = []
                    for task in done:
                        if task.cancelled():
                            continue
                        try:
                            task.result()
                        except BaseException as exc:
                            results.append(exc)

                    for task in pending:
                        # 未能在超时时间内退出的任务，记录并强制取消
                        task.cancel()
                        results.append(asyncio.CancelledError())

                    for exc in results:
                        if not isinstance(exc, asyncio.CancelledError):
                            pending_exceptions.append(exc)

                if pending_exceptions:
                    # 优先抛出第一个异常，帮助上层定位问题
                    raise pending_exceptions[0]
        except OSError as exc:
            raise ConnectionError(
                "无法建立到服务器的 WebSocket 连接，请检查主机、端口或 TLS 配置。"
            ) from exc

        log("info", "与服务器的连接已关闭。")
        return self._collect_output()


class Client:
    """负责批量读取本地音频文件、推送到服务器并保存返回结果。"""

    def __init__(self, args):
        self.ws_uri = self._get_uri(args)  # 根据参数拼接 WebSocket 地址
        self.sample_rate = 24000  # 服务端要求的采样率
        self.frame_size = 1920  # 每帧包含的采样点数
        self.input_dir = Path(args.input_dir).expanduser().resolve()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.response_timeout = args.response_timeout

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """校验输入目录必须存在，并在需要时创建输出目录。"""
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            log(
                "error",
                f"输入目录 {self.input_dir} 不存在或不是文件夹，请检查参数。",
            )
            sys.exit(1)
        if not self.output_dir.exists():
            log("info", f"输出目录 {self.output_dir} 不存在，正在创建。")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_uri(self, args) -> str:
        if args.url is None:
            proto = "ws"
            if args.https:
                proto += "s"
            return f"{proto}://{args.host}:{args.port}/api/chat"
        else:
            proto = "wss"
            if '://' in args.url:
                proto, without_proto = args.url.split('://', 1)
                if proto in ['ws', 'http']:
                    proto = "ws"
                elif proto in ['wss', 'https']:
                    proto = "wss"
                else:
                    log(
                        "error",
                        f"提供的 URL {args.url} 包含未知协议，程序终止。",
                    )
                    sys.exit(1)
            else:
                without_proto = args.url
            return f"{proto}://{without_proto}/api/chat"

    def _list_input_files(self) -> List[Path]:
        """列出输入目录下的所有 wav 文件并按名称排序。"""
        files = sorted(self.input_dir.glob("*.wav"))
        if not files:
            log(
                "warning",
                f"在 {self.input_dir} 未找到任何 .wav 文件，程序将直接退出。",
            )
        return files

    def _resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """若输入采样率与服务器要求不同，则使用线性插值做简单重采样。"""
        if orig_sr == self.sample_rate:
            return audio.astype(np.float32)

        if orig_sr <= 0:
            raise ValueError("原始采样率必须为正数。")

        duration = audio.shape[0] / float(orig_sr)
        target_length = int(round(duration * self.sample_rate))
        if target_length <= 0:
            raise ValueError("无法根据目标采样率计算输出长度。")

        old_times = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        new_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
        resampled = np.interp(new_times, old_times, audio)
        return resampled.astype(np.float32)

    def _prepare_frames(self, audio: np.ndarray) -> List[np.ndarray]:
        """按照固定帧长切分音频，末尾不足的部分做零填充。"""
        frames: List[np.ndarray] = []
        for start in range(0, audio.shape[0], self.frame_size):
            chunk = audio[start : start + self.frame_size]
            if chunk.shape[0] < self.frame_size:
                padded = np.zeros(self.frame_size, dtype=np.float32)
                padded[: chunk.shape[0]] = chunk
                chunk = padded
            frames.append(chunk.astype(np.float32))
        if not frames:
            log("warning", "检测到长度为 0 的音频，自动跳过。")
        return frames

    def _load_audio(self, path: Path) -> List[np.ndarray]:
        """读取 wav 文件为单声道 PCM 帧，返回帧列表。"""
        log("info", f"正在读取输入音频：{path}")
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"无法读取 {path}：{exc}") from exc

        if audio.ndim == 2:
            log("info", "检测到多声道音频，将自动平均为单声道。")
            audio = audio.mean(axis=1)

        audio = np.clip(audio, -1.0, 1.0)
        audio = self._resample_audio(audio, sr)

        frames = self._prepare_frames(audio)
        if not frames:
            raise RuntimeError("音频为空，无法推送到服务器。")
        log("info", f"输入音频共拆分为 {len(frames)} 帧。")
        return frames

    def _write_output(self, path: Path, audio: np.ndarray) -> None:
        """把服务器生成的 PCM 写回本地 wav 文件。"""
        if audio.size == 0:
            log("warning", f"服务器未返回音频，跳过写入 {path}。")
            return
        try:
            sf.write(path, audio, self.sample_rate)
        except Exception as exc:
            raise RuntimeError(f"写入输出文件 {path} 失败：{exc}") from exc
        log("info", f"成功写入输出音频：{path}")

    async def _process_single_file(self, path: Path) -> None:
        """读取单个文件、推送至服务器并保存返回结果。"""
        try:
            frames = self._load_audio(path)
        except Exception as exc:
            log("error", f"读取音频失败：{exc}")
            return

        connection = Connection(
            self.ws_uri,
            frames,
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            response_timeout=self.response_timeout,
        )
        try:
            output_pcm = await connection.run()
        except Exception as exc:
            log("error", f"与服务器交互时出现异常：{exc}")
            return

        output_name = f"output_{path.name}"
        output_path = self.output_dir / output_name
        try:
            self._write_output(output_path, output_pcm)
        except Exception as exc:
            log("error", f"保存输出音频失败：{exc}")

    async def run(self) -> None:
        """顺序处理输入目录下的所有 wav 文件。"""
        files = self._list_input_files()
        if not files:
            return

        for path in files:
            log("info", f"开始处理 {path.name}")
            await self._process_single_file(path)
        log("info", "所有文件均已处理完成。")


def main():
    parser = argparse.ArgumentParser("client_opus")  # 构造命令行解析器
    parser.add_argument("--host", default="localhost", type=str, help="Hostname to connect to.")  # 指定服务器主机
    parser.add_argument("--port", default=8990, type=int, help="Port to connect to.")  # 指定端口为8990
    parser.add_argument("--https", action='store_true', help="Set this flag for using a https connection.")  # 是否使用 HTTPS
    parser.add_argument("--url", type=str, help='Provides directly a URL, e.g. to a gradio tunnel.')  # 直接指定完整 URL
    parser.add_argument("--input-dir", required=True, help="包含输入 wav 文件的目录路径。")
    parser.add_argument("--output-dir", required=True, help="用于保存输出 wav 文件的目录路径。")
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=3.0,
        help="服务器在输入结束后允许的最大静默秒数，超时则认为回复完成。",
    )
    args = parser.parse_args()
    try:
        asyncio.run(Client(args).run())  # 启动客户端主流程
    except KeyboardInterrupt:
        log("warning", "Interrupting, exiting connection.")
    log("info", "All done!")


if __name__ == "__main__":
    main()
