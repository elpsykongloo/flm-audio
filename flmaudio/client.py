# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd
import sphn

from sshkeyboard import listen_keyboard_manual as listen_keyboard_async
from websockets.asyncio.client import connect
from .utils import log


class Connection:
    def __init__(
        self,
        ws_uri: str,
        sample_rate: float = 24000,
        channels: int = 1,
        frame_size: int = 1920,
    ) -> None:
        self.ws_uri = ws_uri
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels

        self._done = False

        self._in_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=frame_size,
            callback=self._on_audio_input,
        )

        self._out_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=frame_size,
            callback=self._on_audio_output,
        )

        self._opus_writer = sphn.OpusStreamWriter(self.sample_rate)
        self._opus_reader = sphn.OpusStreamReader(self.sample_rate)
        self._output_queue = queue.Queue()


    async def _send_loop(self, ws) -> None:
        try:
            while not self._done:
                await asyncio.sleep(0.001)
                msg = self._opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send(b"\x01" + msg)
            await ws.close() # close when client is done
        except Exception as e:
            print(e)
            self._lost_connection()
            return

    async def _decoder_loop(self) -> None:
        all_pcm_data = None
        while not self._done:
            await asyncio.sleep(0.001)
            pcm = self._opus_reader.read_pcm()
            if pcm is None:
                continue
            if all_pcm_data is None:
                all_pcm_data = pcm
            else:
                all_pcm_data = np.concatenate((all_pcm_data, pcm))
            while all_pcm_data.shape[-1] >= self.frame_size:
                self._output_queue.put(all_pcm_data[: self.frame_size])
                all_pcm_data = np.array(all_pcm_data[self.frame_size :])

    async def _recv_loop(self, ws) -> None:
        try:
            async for message in ws:
                if not isinstance(message, bytes):
                    log(
                        "warning", f"unsupported message type {type(message)}"
                    )
                    continue
                if len(message) == 0:
                    log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    self._opus_reader.append_bytes(payload)
                elif kind == 2:  # text
                    payload = message[1:]
                    payload = payload.decode()
                    print(payload, flush=True, end='')

        except Exception as e:
            print(e)
            self._lost_connection()
            return

    def _lost_connection(self) -> None:
        if not self._done:
            log("error", "Lost connection with the server!")
            self._done = True

    def _on_audio_input(self, in_data, frames, time_, status) -> None:
        assert in_data.shape == (self.frame_size, self.channels), in_data.shape
        self._opus_writer.append_pcm(in_data[:, 0])

    def _on_audio_output(self, out_data, frames, time_, status) -> None:
        assert out_data.shape == (self.frame_size, self.channels), out_data.shape
        try:
            pcm_data = self._output_queue.get(block=False)
            assert pcm_data.shape == (self.frame_size,), pcm_data.shape
            out_data[:, 0] = pcm_data
        except queue.Empty:
            out_data.fill(0)

    async def run(self) -> None:
        log("info", f"Connecting to {self.ws_uri}")
        with self._in_stream, self._out_stream:
            async with connect(self.ws_uri, proxy=None) as ws:
                await asyncio.gather(
                    self._recv_loop(ws), self._decoder_loop(), self._send_loop(ws)
                )
        log("info", "Disconnected from the server")


class Client:
    def __init__(self, args):
        self.ws_uri = self._get_uri(args)
        self.connection = None

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
                    log("error", f"The provided URL {args.url} seems to contain a protocol but it is unknown.")
                    sys.exit(1)
            else:
                without_proto = args.url
            return f"{proto}://{without_proto}/api/chat"

    async def _on_key_press(self, key: str) -> None:
        if not self.connection and (key == "pagedown" or key == "down"):
            log("info", "Starting conversation...")
            asyncio.create_task(self._connect())
        elif self.connection and (key == "pageup" or key == "up"):
            log("warning", "Stop current conversation")
            self._disconnect()

    async def _connect(self):
        self.connection = Connection(self.ws_uri)
        await self.connection.run()

    def _disconnect(self):
        if self.connection:
            self.connection._done = True
        self.connection = None

    async def run(self) -> None:
        log("info", "Press '↓' to start the conversation")
        log("info", "Press '↑' to stop the conversation")
        log("info", "Press 'ESC' to quit")
        await listen_keyboard_async(self._on_key_press)


def main():
    parser = argparse.ArgumentParser("client_opus")
    parser.add_argument("--host", default="localhost", type=str, help="Hostname to connect to.")
    parser.add_argument("--port", default=8998, type=int, help="Port to connect to.")
    parser.add_argument("--https", action='store_true', help="Set this flag for using a https connection.")
    parser.add_argument("--url", type=str, help='Provides directly a URL, e.g. to a gradio tunnel.')
    args = parser.parse_args()
    try:
        asyncio.run(Client(args).run())
    except KeyboardInterrupt:
        log("warning", "Interrupting, exiting connection.")
    log("info", "All done!")


if __name__ == "__main__":
    main()
