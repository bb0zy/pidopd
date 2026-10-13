"""Managed local vLLM server on the inference GPU; weights arrive through NCCL."""

import atexit
import os
import signal
import socket
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen


class StudentServer:
    def __init__(self, model_path, gpu, memory_utilization, max_model_len, max_num_seqs):
        # Allocate distinct local ports for HTTP and the weight-transfer group.
        with socket.socket() as http_socket, socket.socket() as group_socket:
            http_socket.bind(("127.0.0.1", 0))
            group_socket.bind(("127.0.0.1", 0))
            port = http_socket.getsockname()[1]
            self.group_port = group_socket.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model_path,
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(memory_utilization),
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--max-num-batched-tokens", "4096",
            "--generation-config", "vllm",
            "--no-enable-prefix-caching",
            "--weight-transfer-config", '{"backend": "nccl"}',
            "--logprobs-mode", "processed_logprobs",
            "--max-logprobs", "0",
            "--disable-log-stats",
        ]
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=gpu,
            VLLM_SERVER_DEV_MODE="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            HF_HUB_OFFLINE="1",
        )
        # Own process group lets cleanup terminate the API server and its engine workers.
        self.process = subprocess.Popen(command, env=env, start_new_session=True)
        atexit.register(self.close)

    def wait_ready(self, attempts=600):
        for _ in range(attempts):
            if self.process.poll() is not None:
                raise RuntimeError(f"Student vLLM server exited with code {self.process.returncode}; see its output above.")
            try:
                with urlopen(f"{self.base_url}/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError):
                pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        raise TimeoutError("Student vLLM server did not become ready; see its output above.")

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
