"""Check inference GPU isolation, server startup failures and owned-process cleanup."""

import json
import os
import subprocess
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import URLError

from train.student_server import StudentServer


class StudentServerTests(unittest.TestCase):
    def setUp(self):
        self.process = Mock(pid=12345)
        self.process.poll.return_value = None
        self.addCleanup(patch.stopall)
        self.popen = patch("train.student_server.subprocess.Popen", return_value=self.process).start()
        patch("train.student_server.atexit.register").start()
        self.killpg = patch("train.student_server.os.killpg", create=True).start()
        self.urlopen = patch("train.student_server.urlopen").start()
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
            self.server = StudentServer("/models/student", "1", 0.5, 3072, 32)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_server_gpu_memory_and_nccl_configuration(self):
        command = self.popen.call_args.args[0]
        env = self.popen.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["VLLM_SERVER_DEV_MODE"], "1")
        self.assertEqual(env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])
        for flag, value in (
            ("--host", "127.0.0.1"), ("--gpu-memory-utilization", "0.5"),
            ("--max-model-len", "3072"), ("--max-num-seqs", "32"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(json.loads(command[command.index("--weight-transfer-config") + 1]), {"backend": "nccl"})
        self.assertNotIn("--enable-sleep-mode", command)
        self.assertNotEqual(int(command[command.index("--port") + 1]), self.server.group_port)

    def test_wait_until_healthy(self):
        response = MagicMock()
        response.__enter__.return_value.status = 200
        self.urlopen.side_effect = [URLError("starting"), response]
        self.process.wait.side_effect = subprocess.TimeoutExpired("vllm", 1)
        self.server.wait_ready(attempts=2)
        self.assertEqual(self.urlopen.call_count, 2)

    def test_early_exit_fails_without_waiting_for_timeout(self):
        self.process.poll.return_value = 1
        self.process.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
            self.server.wait_ready()
        self.urlopen.assert_not_called()

    def test_unhealthy_server_times_out(self):
        self.urlopen.side_effect = URLError("not ready")
        self.process.wait.side_effect = subprocess.TimeoutExpired("vllm", 1)
        with self.assertRaises(TimeoutError):
            self.server.wait_ready(attempts=2)

    def test_close_is_idempotent_and_only_targets_owned_process(self):
        self.server.close()
        self.server.close()
        if os.name == "posix":
            self.assertEqual(self.killpg.call_args.args[0], self.process.pid)
            self.killpg.assert_called_once()
        else:
            self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once_with(timeout=10)


if __name__ == "__main__":
    unittest.main()
