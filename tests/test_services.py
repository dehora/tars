"""Tests for the shared services startup/teardown module."""

import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))


class StartStopTests(unittest.TestCase):
    @mock.patch("tars.mcp._load_mcp_config", return_value=None)
    @mock.patch("tars.taskrunner.TaskRunner")
    def test_start_without_mcp(self, mock_runner_cls, mock_mcp_config) -> None:
        from tars.services import start_services

        mock_runner = mock.Mock()
        mock_runner_cls.return_value = mock_runner

        mcp_client, runner = start_services("claude", "sonnet")

        self.assertIsNone(mcp_client)
        self.assertEqual(runner, mock_runner)
        mock_runner.start.assert_called_once()

    @mock.patch("tars.mcp._load_mcp_config", return_value=None)
    @mock.patch("tars.taskrunner.TaskRunner")
    def test_stop_services(self, mock_runner_cls, mock_mcp_config) -> None:
        from tars.services import start_services, stop_services

        mock_runner = mock.Mock()
        mock_runner_cls.return_value = mock_runner

        mcp_client, runner = start_services("claude", "sonnet")
        stop_services(mcp_client, runner)

        mock_runner.stop.assert_called_once()

    @mock.patch("tars.mcp._load_mcp_config", return_value={"test": {"command": "echo"}})
    @mock.patch("tars.mcp.MCPClient")
    @mock.patch("tars.taskrunner.TaskRunner")
    def test_start_with_mcp(self, mock_runner_cls, mock_mcp_cls, mock_config) -> None:
        from tars.services import start_services, stop_services

        mock_mcp = mock.Mock()
        mock_mcp.discover_tools.return_value = [{"name": "test_tool"}]
        mock_mcp_cls.return_value = mock_mcp
        mock_runner = mock.Mock()
        mock_runner_cls.return_value = mock_runner

        mcp_client, runner = start_services("claude", "sonnet")

        self.assertEqual(mcp_client, mock_mcp)
        mock_mcp.start.assert_called_once()

        stop_services(mcp_client, runner)
        mock_mcp.stop.assert_called_once()
        mock_runner.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
