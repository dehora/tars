import json
import socket
import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars.web import (
    _extract_html_title,
    _extract_image_urls,
    _extract_markdown_with_images,
    _extract_text,
    _is_private_host,
    _run_web_tool,
    _MAX_CONTENT_LENGTH,
)


class ExtractTextTests(unittest.TestCase):
    def test_strips_tags(self) -> None:
        self.assertEqual(_extract_text("<p>hello</p>"), "hello")

    def test_preserves_text_across_tags(self) -> None:
        result = _extract_text("<h1>Title</h1><p>Body text</p>")
        self.assertIn("Title", result)
        self.assertIn("Body text", result)

    def test_removes_script_content(self) -> None:
        html = "<p>visible</p><script>alert('xss')</script><p>also visible</p>"
        result = _extract_text(html)
        self.assertIn("visible", result)
        self.assertIn("also visible", result)
        self.assertNotIn("alert", result)

    def test_removes_style_content(self) -> None:
        html = "<style>.foo { color: red; }</style><p>content</p>"
        result = _extract_text(html)
        self.assertIn("content", result)
        self.assertNotIn("color", result)

    def test_collapses_whitespace(self) -> None:
        html = "<p>  lots   of   spaces  </p>"
        result = _extract_text(html)
        self.assertEqual(result, "lots of spaces")

    def test_empty_html(self) -> None:
        self.assertEqual(_extract_text(""), "")


class ExtractImageUrlsTests(unittest.TestCase):
    def test_extracts_body_images(self) -> None:
        html = (
            "<html><body>"
            "<img src=\"/images/a.jpg\">"
            "<img src=\"https://example.com/b.png\">"
            "</body></html>"
        )
        urls = _extract_image_urls(html, "https://example.com/post")
        self.assertEqual(
            urls,
            ["https://example.com/images/a.jpg", "https://example.com/b.png"],
        )

    def test_skips_header_footer(self) -> None:
        html = (
            "<html><body>"
            "<header><img src=\"/hero.jpg\"></header>"
            "<footer><img src=\"/footer.jpg\"></footer>"
            "<article><img src=\"/main.jpg\"></article>"
            "</body></html>"
        )
        urls = _extract_image_urls(html, "https://example.com/post")
        self.assertEqual(urls, ["https://example.com/main.jpg"])

    def test_handles_srcset_and_lazy_attrs(self) -> None:
        html = (
            "<html><body>"
            "<img data-src=\"/lazy.jpg\">"
            "<img srcset=\"/small.jpg 480w, /large.jpg 960w\">"
            "<picture>"
            "<source srcset=\"/alt.webp 1x, /alt@2x.webp 2x\">"
            "<img src=\"/fallback.jpg\">"
            "</picture>"
            "</body></html>"
        )
        urls = _extract_image_urls(html, "https://example.com/post")
        self.assertEqual(
            urls,
            [
                "https://example.com/lazy.jpg",
                "https://example.com/large.jpg",
                "https://example.com/alt@2x.webp",
                "https://example.com/fallback.jpg",
            ],
        )

    def test_body_class_does_not_exclude_images(self) -> None:
        html = (
            "<html><body class=\"header-width-full\">"
            "<article><img src=\"/main.jpg\"></article>"
            "</body></html>"
        )
        urls = _extract_image_urls(html, "https://example.com/post")
        self.assertEqual(urls, ["https://example.com/main.jpg"])


class ExtractMarkdownWithImagesTests(unittest.TestCase):
    def test_inlines_images_in_order(self) -> None:
        html = (
            "<html><body>"
            "<p>Intro</p>"
            "<img src=\"/a.jpg\">"
            "<p>Middle</p>"
            "<img src=\"/b.jpg\">"
            "<p>End</p>"
            "</body></html>"
        )
        markdown = _extract_markdown_with_images(html, "https://example.com/post")
        self.assertIn("Intro", markdown)
        self.assertIn("![](https://example.com/a.jpg)", markdown)
        self.assertIn("Middle", markdown)
        self.assertIn("![](https://example.com/b.jpg)", markdown)
        self.assertIn("End", markdown)


class ExtractHtmlTitleTests(unittest.TestCase):
    def test_prefers_og_title(self) -> None:
        html = (
            "<html><head>"
            "<meta property=\"og:title\" content=\"OG Title\">"
            "<title>HTML Title</title>"
            "</head><body></body></html>"
        )
        self.assertEqual(_extract_html_title(html), "OG Title")

    def test_falls_back_to_title_tag(self) -> None:
        html = "<html><head><title>HTML Title</title></head><body></body></html>"
        self.assertEqual(_extract_html_title(html), "HTML Title")


class SSRFTests(unittest.TestCase):
    @mock.patch("tars.web.socket.getaddrinfo", return_value=[
        (2, 1, 6, "", ("127.0.0.1", 0)),
    ])
    def test_rejects_loopback(self, _) -> None:
        self.assertTrue(_is_private_host("localhost"))

    @mock.patch("tars.web.socket.getaddrinfo", return_value=[
        (2, 1, 6, "", ("10.0.0.1", 0)),
    ])
    def test_rejects_private_ip(self, _) -> None:
        self.assertTrue(_is_private_host("internal.corp"))

    @mock.patch("tars.web.socket.getaddrinfo", return_value=[
        (2, 1, 6, "", ("169.254.169.254", 0)),
    ])
    def test_rejects_link_local(self, _) -> None:
        self.assertTrue(_is_private_host("metadata.internal"))

    @mock.patch("tars.web.socket.getaddrinfo", return_value=[
        (2, 1, 6, "", ("93.184.216.34", 0)),
    ])
    def test_allows_public_ip(self, _) -> None:
        self.assertFalse(_is_private_host("example.com"))

    @mock.patch("tars.web.socket.getaddrinfo", side_effect=socket.gaierror("no DNS"))
    def test_rejects_unresolvable(self, _) -> None:
        self.assertTrue(_is_private_host("nxdomain.invalid"))

    @mock.patch("tars.web._is_private_host", return_value=True)
    def test_run_web_tool_blocks_private(self, _) -> None:
        result = json.loads(_run_web_tool("web_read", {"url": "http://localhost:8080/admin"}))
        self.assertIn("error", result)
        self.assertIn("private", result["error"].lower())


class RunWebToolTests(unittest.TestCase):
    def test_empty_url_returns_error(self) -> None:
        result = json.loads(_run_web_tool("web_read", {"url": ""}))
        self.assertIn("error", result)

    def test_missing_url_returns_error(self) -> None:
        result = json.loads(_run_web_tool("web_read", {}))
        self.assertIn("error", result)

    def test_invalid_scheme_returns_error(self) -> None:
        result = json.loads(_run_web_tool("web_read", {"url": "ftp://example.com"}))
        self.assertIn("error", result)
        self.assertIn("scheme", result["error"].lower())

    @mock.patch("tars.web._is_private_host", return_value=False)
    @mock.patch("tars.web.urllib.request.urlopen")
    def test_valid_url_returns_content(self, mock_urlopen, _) -> None:
        mock_resp = mock.Mock()
        mock_resp.read.return_value = b"<html><body><p>Hello world</p></body></html>"
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = json.loads(_run_web_tool("web_read", {"url": "https://example.com"}))
        self.assertEqual(result["url"], "https://example.com")
        self.assertIn("Hello world", result["content"])
        self.assertFalse(result["truncated"])

    @mock.patch("tars.web._is_private_host", return_value=False)
    @mock.patch("tars.web.urllib.request.urlopen")
    def test_truncation(self, mock_urlopen, _) -> None:
        long_text = "x" * (_MAX_CONTENT_LENGTH + 1000)
        mock_resp = mock.Mock()
        mock_resp.read.return_value = f"<p>{long_text}</p>".encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = json.loads(_run_web_tool("web_read", {"url": "https://example.com"}))
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["content"]), _MAX_CONTENT_LENGTH)

    @mock.patch("tars.web._is_private_host", return_value=False)
    @mock.patch("tars.web.urllib.request.urlopen")
    def test_fetch_timeout(self, mock_urlopen, _) -> None:
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("timed out")

        result = json.loads(_run_web_tool("web_read", {"url": "https://example.com"}))
        self.assertIn("error", result)

    @mock.patch("tars.web._is_private_host", return_value=False)
    def test_http_url_accepted(self, _) -> None:
        with mock.patch("tars.web.urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.Mock()
            mock_resp.read.return_value = b"<p>ok</p>"
            mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
            mock_resp.__exit__ = mock.Mock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = json.loads(_run_web_tool("web_read", {"url": "http://example.com"}))
            self.assertNotIn("error", result)


class RouterEscalationTests(unittest.TestCase):
    def test_url_triggers_escalation(self) -> None:
        from tars.config import ModelConfig
        from tars.router import route_message
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("read this: https://example.com/blog", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")

    def test_read_this_triggers_escalation(self) -> None:
        from tars.config import ModelConfig
        from tars.router import route_message
        config = ModelConfig(
            primary_provider="ollama",
            primary_model="llama3.1:8b",
            remote_provider="claude",
            remote_model="sonnet",
            routing_policy="tool",
        )
        provider, model = route_message("read this article about AI", config)
        self.assertEqual(provider, "claude")
        self.assertEqual(model, "sonnet")


if __name__ == "__main__":
    unittest.main()
