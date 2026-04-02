"""Tests for pixelflow.assets — all network calls are mocked."""

import hashlib
import io
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pixelflow import assets
from pixelflow.assets import (
    ChecksumError,
    DownloadError,
    _default_cache_dir,
    _format_bytes,
    _parse_sha256_line,
    _sha256_file,
    clear_cache,
    download,
    get_cache_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DATA = b"hello pixelflow"
SAMPLE_SHA256 = hashlib.sha256(SAMPLE_DATA).hexdigest()


def _make_response(data: bytes = SAMPLE_DATA, status: int = 200):
    """Create a mock urllib response."""
    resp = MagicMock()
    buf = io.BytesIO(data)
    resp.read = buf.read
    resp.headers = {"Content-Length": str(len(data))}
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_sidecar_response(hex_hash: str, filename: str = "file.bin"):
    """Create a mock response for a .sha256 sidecar file."""
    content = f"{hex_hash}  ./{filename}\n".encode()
    return _make_response(content)


# ---------------------------------------------------------------------------
# TestCacheDir
# ---------------------------------------------------------------------------

class TestCacheDir:
    def test_default_linux(self, monkeypatch):
        monkeypatch.delenv("DATAMARKIN_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr("sys.platform", "linux")
        result = _default_cache_dir()
        assert result == Path.home() / ".cache" / "datamarkin" / "pixelflow"

    def test_xdg_override(self, monkeypatch):
        monkeypatch.delenv("DATAMARKIN_CACHE_DIR", raising=False)
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")
        result = _default_cache_dir()
        assert result == Path("/tmp/xdg/datamarkin/pixelflow")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DATAMARKIN_CACHE_DIR", "/tmp/custom")
        result = _default_cache_dir()
        assert result == Path("/tmp/custom/pixelflow")

    def test_windows(self, monkeypatch):
        monkeypatch.delenv("DATAMARKIN_CACHE_DIR", raising=False)
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", "/tmp/fakelocal")
        result = _default_cache_dir()
        assert result == Path("/tmp/fakelocal") / "datamarkin" / "pixelflow"

    def test_get_cache_dir(self, monkeypatch):
        monkeypatch.delenv("DATAMARKIN_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr("sys.platform", "linux")
        result = get_cache_dir()
        assert result == Path.home() / ".cache" / "datamarkin" / "pixelflow"


# ---------------------------------------------------------------------------
# TestDownload
# ---------------------------------------------------------------------------

class TestDownload:
    def test_downloads_to_directory(self, tmp_path):
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp):
            result = download(
                "images/dog.jpg",
                directory=tmp_path,
                quiet=True,
            )
        # New behavior: all downloads go under dtmfiles/
        assert result == tmp_path / "dtmfiles" / "images" / "dog.jpg"
        assert result.read_bytes() == SAMPLE_DATA

    def test_default_downloads_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp):
            result = download("dog.jpg", quiet=True)
        # New behavior: simple filenames go to dtmfiles/pixelflow/
        assert result == tmp_path / "dtmfiles" / "pixelflow" / "dog.jpg"
        assert result.read_bytes() == SAMPLE_DATA

    def test_file_exists_skips_download(self, tmp_path):
        # New behavior: file exists under dtmfiles/
        existing = tmp_path / "dtmfiles" / "images" / "dog.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(SAMPLE_DATA)

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = download("images/dog.jpg", directory=tmp_path, quiet=True)

        mock_urlopen.assert_not_called()
        assert result == existing

    def test_file_exists_no_hash_check(self, tmp_path):
        """Existing files are returned immediately — no SHA-256 verification."""
        # New behavior: file exists under dtmfiles/pixelflow/
        existing = tmp_path / "dtmfiles" / "pixelflow" / "file.bin"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"whatever content")

        with patch("urllib.request.urlopen") as mock_urlopen:
            # Even with an explicit sha256, existing file is returned as-is
            result = download("file.bin", directory=tmp_path, quiet=True)

        mock_urlopen.assert_not_called()
        assert result == existing

    def test_force_redownloads(self, tmp_path):
        # New behavior: cached under dtmfiles/
        cached = tmp_path / "dtmfiles" / "images" / "dog.jpg"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"old data")

        resp = _make_response(b"new data")
        with patch("urllib.request.urlopen", return_value=resp):
            result = download(
                "images/dog.jpg",
                directory=tmp_path,
                force=True,
                quiet=True,
            )
        assert result.read_bytes() == b"new data"

    def test_creates_subdirectories(self, tmp_path):
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp):
            result = download(
                "deep/nested/path/file.bin",
                directory=tmp_path,
                quiet=True,
            )
        # New behavior: under dtmfiles/
        assert result.parent == tmp_path / "dtmfiles" / "deep" / "nested" / "path"
        assert result.exists()

    def test_full_url_download(self, tmp_path):
        """Full URLs are used directly without prepending base URL."""
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            download(
                "https://example.com/models/model.pth",
                directory=tmp_path,
                quiet=True,
            )
        req = mock_urlopen.call_args_list[0][0][0]
        assert req.full_url == "https://example.com/models/model.pth"

    def test_full_url_saves_filename_only(self, tmp_path):
        """Full URLs save only the filename under dtmfiles/."""
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            result = download(
                "https://example.com/models/deep/nested/model.pth",
                directory=tmp_path,
                quiet=True,
            )
        # New behavior: saves under dtmfiles/ with filename only
        assert result == tmp_path / "dtmfiles" / "model.pth"

    def test_full_url_with_directory(self, tmp_path):
        """Full URLs work with custom directory parameter."""
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            result = download(
                "https://example.com/model.pth",
                directory=tmp_path / "weights",
                quiet=True,
            )
        # New behavior: under dtmfiles/ within custom dir
        assert result == tmp_path / "weights" / "dtmfiles" / "model.pth"

    def test_default_url(self, tmp_path, monkeypatch):
        """Simple filename defaults to pixelflow namespace."""
        monkeypatch.delenv("DATAMARKIN_BASE_URL", raising=False)
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            download("dog.jpg", directory=tmp_path, quiet=True)
        req = mock_urlopen.call_args_list[0][0][0]
        assert req.full_url == "https://dtmfiles.com/pixelflow/dog.jpg"

    def test_path_with_slash_uses_as_is(self, tmp_path):
        """Path with slash uses first segment as library name."""
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            download("xxx/dog.jpg", directory=tmp_path, quiet=True)
        req = mock_urlopen.call_args_list[0][0][0]
        assert req.full_url == "https://dtmfiles.com/xxx/dog.jpg"

    def test_nested_path_preserved(self, tmp_path):
        """Nested paths are preserved in URL construction."""
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            download("images/models/dog.jpg", directory=tmp_path, quiet=True)
        req = mock_urlopen.call_args_list[0][0][0]
        assert req.full_url == "https://dtmfiles.com/images/models/dog.jpg"


# ---------------------------------------------------------------------------
# TestSha256
# ---------------------------------------------------------------------------

class TestSha256:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(SAMPLE_DATA)
        assert _sha256_file(f) == SAMPLE_SHA256


# ---------------------------------------------------------------------------
# TestSidecar
# ---------------------------------------------------------------------------

class TestSidecar:
    def test_sidecar_verification_pass(self, tmp_path):
        """After download, sidecar is fetched and hash matches."""
        download_resp = _make_response()
        sidecar_resp = _make_sidecar_response(SAMPLE_SHA256)

        def urlopen_side_effect(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else req
            if url.endswith(".sha256"):
                return sidecar_resp
            return download_resp

        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            result = download("file.bin", directory=tmp_path, quiet=True)

        # New behavior: file saved under dtmfiles/pixelflow/
        assert result == tmp_path / "dtmfiles" / "pixelflow" / "file.bin"
        assert result.read_bytes() == SAMPLE_DATA

    def test_sidecar_verification_mismatch_raises(self, tmp_path):
        """Sidecar hash doesn't match downloaded file — raises ChecksumError."""
        download_resp = _make_response()
        sidecar_resp = _make_sidecar_response("0" * 64)

        def urlopen_side_effect(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else req
            if url.endswith(".sha256"):
                return sidecar_resp
            return download_resp

        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            with pytest.raises(ChecksumError, match="from server"):
                download("file.bin", directory=tmp_path, quiet=True)

        # New behavior: file under dtmfiles/pixelflow/ should be deleted on mismatch
        assert not (tmp_path / "dtmfiles" / "pixelflow" / "file.bin").exists()

    def test_sidecar_404_skips_verification(self, tmp_path):
        """Sidecar not found (404) — download succeeds without verification."""
        download_resp = _make_response()

        def urlopen_side_effect(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else req
            if url.endswith(".sha256"):
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
            return download_resp

        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            result = download("file.bin", directory=tmp_path, quiet=True)

        # New behavior: file under dtmfiles/pixelflow/
        assert result == tmp_path / "dtmfiles" / "pixelflow" / "file.bin"
        assert result.read_bytes() == SAMPLE_DATA

    def test_sidecar_network_error_skips_verification(self, tmp_path):
        """Sidecar fetch fails with network error — download still succeeds."""
        download_resp = _make_response()

        def urlopen_side_effect(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else req
            if url.endswith(".sha256"):
                raise urllib.error.URLError("connection refused")
            return download_resp

        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            result = download("file.bin", directory=tmp_path, quiet=True)

        # New behavior: file under dtmfiles/pixelflow/
        assert result == tmp_path / "dtmfiles" / "pixelflow" / "file.bin"


# ---------------------------------------------------------------------------
# TestParseSha256Line
# ---------------------------------------------------------------------------

class TestParseSha256Line:
    def test_standard_shasum_format(self):
        line = f"{SAMPLE_SHA256}  ./file.bin\n"
        assert _parse_sha256_line(line) == SAMPLE_SHA256

    def test_hash_only(self):
        assert _parse_sha256_line(SAMPLE_SHA256) == SAMPLE_SHA256

    def test_uppercase_normalized(self):
        upper = SAMPLE_SHA256.upper()
        assert _parse_sha256_line(upper) == SAMPLE_SHA256

    def test_empty_string(self):
        assert _parse_sha256_line("") is None

    def test_invalid_hex(self):
        assert _parse_sha256_line("z" * 64) is None

    def test_wrong_length(self):
        assert _parse_sha256_line("abcdef1234") is None


# ---------------------------------------------------------------------------
# TestRetry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retries_on_network_error(self, tmp_path):
        fail_resp = urllib.error.URLError("connection refused")
        ok_resp = _make_response()

        with patch("urllib.request.urlopen", side_effect=[fail_resp, ok_resp]):
            with patch("time.sleep"):
                result = download(
                    "file.bin",
                    directory=tmp_path,
                    quiet=True,
                    retries=3,
                )
        assert result.exists()

    def test_no_retry_on_404(self, tmp_path):
        error = urllib.error.HTTPError(
            "https://dtmfiles.com/pixelflow/missing.bin",
            404, "Not Found", {}, io.BytesIO(b""),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(DownloadError, match="404"):
                download("missing.bin", directory=tmp_path, quiet=True)

    def test_all_retries_exhausted(self, tmp_path):
        error = urllib.error.URLError("timeout")
        with patch("urllib.request.urlopen", side_effect=error):
            with patch("time.sleep"):
                with pytest.raises(DownloadError, match="after 2 attempts"):
                    download(
                        "file.bin",
                        directory=tmp_path,
                        quiet=True,
                        retries=2,
                    )


# ---------------------------------------------------------------------------
# TestClearCache
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clear_specific_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATAMARKIN_CACHE_DIR", str(tmp_path))
        f = tmp_path / "pixelflow" / "images" / "dog.jpg"
        f.parent.mkdir(parents=True)
        f.write_bytes(SAMPLE_DATA)

        clear_cache("images/dog.jpg")
        assert not f.exists()

    def test_clear_all(self, tmp_path, monkeypatch):
        cache = tmp_path / "pixelflow"
        monkeypatch.setenv("DATAMARKIN_CACHE_DIR", str(tmp_path))
        (cache / "images").mkdir(parents=True)
        (cache / "images" / "a.jpg").write_bytes(b"a")
        (cache / "images" / "b.jpg").write_bytes(b"b")

        clear_cache()
        assert not cache.exists()

    def test_clear_nonexistent_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATAMARKIN_CACHE_DIR", str(tmp_path))
        # Should not raise
        clear_cache("nonexistent.bin")

    def test_clear_nonexistent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATAMARKIN_CACHE_DIR", str(tmp_path / "nope"))
        # Should not raise
        clear_cache()


# ---------------------------------------------------------------------------
# TestProgress
# ---------------------------------------------------------------------------

class TestProgress:
    def test_quiet_mode_no_output(self, tmp_path, capsys):
        resp = _make_response()
        with patch("urllib.request.urlopen", return_value=resp):
            download("file.bin", directory=tmp_path, quiet=True)
        captured = capsys.readouterr()
        assert "Downloading" not in captured.err

    def test_progress_output(self, tmp_path, capsys):
        # Use a larger file to ensure progress output is generated
        resp = _make_response(b"x" * 1024 * 1024)  # 1MB
        with patch("urllib.request.urlopen", return_value=resp):
            download("file.bin", directory=tmp_path, quiet=False)
        captured = capsys.readouterr()
        # Either progress bar or warning message should appear
        assert "Downloading" in captured.err or "Warning" in captured.err


# ---------------------------------------------------------------------------
# TestFormatBytes
# ---------------------------------------------------------------------------

class TestFormatBytes:
    def test_bytes(self):
        assert _format_bytes(500) == "500 B"

    def test_kilobytes(self):
        assert _format_bytes(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_bytes(2 * 1024 * 1024 * 1024) == "2.0 GB"


# ---------------------------------------------------------------------------
# TestExceptionHierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_download_error_is_exception(self):
        assert issubclass(DownloadError, Exception)

    def test_checksum_error_is_exception(self):
        assert issubclass(ChecksumError, Exception)

    def test_errors_are_independent(self):
        assert not issubclass(DownloadError, ChecksumError)
        assert not issubclass(ChecksumError, DownloadError)
