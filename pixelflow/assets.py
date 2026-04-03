"""Asset download utility for pixelflow.

Downloads files from dtmfiles.com to the current working directory by default.
Uses only stdlib — no external dependencies.

Usage:
    import pixelflow as pf

    # Downloads from https://dtmfiles.com/pixelflow/dog.jpg
    # Saves to ./dtmfiles/pixelflow/dog.jpg
    path = pf.assets.download("dog.jpg")

    # Downloads from https://dtmfiles.com/xxx/dog.jpg
    # Saves to ./dtmfiles/xxx/dog.jpg (preserves path structure)
    path = pf.assets.download("xxx/dog.jpg")

    # Downloads from https://dtmfiles.com/images/dog.jpg
    # Saves to ./dtmfiles/images/dog.jpg
    path = pf.assets.download("images/dog.jpg")

    # Full URL - used directly, saves to ./dtmfiles/
    path = pf.assets.download("https://example.com/models/model.pth")
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

__all__ = ["download", "get_cache_dir", "clear_cache",
           "DownloadError", "ChecksumError"]

_LIBRARY_NAME = "pixelflow"
_DEFAULT_BASE_URL = "https://dtmfiles.com"
_DTMFILES_DIR = "dtmfiles"
_CHUNK_SIZE = 65536  # 64 KB
_SIDECAR_MAX_BYTES = 1024


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DownloadError(Exception):
    """Raised when a download fails after all retries."""


class ChecksumError(Exception):
    """Raised when a downloaded file's SHA-256 does not match the expected value."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _default_cache_dir() -> Path:
    env = os.environ.get("DATAMARKIN_CACHE_DIR")
    if env:
        return Path(env) / _LIBRARY_NAME

    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA")
        if not root:
            root = str(Path.home() / "AppData" / "Local")
    else:
        root = os.environ.get("XDG_CACHE_HOME")
        if not root:
            root = str(Path.home() / ".cache")

    return Path(root) / "datamarkin" / _LIBRARY_NAME


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_sha256_line(text: str) -> Optional[str]:
    """Parse a shasum-format line and return the hex digest, or None.

    Expected format: ``<64-hex-chars>  <filename>`` (or just the hash).
    """
    token = text.strip().split()[0] if text.strip() else ""
    if len(token) == 64:
        try:
            int(token, 16)
            return token.lower()
        except ValueError:
            return None
    return None


def _construct_url(path: str) -> str:
    """Construct full URL from relative path.

    If path contains '/', use as-is. Otherwise prepend 'pixelflow/'.

    Examples:
        "dog.jpg" → "https://dtmfiles.com/pixelflow/dog.jpg"
        "xxx/dog.jpg" → "https://dtmfiles.com/xxx/dog.jpg"
    """
    if "/" in path:
        return f"{_DEFAULT_BASE_URL}/{path}"
    else:
        return f"{_DEFAULT_BASE_URL}/pixelflow/{path}"


def _fetch_server_sha256(url: str) -> Optional[str]:
    """Fetch ``<url>.sha256`` from the server and return the hash, or None.

    Single attempt, reads at most 1 KB. Returns None on any failure.
    """
    sidecar_url = url + ".sha256"
    try:
        req = urllib.request.Request(sidecar_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(_SIDECAR_MAX_BYTES)
        return _parse_sha256_line(data.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _show_progress(
    filename: str, downloaded: int, total: Optional[int], speed: float = 0.0
) -> None:
    if total and total > 0:
        pct = min(100, int(downloaded * 100 / total))
        bar_width = 30
        filled = int(bar_width * downloaded // total)
        bar = "█" * filled + "░" * (bar_width - filled)
        speed_str = f"/ {_format_bytes(int(speed))}/s" if speed > 0 else ""
        sys.stderr.write(
            f"\rDownloading {filename}: [{bar}] {_format_bytes(downloaded)} / {_format_bytes(total)} ({pct}%){speed_str}"
        )
    else:
        sys.stderr.write(
            f"\rDownloading {filename}: {_format_bytes(downloaded)}"
        )
    sys.stderr.flush()


def _download_file(
    url: str,
    dest: Path,
    *,
    quiet: bool = False,
    retries: int = 3,
) -> None:
    tmp = dest.with_suffix(dest.suffix + ".download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = dest.name

    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total else None
                downloaded = 0
                start_time = time.time()
                last_update = start_time

                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        current_time = time.time()
                        # Update progress every 0.5 seconds or at end
                        if not quiet and (current_time - last_update >= 0.1 or not chunk):
                            elapsed = current_time - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            _show_progress(filename, downloaded, total, speed)
                            last_update = current_time

            if not quiet and downloaded > 0:
                sys.stderr.write("\n")
                sys.stderr.flush()

            os.replace(tmp, dest)
            return

        except urllib.error.HTTPError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.code < 500:
                _cleanup_tmp(tmp)
                raise DownloadError(
                    f"Download failed: {url} (HTTP {e.code})"
                ) from e

            # Don't retry server errors that return HTML error pages
            content_type = e.headers.get("Content-Type", "") if e.headers else ""
            if "text/html" in content_type:
                _cleanup_tmp(tmp)
                raise DownloadError(
                    f"Download failed: {url} "
                    f"(server returned HTTP {e.code} with HTML error page)"
                ) from e

            last_error = e
        except (urllib.error.URLError, OSError) as e:
            last_error = e

        _cleanup_tmp(tmp)

        if attempt < retries - 1:
            wait = 2 ** attempt
            if not quiet:
                sys.stderr.write(
                    f"\nRetry {attempt + 1}/{retries - 1} in {wait}s...\n"
                )
                sys.stderr.flush()
            time.sleep(wait)

    raise DownloadError(
        f"Failed to download {url} after {retries} attempts: {last_error}"
    ) from last_error


def _cleanup_tmp(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download(
    path: str,
    *,
    directory: Optional[Union[str, Path]] = None,
    force: bool = False,
    quiet: bool = False,
    retries: int = 3,
) -> Path:
    """Download a file and return the local Path.

    Files are saved under ``./dtmfiles/`` preserving path structure.

    URL construction:
    - Simple filename (e.g., ``"dog.jpg"``) downloads from
      ``https://dtmfiles.com/pixelflow/dog.jpg``
    - Path with slash (e.g., ``"xxx/dog.jpg"``) downloads from
      ``https://dtmfiles.com/xxx/dog.jpg``
    - Full URLs are used directly

    Examples:
        ```python
        download("dog.jpg")                    # → ./dtmfiles/pixelflow/dog.jpg
        download("xxx/dog.jpg")                # → ./dtmfiles/xxx/dog.jpg
        download("https://example.com/file.zip")  # → ./dtmfiles/file.zip
        ```

    Args:
        path: Remote path relative to dtmfiles.com (e.g. ``"dog.jpg"`` or
            ``"xxx/dog.jpg"``), or a full URL starting with http:// or https://.
        directory: Where to save the file. Defaults to the current working
            directory. Use :func:`get_cache_dir` for the standard cache
            location.
        force: If ``True``, re-download even when the file already exists.
        quiet: Suppress progress output to stderr.
        retries: Number of download attempts (default 3).

    Returns:
        Path to the downloaded file.

    Raises:
        DownloadError: If the download fails after all retries.
        ChecksumError: If SHA-256 verification fails.
    """
    if directory is not None:
        dest_dir = Path(directory)
    else:
        dest_dir = Path.cwd()

    # Handle full URLs vs relative paths
    if path.startswith(("http://", "https://")):
        url = path
        filename = Path(urlparse(path).path).name
        local_path = dest_dir / _DTMFILES_DIR / filename
    else:
        url = _construct_url(path)
        # Local path mirrors URL structure after dtmfiles.com/
        # e.g., "dog.jpg" → pixelflow/dog.jpg, "xxx/dog.jpg" → xxx/dog.jpg
        if "/" in path:
            local_path = dest_dir / _DTMFILES_DIR / path
        else:
            local_path = dest_dir / _DTMFILES_DIR / "pixelflow" / path

    # File exists and no force — return immediately (no hash check)
    if local_path.exists() and not force:
        return local_path

    _download_file(url, local_path, quiet=quiet, retries=retries)

    # Automatic sidecar integrity verification
    server_hash = _fetch_server_sha256(url)
    if server_hash is not None:
        actual = _sha256_file(local_path)
        if actual != server_hash:
            local_path.unlink(missing_ok=True)
            raise ChecksumError(
                f"SHA-256 mismatch for {path}: "
                f"expected {server_hash} (from server), got {actual}"
            )
    elif not quiet:
        sys.stderr.write(
            f"Warning: No SHA-256 sidecar found for {path}, "
            f"skipping integrity verification.\n"
        )

    return local_path


def get_cache_dir() -> Path:
    """Return the standard cache directory path.

    Returns:
        ``~/.cache/datamarkin/pixelflow/`` (or platform equivalent).
    """
    return _default_cache_dir()


def clear_cache(
    path: Optional[str] = None,
) -> None:
    """Remove files from the standard cache directory.

    Args:
        path: If provided, remove only this specific cached file.
            If ``None``, remove the entire cache directory.
    """
    cache = _default_cache_dir()

    if path is not None:
        target = cache / path
        if target.is_file():
            target.unlink()
    else:
        if cache.is_dir():
            shutil.rmtree(cache)
