import atexit
from pathlib import Path
from typing import Union, Iterator, Optional, Dict
import cv2
import numpy as np


class DisplayExit(Exception):
    """Exception raised when user wants to exit display (e.g., presses 'q')."""
    pass


class MediaInfo:
    """Media metadata container."""
    
    def __init__(self, width: int, height: int, fps: float, frame_count: int, 
                 duration: float, codec: str = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration = duration
        self.codec = codec
    
    def __repr__(self):
        return (f"MediaInfo(resolution={self.width}x{self.height}, "
                f"fps={self.fps:.2f}, frames={self.frame_count}, "
                f"duration={self.duration:.2f}s)")


class Media:
    """Unified media handler for videos, images, and 3D files."""
    
    def __init__(self, source: Union[str, int], width: int = None):
        self.source = source
        self._info = None
        self._cap = None
        self._resize_width = width
        self._source_type = self._detect_source_type(source)
    
    def _detect_source_type(self, source: Union[str, int]) -> str:
        """Detect the type of media source."""
        if isinstance(source, int):
            return "webcam"
        elif isinstance(source, str):
            if source.startswith(('rtsp://', 'rtmp://', 'udp://')):
                return "stream"
            elif source.startswith(('http://', 'https://')):
                return "url"
            elif Path(source).exists():
                return "file"
            else:
                # Try to find in local directory with same name
                local_file = Path(source).name
                if Path(local_file).exists():
                    self.source = local_file
                    return "file"
                return "mapped"  # Will need resource manager
        return "unknown"
    
    def _get_capture(self) -> cv2.VideoCapture:
        """Get or create cached VideoCapture."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.source}")
        return self._cap
    
    @property
    def info(self) -> MediaInfo:
        """Get media metadata."""
        if self._info is None:
            cap = self._get_capture()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            self._info = MediaInfo(width, height, fps, frame_count, duration, codec)
        return self._info
    
    @property
    def frames(self) -> Iterator[np.ndarray]:
        """Generate frames from the media source."""
        cap = self._get_capture()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if width specified
            if self._resize_width:
                h, w = frame.shape[:2]
                height = int(h * self._resize_width / w)
                frame = cv2.resize(frame, (self._resize_width, height))
            
            yield frame
    
    def __del__(self):
        """Clean up VideoCapture on destruction."""
        if self._cap is not None:
            self._cap.release()


# Global writer cache with specs
_writers: Dict[str, cv2.VideoWriter] = {}
_writer_specs: Dict[str, MediaInfo] = {}

def write_frame(output_path: str, frame: np.ndarray, video_info: MediaInfo = None, width: int = None):
    """Write a single frame to video file. Auto-manages video writer lifecycle.
    
    Args:
        output_path: Path to output video file
        frame: Frame to write (numpy array)
        video_info: Video metadata (required on first call, cached afterward)
        width: Optional width for resizing output (maintains aspect ratio)
    """
    # Resize frame if width specified
    if width:
        h, w = frame.shape[:2]
        height = int(h * width / w)
        frame = cv2.resize(frame, (width, height))
    
    if output_path not in _writers:
        # Get video_info from parameter or cache
        if video_info is None:
            if output_path in _writer_specs:
                video_info = _writer_specs[output_path]
            else:
                raise ValueError(f"First call to write_frame('{output_path}') requires video_info parameter")
        
        # Adjust video_info if frame was resized
        if width:
            h, w = frame.shape[:2]
            video_info = MediaInfo(w, h, video_info.fps, video_info.frame_count, 
                                 video_info.duration, video_info.codec)
        
        # Cache the specs for future calls
        _writer_specs[output_path] = video_info
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, video_info.fps, 
            (video_info.width, video_info.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        _writers[output_path] = writer
    
    _writers[output_path].write(frame)

@atexit.register
def _cleanup_writers():
    """Clean up all video writers on exit."""
    for writer in _writers.values():
        if writer.isOpened():
            writer.release()
    _writers.clear()
    _writer_specs.clear()




# Display functions
def show_frame(window_name: str, frame: np.ndarray, wait_key: int = 1, width: int = None) -> None:
    """Display a frame in a window. Exits program gracefully if 'q' is pressed.
    
    Args:
        window_name: Name of the display window
        frame: Frame to display (numpy array)
        wait_key: Milliseconds to wait for key press (default: 1)
        width: Optional width for display resize (maintains aspect ratio, huge performance boost!)
    """
    # Resize for display if width specified (performance optimization)
    if width:
        h, w = frame.shape[:2]
        height = int(h * width / w)
        display_frame = cv2.resize(frame, (width, height))
    else:
        display_frame = frame
    
    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(wait_key) & 0xFF
    
    if key == ord('q'):
        close_display()
        exit(0)  # Clean program exit


def close_display():
    """Close all display windows."""
    cv2.destroyAllWindows()


