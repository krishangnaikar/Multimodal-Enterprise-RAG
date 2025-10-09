from __future__ import annotations
import cv2
import os
from pathlib import Path
import ffmpeg
import imageio_ffmpeg as iio_ff
import shutil

def extract_frames(path, out_dir, every_n = 60):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    frame_count = 0
    saved = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n == 0:
            frame_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved.append(frame_path)
            idx += 1
        frame_count += 1
    cap.release()
    return saved

def _get_ffmpeg_bin():
    try:
        return iio_ff.get_ffmpeg_exe()
    except Exception:
        pass
    which = shutil.which("ffmpeg")
    return which

def extract_audio(video_path, out_audio_path, codec = "mp3", overwrite = True):
    ffmpeg_bin = _get_ffmpeg_bin()
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg (e.g., `choco install ffmpeg`) "
            "or keep imageio-ffmpeg in requirements."
        )

    out = Path(out_audio_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        return str(out)

    stream = ffmpeg.input(video_path)
    if codec == "wav":
        stream = ffmpeg.output(stream.audio, str(out), acodec="pcm_s16le", ac=1, ar="16000")
    else:
        stream = ffmpeg.output(stream.audio, str(out), acodec="libmp3lame", ac=1, ar="16000", audio_bitrate="96k")

    try:
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr.decode('utf-8', 'ignore') if hasattr(e, 'stderr') else e}") from e

    return str(out)