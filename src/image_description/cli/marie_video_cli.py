"""Produce a portrait or square H.264 slideshow with gentle movement and crossfades."""
import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def make_command(paths, output, duration, fade, fps, movement):
    """Build an FFmpeg filter graph; each clip overlaps the previous by fade seconds."""
    n = len(paths)
    clip = (duration + (n - 1) * fade) / n
    frames = max(2, round(clip * fps))
    width, height = Image.open(paths[0]).size
    if width % 2 or height % 2:
        raise ValueError("Images must have even pixel dimensions")
    filters = []
    for i in range(n):
        # zoompan always emits exactly frames frames per still image.
        if movement == "none":
            zoom = "1"
            x = "0"
            y = "0"
        else:
            zoom = "min(1.08,1+on*0.08/" + str(max(frames - 1, 1)) + ")"
            x = "(iw-iw/zoom)*on/" + str(max(frames - 1, 1)) if i % 2 == 0 else "(iw-iw/zoom)*(1-on/" + str(max(frames - 1, 1)) + ")"
            y = "(ih-ih/zoom)/2"
        filters.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={fps},"
            f"trim=duration={clip:.6f},setpts=PTS-STARTPTS,format=yuv420p[v{i}]"
        )
    last = "v0"
    for i in range(1, n):
        target = f"xf{i}"
        offset = i * (clip - fade)
        filters.append(
            f"[{last}][v{i}]xfade=transition=fade:duration={fade:.6f}:"
            f"offset={offset:.6f}[{target}]"
        )
        last = target
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for path in paths:
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", str(path)]
    cmd += ["-filter_complex", ";".join(filters), "-map", f"[{last}]",
            "-t", str(duration), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-r", str(fps), str(output)]
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder of prepared images")
    parser.add_argument("--output", type=Path, required=True, help="Resulting .mp4")
    parser.add_argument("--seconds", type=float, default=15, help="Total video duration, default 15")
    parser.add_argument("--fade", type=float, default=0.6, help="Crossfade duration, default 0.6")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--movement", choices=["zoom", "none"], default="zoom")
    args = parser.parse_args(argv)
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required; install it and ensure it is on PATH")
    if not args.input.is_dir():
        parser.error("input must be an existing directory")
    paths = sorted((p for p in args.input.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()),
                   key=lambda p: p.name.lower())
    if not paths:
        parser.error("no images found")
    if args.seconds <= 0 or args.fps <= 0 or args.fade < 0:
        parser.error("seconds and fps must be positive; fade cannot be negative")
    if len(paths) > 1 and (args.fade == 0 or args.fade >= args.seconds / (len(paths) - 1)):
        parser.error("fade must be greater than zero and smaller than seconds/(image count - 1)")
    if len(paths) == 1 and args.fade != 0:
        args.fade = 0
    with Image.open(paths[0]) as sample:
        dimensions = sample.size
    for path in paths[1:]:
        with Image.open(path) as photo:
            if photo.size != dimensions:
                parser.error("all prepared images must have identical dimensions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = make_command(paths, args.output, args.seconds, args.fade, args.fps, args.movement)
    subprocess.run(command, check=True)
    print(f"Wrote {args.output} ({len(paths)} images, {args.seconds:g} seconds)")


if __name__ == "__main__":
    main()
