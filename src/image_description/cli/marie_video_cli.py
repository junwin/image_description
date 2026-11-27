"""Create a 9:16 Marie-Parie slideshow MP4 with optional gentle movement."""
import argparse
import shutil
import subprocess
from pathlib import Path


def total_duration(image_count, seconds_per_image=2.0, fade=0.4):
    """Crossfades overlap adjacent clips, shortening the finished movie."""
    if image_count < 1 or seconds_per_image <= 0 or fade < 0:
        raise ValueError("Image count/duration must be positive; fade cannot be negative")
    if image_count > 1 and fade >= seconds_per_image:
        raise ValueError("Fade must be shorter than the time per image")
    return image_count * seconds_per_image - (image_count - 1) * fade


def make_command(paths, output, duration, fade, fps, movement, output_size=(720, 1280), supersample=2):
    """Build an FFmpeg command; scale/letterbox mixed aspect ratios as needed.

    Each image is displayed for (duration + overlaps) / N seconds, with
    fade seconds of overlap between successive images. Final duration is
    exactly duration (to frame precision).
    """
    n = len(paths)
    if n == 0 or duration <= 0 or fps <= 0 or fade < 0:
        raise ValueError("Invalid video duration, FPS, fade, or empty input")
    width, height = output_size
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise ValueError("H.264 output dimensions must be positive and even")
    if supersample not in (1, 2, 3):
        raise ValueError("supersample must be 1, 2 or 3")
    if movement not in ("none", "zoom"):
        raise ValueError("movement must be none or zoom")
    clip = (duration + (n - 1) * fade) / n
    if n > 1 and fade >= clip:
        raise ValueError("Crossfade must be shorter than each clip")
    frames = max(2, round(clip * fps))
    # Zoompan samples its crop at integer input pixels. Enlarging the working
    # frame before motion and reducing with Lanczos attenuates visible stepping.
    motion_width, motion_height = width * supersample, height * supersample
    filters = []
    for i in range(n):
        if movement == "none":
            zoom, x, y = "1", "0", "0"
        else:
            # Small (~4.5%) animated zoom, alternating direction across clips.
            zoom = "min(1.045,1+on*0.045/" + str(max(frames - 1, 1)) + ")"
            position = "on/" + str(max(frames - 1, 1))
            x = ("(iw-iw/zoom)*" + position if i % 2 == 0
                 else "(iw-iw/zoom)*(1-" + position + ")")
            y = "(ih-ih/zoom)/2"
        filters.append(
            f"[{i}:v]scale={motion_width}:{motion_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={motion_width}:{motion_height}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:"
            f"s={motion_width}x{motion_height}:fps={fps},"
            f"scale={width}:{height}:flags=lanczos,trim=duration={clip:.6f},"
            f"setpts=PTS-STARTPTS,fps={fps},settb=AVTB,format=yuv420p[v{i}]"
        )
    if n > 1 and fade == 0:
        joined = "".join(f"[v{i}]" for i in range(n))
        filters.append(f"{joined}concat=n={n}:v=1:a=0[finished]")
        last = "finished"
    else:
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
    cmd += [
        "-filter_complex", ";".join(filters), "-map", f"[{last}]",
        "-t", str(duration), "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-r", str(fps), str(output),
    ]
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder of prepared images")
    parser.add_argument("--output", type=Path, required=True, help="Resulting .mp4")
    parser.add_argument("--seconds-per-image", type=float, default=2.0,
                        help="Display time per photo including transition (default 2)")
    parser.add_argument("--seconds", type=float,
                        help="Optional legacy override: entire video duration")
    parser.add_argument("--fade", type=float, default=0.4,
                        help="Overlapping transition duration (default 0.4)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=720, help="Output width (default 720)")
    parser.add_argument("--height", type=int, default=1280, help="Output height (default 1280)")
    parser.add_argument("--supersample", type=int, choices=[1, 2, 3], default=2,
                        help="Render motion at 2x resolution before downsampling (1 disables)")
    parser.add_argument("--movement", choices=["zoom", "none"], default="zoom",
                        help="Subtle zoom/pan or static images")
    args = parser.parse_args(argv)
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required; install it using sudo apt install ffmpeg")
    if not args.input.is_dir():
        parser.error("input must be an existing directory")
    paths = sorted((p for p in args.input.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: p.name.casefold())
    if not paths:
        parser.error("no images found")
    if args.fps < 1 or min(args.width, args.height) < 1 or args.width % 2 or args.height % 2:
        parser.error("fps must be positive; width and height must be positive even numbers")
    try:
        duration = total_duration(len(paths), args.seconds_per_image, args.fade)
        if args.seconds is not None:
            duration = args.seconds
        if duration <= 0:
            raise ValueError("Total duration must be positive")
        command = make_command(paths, args.output, duration, args.fade,
                               args.fps, args.movement, (args.width, args.height), args.supersample)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    print(f"Wrote {args.output} ({len(paths)} photos, {duration:g}s, "
          f"{args.width}x{args.height}, {args.fade:g}s crossfade)")


if __name__ == "__main__":
    main()
