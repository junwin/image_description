"""Quick smoke test: verify original file is untouched."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_description.image.openai_client import _encode_image_to_base64

test_dir = "/home/junwin/pishare/photography/work/2026/output"
imgs = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
test_img = os.path.join(test_dir, imgs[0])

mtime_before = os.path.getmtime(test_img)
size_before = os.path.getsize(test_img)

# Run resize — must not touch disk
_full = _encode_image_to_base64(test_img, max_side=2048)

mtime_after = os.path.getmtime(test_img)
size_after = os.path.getsize(test_img)

print(f"File: {imgs[0]}")
print(f"Size before:  {size_before}")
print(f"Size after:   {size_after}")
print(f"Mtime before: {mtime_before}")
print(f"Mtime after:  {mtime_after}")
unchanged = mtime_before == mtime_after and size_before == size_after
print(f"Unchanged:    {unchanged}")
assert unchanged, "File was modified on disk!"
print("OK — original file untouched")
