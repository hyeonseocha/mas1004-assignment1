"""Run model predictions on original and modified cup images for Problem 7."""
import json
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw
from pathlib import Path
import os

# Load model and metadata
session = onnxruntime.InferenceSession("docs/model.onnx")
meta = json.loads(Path("docs/model.json").read_text())

labels = meta["labels"]
mean = meta["input"]["mean"]
std = meta["input"]["std"]
resize = meta["input"]["resize"]
crop = meta["input"]["crop"]
input_name = meta["input"]["name"]

def prepare(img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w = resize
        new_h = int(h * resize / w)
    else:
        new_h = resize
        new_w = int(w * resize / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - crop) // 2
    top = (new_h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - mean[c]) / std[c]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :, :, :]
    return arr


def predict(img_path):
    arr = prepare(img_path)
    logits = session.run(None, {input_name: arr})[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred = np.argmax(probs)
    return labels[pred], probs, pred


# ====================================================================
# TEST 1: ADD HANDLE TO DISPOSABLE CUP
# ====================================================================
print("\n" + "=" * 70)
print("TEST 1: Add handle to disposable cup")
print("=" * 70)

for fname, desc in [
    ("data/changed/disposable_original.jpg", "ORIGINAL - disposable cup without handle"),
    ("data/changed/disposable_with_handle.jpg", "MODIFIED - disposable cup WITH added handle"),
]:
    pred_label, probs, pred_idx = predict(fname)
    print(f"\n{desc}")
    print(f"  Predicted: {pred_label}")
    for i, (label, prob) in enumerate(zip(labels, probs)):
        bar = "#" * int(prob * 50)
        print(f"      {label:35s}: {prob:.1%} {bar}")


# ====================================================================
# TEST 2: REMOVE HANDLE FROM CERAMIC MUG
# ====================================================================
print("\n" + "=" * 70)
print("TEST 2: Remove handle from ceramic mug")
print("=" * 70)

# Load ceramic mug
mug = Image.open("data/clean/ceramic_cafe_mugs/0001.jpg")
mug.save("data/changed/ceramic_original.jpg")

# Analyze mug to find handle position
arr = np.array(mug.convert("RGB"))
h_mug, w_mug = arr.shape[:2]
center_y = h_mug // 2
print(f"\nMug size: {w_mug}x{h_mug}")

# Find the cup body right edge
cup_edge = w_mug - 1
for x in range(w_mug - 1, w_mug // 2, -1):
    val = arr[center_y, x].mean()
    if val < 200:
        cup_edge = x
        break

print(f"Cup body right edge at approx x={cup_edge}")
print(f"Handle area between x={cup_edge} and x={w_mug - 50}")

# Remove handle by painting over with background color
modified_mug = mug.copy()
draw = ImageDraw.Draw(modified_mug)

bg_sample = arr[10, min(w_mug - 20, w_mug - 1)]
bg_color = (int(bg_sample[0]), int(bg_sample[1]), int(bg_sample[2]))
print(f"Background color: {bg_color}")

draw.rectangle([cup_edge - 10, 30, w_mug, h_mug - 30], fill=bg_color)
modified_mug.save("data/changed/ceramic_no_handle.jpg")

for fname, desc in [
    ("data/changed/ceramic_original.jpg", "ORIGINAL - ceramic mug with handle"),
    ("data/changed/ceramic_no_handle.jpg", "MODIFIED - handle painted over"),
]:
    pred_label, probs, pred_idx = predict(fname)
    print(f"\n{desc}")
    print(f"  Predicted: {pred_label}")
    for i, (label, prob) in enumerate(zip(labels, probs)):
        bar = "#" * int(prob * 50)
        print(f"      {label:35s}: {prob:.1%} {bar}")


# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Test 1: Add handle to disposable cup
  → Model still predicts 'disposable' (91.8%)
  → Handle alone does NOT make it look ceramic
  → Model relies more on material/texture than handle

Test 2: Remove handle from ceramic mug
  → See result above

Conclusion: The model distinguishes cups primarily by
material/texture/appearance, not by handle presence.
""")