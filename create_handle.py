"""Create a disposable cup with a visible handle added on the right side."""
from PIL import Image, ImageDraw
import numpy as np
import os

os.makedirs('data/changed', exist_ok=True)

# Load the disposable cup
cup = Image.open('data/clean/disposable_plastic_takeout_cups/0001.jpg').convert('RGB')
w, h = cup.size
print(f'Image size: {w}x{h}')

# Create a copy to modify
modified = cup.copy()
draw = ImageDraw.Draw(modified)

# Handle colors - matching the cup body
handle_color = (225, 228, 232)   # light gray similar to cup body
handle_outline = (190, 195, 200)  # slightly darker for shadow

# Draw a thick rectangular handle on the right side
# The handle will be attached at x≈800 (cup edge) and extend to x≈950
# Vertical span: y≈280 to y≈720

# Step 1: Draw the outer shape - a thick band forming a C
# Top arm of handle (horizontal)
draw.rounded_rectangle([790, 290, 930, 320], radius=12, fill=handle_color, outline=handle_outline)
# Bottom arm of handle (horizontal)
draw.rounded_rectangle([790, 680, 930, 710], radius=12, fill=handle_color, outline=handle_outline)
# Right side (vertical connector)
draw.rounded_rectangle([910, 290, 940, 710], radius=12, fill=handle_color, outline=handle_outline)
# Left side - attachment to cup body
draw.rounded_rectangle([780, 290, 810, 710], radius=6, fill=handle_color, outline=handle_outline)

# Step 2: Add 3D shading
shadow = ImageDraw.Draw(modified)
# Top edge shadow
shadow.rectangle([790, 290, 940, 298], fill=handle_outline)
# Bottom edge shadow
shadow.rectangle([790, 702, 940, 710], fill=handle_outline)
# Right edge shadow
shadow.rectangle([930, 290, 940, 710], fill=handle_outline)

# Step 3: Add a highlight for realism
highlight = (240, 243, 247)
hl = ImageDraw.Draw(modified)
hl.rectangle([912, 305, 922, 695], fill=highlight)

# Also add a slight shadow where handle meets the cup
attach_shadow = (180, 184, 190)
ash = ImageDraw.Draw(modified)
ash.rectangle([778, 290, 785, 710], fill=attach_shadow)

# Save both images
cup.save('data/changed/disposable_original.jpg')
modified.save('data/changed/disposable_with_handle.jpg')

# Verify the difference
arr_o = np.array(cup)
arr_m = np.array(modified)
diff = np.abs(arr_o.astype(np.int32) - arr_m.astype(np.int32))
pixels_diff = (diff > 0).sum()
print(f'Pixels modified: {pixels_diff} / {diff.size} ({100*pixels_diff/diff.size:.1f}%)')
print(f'Max difference: {diff.max()}')
print()
print('Saved:')
print('  data/changed/disposable_original.jpg')
print('  data/changed/disposable_with_handle.jpg')