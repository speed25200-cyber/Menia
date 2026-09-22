"""Reproduce the opaque geometric Menia icon (optional authoring tool, Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw

size = 2048
canvas = Image.new("RGB", (size, size))
pixels = canvas.load()
for y in range(size):
    for x in range(size):
        t = (x + y) / (2 * size)
        pixels[x, y] = tuple(round(a + (b-a)*t) for a, b in zip((12, 27, 46), (21, 65, 75)))
draw = ImageDraw.Draw(canvas)
points = [(520, 1400), (520, 620), (1024, 1140), (1528, 620), (1528, 1400)]
draw.line(points, fill=(224, 247, 241), width=126, joint="curve")
for x, y in points:
    draw.ellipse((x-63,y-63,x+63,y+63), fill=(224, 247, 241))
draw.ellipse((952, 402, 1096, 546), fill=(108, 230, 185))
canvas.resize((1024,1024), Image.Resampling.LANCZOS).save(
    Path(__file__).parent / "MeniaApp/Assets.xcassets/AppIcon.appiconset/AppIcon.png")
