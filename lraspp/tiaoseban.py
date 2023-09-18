from PIL import Image
import cv2
import json
palette_path = "./palette.json"
with open(palette_path, "rb") as f:
    pallette_dict = json.load(f)
    pallette = []
    for v in pallette_dict.values():
        pallette += v

mask =Image.open(r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\SegmentationClass\1_625.png").convert("L")
mask.putpalette(pallette)
mask.save(r"C:\Users\12955\Desktop\change_driven SemCom\Video_Generation\1_%d.png")
