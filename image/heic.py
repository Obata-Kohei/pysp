import os
from PIL import Image
import cv2
import pillow_heif
import numpy as np

def read_heic(heic_path):
    return pillow_heif.read_heif(heic_path)

def heic2jpg(heic):
    # HEICファイルを読み込む
    jpg = Image.frombytes(
        heic.mode,
        heic.size,
        heic.data,
        "raw",
        heic.mode,
        heic.stride,
    )
    jpg = np.asarray(jpg)

    return jpg

def heic2jpgAll(dir_path):
    for filename in os.listdir(dir_path):
        if filename.lower().endswith('.heic'):
            heic_file_path = os.path.join(dir_path, filename)
            jpg_file_path = os.path.join(dir_path, filename.rsplit('.', 1)[0] + '.jpg')
            img = heic2jpg(read_heic(heic_file_path))
            cv2.imwrite(jpg_file_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

