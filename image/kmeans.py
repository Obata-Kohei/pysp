import numpy as np
import cv2

def kmeans(img, k, attempts=10, type_criteria=cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter_criteria=100, eps_criteria=0.2):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像を2次元配列に変換
    pixels = img.reshape(-1, 3)
    pixels = np.float32(pixels)

    # K-meansクラスタリングのパラメータ設定
    criteria = (type_criteria, max_iter_criteria, eps_criteria)

    # K-meansクラスタリングを実行
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # クラスタの中心をuint8に変換
    centers = np.uint8(centers)

    # 各ピクセルを最も近いクラスタの中心に置き換え
    new_pixels = centers[labels.flatten()]
    new_img = new_pixels.reshape(img.shape)

    return new_img
