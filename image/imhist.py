import cv2
import matplotlib.pyplot as plt

def display_histogram(image):
    # 画像がグレースケールかカラーかを判別
    if len(image.shape) == 2:
        # グレースケール画像の場合
        plt.figure(figsize=(10, 5))
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.xlim([0, 256])
        plt.show()

    elif len(image.shape) == 3:
        # カラー画像の場合
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(15, 5))
        plt.title('Color Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of Pixels')

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.show()

    else:
        print("Unsupported image format")
img = cv2.imread("/Users/kohei/Desktop/情報通信工学実験第2/2-2 Digital Signal and Image Processing/src/lena_g.bmp")
display_histogram(img)