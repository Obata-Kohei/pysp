import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(image, gamma):
    """
    ガンマ補正を行う関数

    Args:
        image (numpy.ndarray): 入力画像（BGR形式）
        gamma (float): ガンマ値（1.0以下で暗く、1.0以上で明るくなる）

    Returns:
        numpy.ndarray: ガンマ補正後の画像
    """
    # ガンマ補正用のルックアップテーブルを作成
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    
    # ルックアップテーブルを用いて画像にガンマ補正を適用
    corrected_image = cv2.LUT(image, table)
    return corrected_image



def CRTfilter(img: np.ndarray, direction: str='h'):
    """
    CRTモニタ風の, しま模様を施した画像を返す。ぼかしはかけない

    Parameters:
    - img (np.ndarray): 入力画像
    - direction (str): 'h': 横向き，'v': 縦向きにフィルタをかける

    Returns:
    - np.ndarray: 出力画像
    """

    h, w = img.shape[0], img.shape[1]

    crt = np.zeros((h, w, 3), dtype=np.uint8)
    if direction == 'h':
        for i in range(h):
            if i % 3 == 0:
                crt[i, :, :] = [0, 255, 0]  # 緑 (G)
            elif i % 3 == 1:
                crt[i, :, :] = [255, 0, 0]  # 青 (B)
            elif i % 3 == 2:
                crt[i, :, :] = [0, 0, 255]  # 赤 (R)
    elif direction == 'v':
        for j in range(w):
            if j % 3 == 0:
                crt[:, j, :] = [0, 255, 0]  # 緑 (G)
            elif j % 3 == 1:
                crt[:, j, :] = [255, 0, 0]  # 青 (B)
            elif j % 3 == 2:
                crt[:, j, :] = [0, 0, 255]  # 赤 (R)
    else:
        return None

    dst = (img.astype(np.float32) * crt.astype(np.float32) / 255)
    dst = gamma_correction(dst.astype(np.uint8), 2.0)  # 明るさを上げる(CRTフィルタ適用後は明るさが1/3になる)
    dst = np.clip(dst, 0, 255).astype(np.uint8)  # 値を255にクリップ

    return dst

if __name__ == '__main__':
    # 入力画像の読み込み
    img = cv2.imread('./img/daiwa.jpeg')  # OpenCVはBGR形式で読み込み
    if img is None:
        raise FileNotFoundError("指定した画像ファイルが見つかりません。パスを確認してください。")

    blr = cv2.GaussianBlur(img, (21, 21), 0, 0)  # 画像サイズによってカーネルサイズ変わる
    dst = cv2.cvtColor(CRTfilter(blr), cv2.COLOR_BGR2RGB)

    # 表示
    plt.imshow(dst)
    #plt.axis('off')
    plt.show()
