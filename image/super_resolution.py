# 超解像アルゴリズムの実装


import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle


ITERATIVE_BACK_PROJECTION = 0
TOTAL_VARIATION_REGULARIZATION = 1
#MAP_ESTIMATION = 2  #実用的でないので実装しない


def iterative_back_projection(img: np.ndarray, scaling_factor: int, iterations: int = 1000, error_scale: float = 0.1) -> np.ndarray:
    """
    Iterative Back Projection (IBP) による超解像処理
    
    Parameters:
    - img (np.ndarray): 入力の低解像度画像
    - scaling_factor (int): 拡大率
    - iterations (int): 反復回数
    - error_scale (float): 誤差スケーリング係数
    
    Returns:
    - np.ndarray: 超解像処理された高解像度画像
    """
    # 初期高解像度画像を作成（双一次補間で拡大）
    dst = cv2.resize(img, (img.shape[1] * scaling_factor, img.shape[0] * scaling_factor), interpolation=cv2.INTER_LINEAR)

    for i in range(iterations):
        # ダウンプロジェクション：高解像度画像を低解像度サイズに縮小
        downsampled_image = cv2.resize(dst, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 誤差計算：低解像度画像とダウンサンプリングした画像との差を求める
        error = img - downsampled_image
        
        # バックプロジェクション：誤差を高解像度に拡大し、スケーリングして加算
        upsampled_error = cv2.resize(error, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_LINEAR)
        dst = dst + error_scale * upsampled_error

    # 結果をクリップして、値が0～255の範囲に収まるようにする
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst


def total_variation_regularization(img: np.ndarray, scaling_factor: int, tv_weight: float = 0.1) -> np.ndarray:
    """
    Total Variation 正則化を用いた超解像処理を行う関数

    Parameters:
    - img (np.ndarray): 入力画像
    - scaling_factor (int): 拡大率
    - tv_weight (float): TV正則化の重み, 高いほどノイズ低減

    Returns:
    - np.ndarray: 超解像処理された画像
    """

    # 1. TV正則化拡大（骨格成分の抽出）
    u_image = denoise_tv_chambolle(img, weight=tv_weight)
    u_image = (u_image * 255).astype(np.uint8)  # 正規化後、uint8型に変換

    # 2. ダウンサンプリング（骨格成分を用いてテクスチャ成分を計算）
    u_downsampled = cv2.resize(u_image, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    texture_component = cv2.subtract(img, u_downsampled)  # 負の値が出ないように引き算

    # 3. テクスチャ成分の抽出とクリッピング
    v_image = np.clip(texture_component, 0, 255)  # 値を0-255の範囲にクリップ

    # 4. 線形補間拡大（テクスチャ成分の拡大）
    V_image = cv2.resize(v_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    # 5. 骨格成分の拡大
    U_image = cv2.resize(u_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    # 6. 合成（拡大画像の生成）
    dst = cv2.add(U_image, V_image)  # 負の値が出ないように加算

    return dst


# 関数のエイリアス
ibp = iterative_back_projection
tv = total_variation_regularization
