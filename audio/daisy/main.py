import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf


import numpy as np
import scipy.signal as signal
import soundfile as sf

fs = 16000       # サンプリング周波数
duration = 1.0   # 各母音の長さ（秒）
f0_male = 120    # 男性の基本周波数
f0_female = 200  # 女性の基本周波数

# 男性の母音フォルマント値（参考値, Hz）
# (F1, F2, F3)
formants_male = {
    "a": [(730, 80), (1090, 90), (2440, 120)],
    "i": [(270, 60), (2290, 100), (3010, 120)],
    "u": [(300, 70), (870, 80), (2240, 110)],
    "e": [(530, 70), (1840, 90), (2480, 120)],
    "o": [(570, 80), (840, 90), (2410, 120)],
}

# 女性のフォルマントは男性の値を全体的に25%高くする
formants_female = {
    v: [(int(f*1.25), bw) for f, bw in formants_male[v]] for v in formants_male
}

def excitation_signal(f0, duration, fs):
    """有声音励振（インパルス列）"""
    t = np.arange(0, duration, 1/fs)
    sig = np.zeros_like(t)
    step = int(fs / f0)
    sig[::step] = 1.0
    return sig

def resonator(f, bw, fs):
    """フォルマント共振器フィルタ係数"""
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * f / fs
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r**2]
    return b, a

def synthesize_vowel(formants, f0, duration, fs):
    """フォルマント合成で母音を生成"""
    exc = excitation_signal(f0, duration, fs)
    y = exc.copy()
    for f, bw in formants:
        b, a = resonator(f, bw, fs)
        y = signal.lfilter(b, a, y)
    return y / np.max(np.abs(y))

# 各母音を合成・保存
for vowel in formants_male.keys():
    # 男性版
    y_m = synthesize_vowel(formants_male[vowel], f0_male, duration, fs)
    sf.write(f"male_{vowel}.wav", y_m, fs)

    # 女性版
    y_f = synthesize_vowel(formants_female[vowel], f0_female, duration, fs)
    sf.write(f"female_{vowel}.wav", y_f, fs)

print("男性・女性版の /a, i, u, e, o/ を wav に保存しました。")
