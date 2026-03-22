import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from util import time_axis, Excitation, Formant

fs = 44100       # サンプリング周波数
duration = 1.0   # 各母音の長さ（秒）
f0_male = 120    # 男性の基本周波数
f0_female = 200  # 女性の基本周波数

# 男性の母音フォルマント値（参考値, Hz）
# (F1, F2, F3)
BASE_FORMANTS = {
    "a": [(730, 80, 1.0), (1090, 90, 1.0), (2440, 120, 1.0)],
    "i": [(270, 60, 1.0), (2290, 100, 1.0), (3010, 120, 1.0)],
    "u": [(300, 70, 1.0), (870, 80, 1.0), (2240, 110, 1.0)],
    "e": [(530, 70, 1.0), (1840, 90, 1.0), (2480, 120, 1.0)],
    "o": [(570, 80, 1.0), (840, 90, 1.0), (2410, 120, 1.0)],
}

# 女性のフォルマントは男性の値を全体的に25%高くする
formants_female = {
    v: [(int(f*1.25), bw, g) for f, bw, g in BASE_FORMANTS[v]] for v in BASE_FORMANTS
}


# ----- excitation -----
def impulse(f0, duration, fs):
    """有声音励振（インパルス列）"""
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    sig = np.zeros_like(t)
    step = int(fs / f0)
    sig[::step] = 1.0
    return sig

# duty=50% rect
#excitation = (np.sin(2*np.pi*f0*t) > 0).astype(float)  # 矩形波的パルス

def rosenberg_wave(t, f0, tau1=0.90, tau2=0.05):
    """
    Rosenberg波形を生成する関数
    https://moge32.blogspot.com/2012/08/3c.html

    開時間率Tp, 閉時間率Tnは「1周期に対する比率」で声門の開閉時間を表している
    tau1 = Tp * T0
    tau2 = Tn * T0

    Parameters
    ----------
    t : ndarray
        時間軸 [秒] (numpy配列)
    f0 : float
        基本周波数 [Hz]
    tau1 : float
        声門開大期の長さ [秒]
    tau2 : float
        声門閉小期の長さ [秒]

    Returns
    -------
    y : ndarray
        Rosenberg波形
    """
    period = 1.0 / f0         # 1周期の長さ
    nT = (t % period)         # 各サンプルの周期内の位相時間（0～period）

    y = np.zeros_like(t)

    # 開口上昇 (0 <= t < tau1)
    idx = (nT < tau1)
    y[idx] = 0.5 * (1 - np.cos(np.pi * nT[idx] / tau1))

    # 閉口下降 (tau1 <= t < tau1+tau2)
    idx = (nT >= tau1) & (nT < tau1 + tau2)
    y[idx] = np.cos(np.pi * (nT[idx] - tau1) / (2 * tau2))

    # tau1+tau2以降は閉鎖（ゼロのまま）
    return y


# ----- formant filter -----
def resonator(f, bw, gain, fs):
    """フォルマント共振器フィルタ係数（ゲイン付き）"""
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * f / fs
    b = [gain * (1 - r)]
    a = [1, -2 * r * np.cos(theta), r**2]
    return b, a

def synth_vowel(formants, f0, duration, fs):
    """フォルマント合成で母音を生成"""
    exc = impulse(f0, duration, fs)
    y = exc.copy()
    for f, bw, g in formants:
        b, a = resonator(f, bw, g, fs)
        y = sig.lfilter(b, a, y)
    return y / np.max(np.abs(y))



def synth_s(fs, duration=0.05, f_center=6e3, bw=4e3, intensity=0.17):
    """/s/ の合成
    . 帯域をどう選ぶか
    子音 /s/ の音は「摩擦音」であり，高周波成分に強いエネルギーを持つノイズです。
    母音とは違い，特定のフォルマントではなく「帯域ノイズの強さ」が特徴を決めます。
    4-8 kHz 帯域
    → 一般的な /s/ のエネルギー帯域
    → 標準的に「すー」という摩擦感

    5-9 kHz など高めの帯域
    → より鋭く、明るい /s/
    → 女性や子供の声に近い印象

    3-6 kHz など低めの帯域
    → こもった /s/
    → 高齢男性的、あるいは /ʃ/（「し」）に近い音

    帯域設定が意図する効果
    中心周波数を高める → 明るく鋭い音
    中心周波数を低める → 暗くこもった音
    帯域を広げる → ノイズっぽさ強調
    帯域を狭める → 笛のような鋭い音
    """
    noise = np.random.randn(int(fs*duration))
    # バンドパスフィルタで 4–8 kHz を強調
    b, a = sig.butter(4, [f_center-bw/2, f_center+bw/2], btype='band', fs=fs)
    s = sig.lfilter(b, a, noise)
    return intensity * s / np.max(np.abs(s))



# -----
class TimeAxis:
    STD_SAMPLE_RATE = 44100
    def __init__(self, dur_sec, fs=STD_SAMPLE_RATE):
        self.dur_sec = dur_sec
        self.fs = fs
        self.t = np.linspace(0, self.dur_sec, int(dur_sec * fs), endpoint=False)

class Model:
    BASE_FORMANT = {
        "a": [(730, 80, 1.0), (1090, 90, 1.0), (2440, 120, 1.0)],
        "i": [(270, 60, 1.0), (2290, 100, 1.0), (3010, 120, 1.0)],
        "u": [(300, 70, 1.0), (870, 80, 1.0), (2240, 110, 1.0)],
        "e": [(530, 70, 1.0), (1840, 90, 1.0), (2480, 120, 1.0)],
        "o": [(570, 80, 1.0), (840, 90, 1.0), (2410, 120, 1.0)],
    }
    def __init__(self, name="base", formants=BASE_FORMANT):
        self.name = name
        self.formants = formants


def main():
    s_sound = synth_s(fs, 0.15)
    a_vowel = synth_vowel(formants_female["a"], f0_female, 0.8, fs)  # 800ms の /a/
    wave = np.concatenate([s_sound, a_vowel])
    sd.play(wave, fs)
    sd.wait()
    #sf.write("sa.wav", wave, fs)


if __name__ == "__main__":
    main()
