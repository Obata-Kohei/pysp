import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

def time_axis(dur_sec, fs):
    return np.linspace(0, dur_sec, int(fs*dur_sec), endpoint=False)

class Excitation:
    def __init__(self, dur_sec, f0, fs=44100):
        self.dur_sec = dur_sec
        self.f0 = f0
        self.fs = fs
        self.t = time_axis(self.dur_sec, self.fs)

    def impulse(self):
        """有声音励振（インパルス列）"""
        sig = np.zeros_like(self.t)
        step = int(self.fs / self.f0)
        sig[::step] = 1.0
        return sig
    
    def sqr(self):
        return (np.sin(2 * np.pi * self.f0 * self.t) > 0).astype(float)
    
    def rosenberg(self, tau1=0.90, tau2=0.05):
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
        period = 1.0 / self.f0         # 1周期の長さ
        nT = (self.t % period)         # 各サンプルの周期内の位相時間（0～period）

        y = np.zeros_like(self.t)

        # 開口上昇 (0 <= t < tau1)
        idx = (nT < tau1)
        y[idx] = 0.5 * (1 - np.cos(np.pi * nT[idx] / tau1))

        # 閉口下降 (tau1 <= t < tau1+tau2)
        idx = (nT >= tau1) & (nT < tau1 + tau2)
        y[idx] = np.cos(np.pi * (nT[idx] - tau1) / (2 * tau2))

        # tau1+tau2以降は閉鎖（ゼロのまま）
        return y
    
    def white_noise(self):
        return np.random.randn(int(self.fs * self.duration))

class Formant:
    def __init__(self, f, bw, gain):
        self.f = f
        self.bw = bw
        self.gain = gain

    def filter(self):
        """フォルマント共振器フィルタ係数（ゲイン付き）"""
        r = np.exp(-np.pi * self.bw / self.fs)
        theta = 2 * np.pi * self.f / self.fs
        b = [self.gain * (1 - r)]
        a = [1, -2 * r * np.cos(theta), r**2]
        return b, a
    
