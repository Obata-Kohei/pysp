import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd

fs = 44100  # Hz

def gent(dur_sec, fs=44100):
    return np.linspace(0, dur_sec, int(dur_sec * fs), endpoint=False)

def sin(t, f):
    return np.sin(2 * np.pi * f * t)

def sqr(t, f):
    return np.sign(np.sin(2 * np.pi * f * t))

def tri(t, f):
    return 2 * np.abs(2 * (t * f % 1) - 1) - 1

def saw(t, f):
    return 2 * (f * t % 1) - 1

def white_noise(t):
    return np.random.uniform(-1, 1, len(t))

def pink_noise(t):
    # Voss-McCartneyアルゴリズム
    N = len(t)
    rows = 64  # 精度を上げたい場合は増やす
    array = np.random.randn(rows, N)
    array = np.cumsum(array, axis=1)
    noise = np.sum(array, axis=0)
    noise = noise / np.max(np.abs(noise))  # 正規化
    return noise

def EG_ADSR(t, sample_rate, attack=0.1, decay=0.2, sustain=0.7, release=0.5):
    env = np.zeros_like(t)
    N = len(t)
    A = int(attack * sample_rate)
    D = int(decay * sample_rate)
    R = int(release * sample_rate)
    S = N - (A + D + R)

    if A > 0:
        env[:A] = np.linspace(0, 1, A)
    if D > 0:
        env[A:A+D] = np.linspace(1, sustain, D)
    if S > 0:
        env[A+D:A+D+S] = sustain
    if R > 0:
        env[A+D+S:] = np.linspace(sustain, 0, R)
    return env

def show_waveform(t, wave):
    plt.figure(figsize=(12, 6))
    plt.plot(t[:], wave[:], color='b')
    plt.title('Wave Form')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def show_one(t, wave):
    pass

def show_spc(wave, fs, log=False):
    fft = np.fft.fft(wave)
    freqs = np.fft.fftfreq(len(fft), 1/fs)
    # 片側スペクトルだけ取り出し
    half = len(fft)//2
    plt.figure(figsize=(8,4))
    plt.plot(freqs[:half], np.abs(fft[:half]))
    if log:
        plt.xscale("log")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude Spectrum")
    plt.title("Spectrum of the Waveform")
    plt.grid()
    plt.show()

def show_psd(wave, fs, log=False):
    f, Pxx = sig.welch(wave, fs=fs, nperseg=1024)
    plt.figure(figsize=(8,4))
    plt.semilogy(f, Pxx)  # 対数表示にすると見やすい
    if log:
        plt.xscale("log")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density")
    plt.title("Power Spectral Density (Welch Method)")
    plt.grid()
    plt.show()

def play(wave, fs):
    sd.play(wave, fs)
    sd.wait()

def load_wav(path):
    fs, wave = wavfile.read(path)  # fs: サンプリング周波数, wave: ndarray
    # int16 など整数型の場合は [-1, 1] に正規化するのが一般的
    if wave.dtype != np.float32 and wave.dtype != np.float64:
        wave = wave.astype(np.float32) / np.iinfo(wave.dtype).max
    if wave.ndim > 1:
        wave = wave.mean(axis=1)  # モノラル化
    return fs, wave





def make_bgm(video_length, filename):

    fs = 44100

    t = gent(video_length, fs=fs)

    base = 528
    freqs = [
        base * 1/2,
        base * 2/3,
        base * 4/5,
        base * 1,
        base * 5/4,
        #base * 3/2,
    ]
    attack = 6
    decay = 0.0
    sustain = 0.8
    release = 15
    max_volume_duration_sec = (video_length - release) / len(freqs) - attack

    sounds = list()
    note_length = attack + max_volume_duration_sec + release
    t_note = gent(note_length, fs)
    for f in freqs:
        w = sin(t_note, f)
        ws = [1/f/(n+1) * sin(t_note, (f+n)*(1 + 0.01*np.random.randn())) for n in np.linspace(0, 5, 7)]
        w = np.sum(ws, axis=0)
        env = EG_ADSR(t_note, fs, attack, decay, sustain, release)
        w *= env

        #w = 2 * (w - np.min(w)) / (np.max(w) - np.min(w)) - 1
        #w *= 0.5
        #w /= np.max(np.abs(w))
        #w *= 1 / np.sqrt(f)

        sounds.append(w)

    # 出力音声の作成
    out = np.zeros_like(t)
    shift = int((attack + max_volume_duration_sec) * fs)
    for i, w in enumerate(sounds):
        start = i * shift
        end = start + len(w)
        if end > len(out):
            end = len(out)
        out[start:end] += w[:end-start]

    # RMSでだいたいの音量
    rms = np.sqrt(np.mean(out**2))
    out *= 0.03 / rms
    # ピークを -3dBFS に制限
    peak = np.max(np.abs(out))
    target_peak = 10**(-3/20)
    if peak > target_peak:
        out *= target_peak / peak

    #show_waveform(t, out)
    play(out, fs)

    audio_int16 = (out * 32767).astype(np.int16)
    wavfile.write(filename, fs, audio_int16)

