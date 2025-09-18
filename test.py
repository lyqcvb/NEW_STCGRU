import numpy as np
import matplotlib.pyplot as plt
from spectrum import arburg

# 构造测试信号（两个正弦 + 噪声）
fs = 256
N = 512
t = np.arange(N) / fs
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 80 * t) + np.random.randn(N) * 0.1

# Burg方法估计 (AR阶数=15)
ar, var, ref = arburg(x, order=15)

# 计算频谱
frequencies = np.linspace(0, fs/2, 512)
psd = var / np.abs(np.fft.fft(ar, 1024))**2
psd = psd[:512]

plt.semilogy(frequencies, psd)
plt.title("Burg Power Spectral Density Estimation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB)")
plt.grid()
plt.show()
