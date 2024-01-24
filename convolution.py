import numpy as np
import matplotlib.pyplot as plt
import math

# ガウス関数を定義
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# 入力データを生成
x = np.linspace(0, 2, 10000)  # -5から5までの範囲で100点のデータを生成
input_data = 1/((x-1)**2+0.005**2/4)       # 例としてsin関数を使います

y = np.linspace(-0.035, 0.035, 1000)

# ガウスカーネルを生成
mu = 0
sigma = 0.008/math.sqrt(2*math.log(2))
gaussian_kernel = gaussian(y, mu, sigma)

# 畳み込みを実行
convolved_data = np.convolve(input_data, gaussian_kernel, mode='same')

# グラフで表示
plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.plot(x, input_data)
plt.title('Input Data')

plt.subplot(132)
plt.plot(y, gaussian_kernel)
plt.title('Gaussian Kernel')

plt.subplot(133)
plt.plot(x, convolved_data)
plt.title('Convolved Data')
plt.tight_layout()
plt.show()