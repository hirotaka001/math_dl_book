import numpy as np
import scipy.special as scm
import matplotlib.pyplot as plt
from scipy import integrate

# 正規分布関数の定義
def gauss(x, n):
    m = n/2
    return np.exp(-(x-m)**2/m)/np.sqrt(m*np.pi)

# 正規分布関数と二項分布の重ね書き
N = 1000
M = 2**N
X = range(440, 561)
plt.bar(X, [scm.comb(N, i)/M for i in X])
plt.plot(X, gauss(np.array(X), N), c='k', linewidth=2)
# plt.show()
plt.savefig('gauss.png')

# 数値計算による定積分
def normal(x):
    return np.exp(-((x-500)**2)/500)/np.sqrt(500*np.pi)
print(integrate.quad(normal, 0, 480))

# 正規分布関数
def std(x, sigma=1):
    return (np.exp(-(x/sigma)**2/2)) / (np.sqrt(2*np.pi)*sigma)

# シグモイド関数(確率分布関数)
def sigmoid(x):
    return (1/(1+np.exp(x)))

# 座標値の計算
x = np.linspace(-5, 5, 1000)
y_std = std(x, 1.6)
sig = sigmoid(x)
y_sig = sig * (1-sig)

# グラフ描画
plt.figure(figsize=(8, 8))
plt.plot(x, y_std, label="std", c='k', lw=3, linestyle='-.')
plt.plot(x, y_sig, label="sig", c='b', lw=3)
plt.legend(fontsize=14)
plt.grid(lw=2)
plt.savefig('gause_sigmoid.png')