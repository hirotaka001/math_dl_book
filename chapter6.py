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

## 正規分布関数とシグモイド関数
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

## 最尤推定
def L(p, n, k):
    return ((p**k) * ((1-p)**(n-k)))

x = np.linspace(0, 1, 1000)
y = L(x, 5, 2)
x0 = np.asarray([0.4, 0.4])
y0 = np.asarray([0, L(0.4, 5, 2)])

plt.figure(figsize=(6, 6))
plt.plot(x, y, c='b', lw=3)
plt.plot(x0, y0, linestyle='dashed', c='k', lw=3)
plt.xticks(size=16)
plt.yticks(size=16)
plt.grid(lw=2)
plt.xlabel("p", fontsize=16)
plt.ylabel("L(p)", fontsize=16)
plt.savefig('maximum_likelihood_estimation.png')
