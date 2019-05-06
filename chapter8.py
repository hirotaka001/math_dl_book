# 必要ライブラリの宣言
import sys, io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

# sys.stdoutのエンコード変更
enc = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding=enc)

# PDF出力用
set_matplotlib_formats('png', 'pdf')

## シグモイド関数のグラフ
# 図8-4
xx = np.linspace(-6, 6, 500)
yy = 1 / (1 + np.exp(-xx))

plt.figure(figsize=(6, 6))
plt.ylim(-3, 3)
plt.xlim(-3, 3)
plt.xticks(np.linspace(-3, 3, 13))
plt.yticks(np.linspace(-3, 3, 13))
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid()
plt.plot(xx, yy, c='b', label=r'$\dfrac{1}{1+\exp{(-x)}}$', lw=1)
plt.plot(xx, yy, c='k', label=r'$y = x$', lw=1)
plt.plot([-3, 3], [0, 0], c='k')
plt.plot([0, 0], [-3, 3], c='k')
plt.plot([-3, 3], [1, 1], linestyle='-.', c='k')
plt.legend(fontsize=14)
plt.savefig('sigmoid_function.png')