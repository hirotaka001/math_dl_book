#!/usr/bin/env python
# 必須ライブラリの宣言
import sys,io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from sklearn.datasets import load_boston

# sys.stdoutのエンコード変更
enc = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding=enc)

# PDF出力用
set_matplotlib_formats('png', 'pdf')

# 学習用データ準備
boston = load_boston()
x_org, yt = boston.data, boston.target
feature_names = boston.feature_names
print('元データ', x_org.shape, yt.shape)
print('項目名', feature_names)

# データ絞り込み(項目 RMのみ)
x_data = x_org[:,feature_names == 'RM']
print('絞り込み後', x_data.shape)

# ダミー変数を追加
x = np.insert(x_data, 0, 1.0, axis=1)
print('ダミー変数追加後', x.shape)

# 入力データxの表示(ダミー変数を含む)
print(x.shape)
print(x[:5, :])

# 正解データ yの表示
print(yt[:5])

# 散布図の表示
plt.scatter(x[:, 1], yt, s=10, c='b')
plt.xlabel('ROOM', fontsize=14)
plt.ylabel('PRICE', fontsize=14)
plt.savefig('scatter_plot.png')

# 予測関数(1,x)の値から予測値ypを計算する. @:内積
def pred(x, w):
    return (x @ w)

## 勾配降下法の初期化処理
# データ系列総数
M = x.shape[0]
# 入力データ次元数(ダミー変数を含む)
D = x.shape[1]
# 繰り返し回数
iters = 50000
# 学習率
alpha = 0.01
# 重みベクトルの初期値(すべての値を1にする)
w = np.ones(D)
# 評価結果記録用(損失関数値のみ記録)
history = np.zeros((0, 2))

# 繰り返しループ
for k in range(iters):
    # 予測値の計算(7.8.1)
    yp = pred(x, w)
    # 誤差の計算(7.8.2)
    yd = yp - yt
    # 勾配降下法の実装(7.8.4) .Tは転置行列
    w = w - alpha * (x.T @ yd) / M
    # 学習曲線描画用データの計算、保存
    if (k % 100 == 0):
        # 損失関数値の計算(7.6.1) mean:平均値の計算
        loss = np.mean(yd**2) / 2
        # 計算結果の記録
        history = np.vstack((history, np.array([k, loss])))
        # 画面表示
        print("iter = %d loss = %f" % (k, loss))

# 最終的な損失関数初期値、最終値
print('損失関数初期値: %f' % history[0, 1])
print('損失関数最終値: %f' % history[-1, 1])

# 下記直線描画用の座標値計算
xall = x[:, 1].ravel()
xl = np.array([[1, xall.min()], [1, xall.max()]])
yl = pred(xl, w)

# 散布図と回帰直線の描画
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 1], yt, s=10, c='b')
plt.xlabel('ROOM', fontsize=14)
plt.ylabel('PROCE', fontsize=14)
plt.plot(xl[:, 1], yl, c='k')
plt.savefig('scatter_regressionline_plot.png')

# 学習曲線の表示(最初の1個分を覗く)
plt.plot(history[1:, 0], history[1:, 1])
plt.savefig('learning_curve.png')

## Numpyを使ったコーディングテクニック
# ベクトル・ベクトル間の内積
# w = (1, 2)
w = np.array([1, 2])
print(w)
print(w.shape)
# x = (3, 4)
x = np.array([3, 4])
print(x)
print(x.shape)
# (3.7.2)式の内積の実装例
# y = 1*3 + 2*4 = 11
y = x @ w
print('ベクトル・ベクトル間の内積 y= ', y)

# 行列・ベクトル間の内積
# X は3行2列の行列
X = np.array([[1, 2], [3, 4], [5, 6]])
print(X)
print(X.shape)
Y = X @ w
print(Y)
print('行列・ベクトル間の内積 Y= ', Y.shape)

# データ系列方向の行列・ベクトル間の内積
# 転置行列の作成
XT = X.T
print(X)
print(XT)
yd = np.array([1, 2, 3])
print(yd)
# 勾配値の計算(の一部)
grad = XT @ yd
print('勾配値の計算 grad= ', grad)

## 重回帰モデルへの拡張
# 列(LSTAT:低所得者率)の追加
x_add = x_org[:,feature_names == 'LSTAT']
x2 = np.hstack((x, x_add))
print(x2.shape)
# 入力データxの表示(ダミーデータを含む)
print(x2[:5, :])

# 初期化処理
# データ系列総数
M = x2.shape[0]
# 入力データ次元数(ダミー変数を含む)
D = x2.shape[1]
# 繰り返し回数
iters = 50000
# 学習率
alpha = 0.01
# 重みベクトルの初期値(すべての値を1にする)
w = np.ones(D)
# 評価結果記録用(損失関数値のみ記録)
history = np.zeros((0, 2))

# 繰り返しループ
for k in range(iters):
    # 予測値の計算(7.8.1)
    yp = pred(x2, w)
    # 誤差の計算(7.8.2)
    yd = yp - yt
    # 勾配降下法の実装(7.8.4)
    w = w - alpha * (x2.T @ yd) / M

    # 学習曲線描画用データの計算、保存
    if(k % 100 == 0):
        # 損失関数値の計算(7.6.1)
        loss = np.mean(yd ** 2) / 2
        # 計算結果の記録
        history = np.vstack((history, np.array([k, loss])))
        # 画面表示
        print("iter = %d loss = %f" % (k, loss))

# 初期化処理(パラメータを的殺な値に変更)
# データ系列総数
M = x2.shape[0]
# 入力データ次元数(ダミー変数を含む)
D = x2.shape[1]
# 繰り返し回数
iters = 2000
# 学習率
alpha = 0.001
# 重みベクトルの初期値(すべての値を1にする)
w = np.ones(D)
# 評価結果記録用(損失関数値のみ記録)
history = np.zeros((0, 2))

# 繰り返しループ
for k in range(iters):
    # 予測値の計算(7.8.1)
    yp = pred(x2, w)
    # 誤差の計算(7.8.2)
    yd = yp - yt
    # 勾配降下法の実装(7.8.4)
    w = w -alpha * (x2.T @ yd) / M
    # 学習曲線描画用データの計算、保存
    if(k % 100 == 0):
        # 損失関数値の計算(7.6.1)
        loss = np.mean(yd ** 2) / 2
        # 計算結果の記録
        history = np.vstack((history, np.array([k, loss])))
        # 画面表示
        print("iter = %d loss = %f" % (k, loss))

# 最終的な損失関数の初期値、最終値
print('損失関数初期値: %f' % history[0, 1])
print('損失関数最終値: %f' % history[-1, 1])

# 学習曲線の表示(最初の10個分を除く)
plt.plot(history[:, 0], history[:, 1])
plt.savefig('Multiple_regression_model.png')