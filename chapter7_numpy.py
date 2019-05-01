#!/usr/bin/env python
# 必須ライブラリの宣言
import sys,io
import numpy as np

# sys.stdoutのエンコード変更
enc = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding=enc)

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