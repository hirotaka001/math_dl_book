#!/usr/bin/env python
# 必須ライブラリの宣言
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from sklearn.datasets import load_boston

import sys,io
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
