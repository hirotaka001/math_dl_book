# 必要ライブラリの宣言
import sys, io
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# sys.stdoutのエンコード変更
enc = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding=enc)

# PDF出力用
set_matplotlib_formats('png', 'pdf')

## 10.7 実装その１
# データ読み込み
mnist_file = 'mnist-original.mat'
mnist_path = 'mldata'
mnist_url = 'https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat'

# ファイルの存在確認
mnist_fullpath = os.path.join('.', mnist_path, mnist_file)
if not os.path.isfile(mnist_fullpath):
    # データダウンロード
    mldir = os.path.join('.', 'mldata')
    os.makedirs(mldir, exist_ok=True)
    print("download %s started." % mnist_file)
    urllib.request.urlretrieve(mnist_url, mnist_fullpath)
    print("download %s finished." % mnist_file)

mnist = fetch_mldata('MNIST original', data_home='.')

x_org, y_org = mnist.data, mnist.target

# 入力データの加工
# step1 データ正規化 値の範囲を[0, 1]とする
x_norm = x_org / 255.0

# 先頭にダミー変数(1)を追加
x_all = np.insert(x_norm, 0, 1, axis=1)

print('ダミー変数追加後', x_all.shape)

# step2 yをOne-hot-Vectorに
ohe = OneHotEncoder(spare=False)
y_all_one = ohe.fit_transform(np.c_[y_org])
print('One Hot Vectotor化後', y_all_one.shape)

# stap3 学習データ、検証データに分割
x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=60000, test_size=10000, shuffle=False
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, y_train_one.shape, y_test_one.shape)

# データ内容の確認
N = 20
np.random.seed(123)
indexes = np.random.choice(y_test.shape[0], N, replace=False)
x_selected = x_test[indexes, 1:]
y_selected = y_test[indexes]
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i+1)
    plt.imshow(x_selected[i].reshape(28, 28), cmap='gray_r')
    ax.set_title('%d' %y_selected[i], fontsize=16)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('data_chapter10.png')