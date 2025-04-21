import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from t0 import plot_graph


# データセットの準備
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# データセットの確認
print("x_train shape:", x_train.shape) # (50000, 32, 32, 3) : 50000枚の32x32のRGB(3)画像
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# データセットの可視化
class_name = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]

def disp_data(x_data, y_data):
    plt.figure(figsize=(10, 10))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_data[i])
        plt.xlabel(class_name[y_data[i][0]])
    plt.show()
    plt.savefig("Chapter5/cifar10_data.png")
disp_data(x_train, y_train)


# モデルの構築
model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3))) # 32x32x3の画像を1次元に変換 -> 3072x1の1次元配列
model.add(layers.Dense(128, activation='relu')) # 隠れ層
model.add(layers.Dense(10, activation='softmax')) # 出力層　(0~9)の10クラス分類
model.summary() # モデルの概要を表示

# モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # 最適化手法にadam、損失関数にsparse_categorical_crossentropyを指定

history = model.fit(x_train, y_train, epochs=20, 
                    validation_data=(x_test, y_test),
                    verbose=0) # 学習を実行

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) # テストデータで評価
print(f"テストデータの損失: {test_loss:.2%}")
print(f"テストデータの正解率: {test_acc:.2%}")

# 学習過程の可視化
plot_graph(history, "Chapter5/cifar10_history.png")


# 予測の実行
pre = model.predict(x_test) # テストデータで予測

plt.figure(figsize=(12, 10))

for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    
    index = np.argmax(pre[i]) # 予測結果の最大値のインデックスを取得
    pct = pre[i][index] # 予測結果の最大値を取得
    ans = ""
    
    if index != y_test[i][0]:
        ans = "×--o["+class_name[y_test[i][0]]+"]"
    
    lbl = f"{class_name[index]}({pct:.0%}){ans}"
    
    plt.xlabel(lbl)
plt.show()
plt.savefig("Chapter5/cifar10_predict.png")