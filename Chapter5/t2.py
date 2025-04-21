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
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3))) # 畳み込み層 (32, 5x5フィルタ), 入力は32x32x3(RGB)の画像
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # プーリング層 (2x2) : 画像を2x2の範囲に分割して縮小
model.add(layers.Dropout(0.2)) # ドロップアウト層 (20%) : 過学習を防ぐために、20%のニューロンをランダムに無効化
model.add(layers.Conv2D(64, (5, 5), activation='relu')) # 畳み込み層 (64, 5x5フィルタ)
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # プーリング層 (2x2)
model.add(layers.Dropout(0.2)) # ドロップアウト層 (20%)
model.add(layers.Flatten()) # 平坦化層　: 2次元の画像を1次元に変換
model.add(layers.Dense(64, activation='relu'))  # 全結合層 (64, relu) : 64個のニューロンを持つ
model.add(layers.Dropout(0.2)) # ドロップアウト層 (20%)
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # 出力層 (10, softmax) : 10クラス分類
# モデルの要約
model.summary(line_length=120) # モデルの要約を表示, line_length=120で行の長さを指定


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

# 中間層を可視化

# 中間層の名前を取得
hidden_layers = []

for i, val in enumerate(model.layers):
    print(f"{i}: {val.name}")
    hidden_layers.append(val.output)

hidden_model = keras.models.Model(inputs=model.inputs, outputs=hidden_layers) # 中間層のモデルを作成
# 中間層の出力を取得
hidden_out = hidden_model.predict(x_test) # テストデータで予測


# 中間層の出力を可視化
i = 10 # i番目の画像を可視化
plt.imshow(x_test[i]) # 元画像を表示
plt.xlabel(class_name[y_test[i][0]]) # 正解ラベルを表示
plt.show()
plt.savefig("Chapter5/cifar10_hidden.png") # 画像を保存

# 上記の画像が0番目の畳み込み層で、どのような特徴を捉えているかを可視化

def disp_hidden_data(data, w, path=None):
    plt.figure(figsize=(12, 8))
    num = data.shape[2] # データの次元数を取得
    for i in range(num):
        plt.subplot(int(num/w)+1, w, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data[:,:,i], cmap='Blues')    
    plt.savefig(path)

disp_hidden_data(hidden_out[0][i], 8, "Chapter5/cifar10_hidden_0.png") # 0 : 畳み込み層1(conv2d_1)

disp_hidden_data(hidden_out[1][i], 8, "Chapter5/cifar10_hidden_1.png") # 1 : プーリング層1(max_pooling2d_1)


