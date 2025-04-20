import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers


# データの準備
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train , x_test = x_train / 255.0, x_test / 255.0 #正規化

print(f"学習データ(問題画像): {x_train.shape}") # データの形状を表示, 28x28の画像が60000枚
print(f"テストデータ(問題画像): {x_test.shape}")


# モデルの構築
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)), # 28x28の画像を784次元のベクトルに変換 = 28x28の2次元データのままでは入力できない。そこで、layers.Flatten()を使って1次元配列に変換してから入力する。 28x28=784x1の1次元配列に変換
    layers.Dense(128, activation='relu'), # 隠れ層
    layers.Dense(10, activation='softmax') # 出力層 : その画像が0~9のどれかに分類されるので、出力は10次元ベクトル
])

model.summary() # モデルの概要を表示

# モデルのコンパイル
model.compile(optimizer='adam', # 最適化手法
                loss='sparse_categorical_crossentropy', # 損失関数
                metrics=['accuracy']) # 評価指標

# モデルの学習
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0) # epochs: 学習回数, validation_data: テストデータを指定, verbose: 進捗を表示するかどうか

# 学習結果の表示
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"テストデータの損失: {test_loss:.2%}")
print(f"テストデータの正解率: {test_acc:.2%}")



param = [['正解率', 'accuracy', 'val_accuracy'], ["誤差", 'loss', 'val_loss']]
    
plt.figure(figsize=(10, 4))
    
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-", label=param[i][1])
    plt.plot(history.history[param[i][2]], "o-", label=param[i][2])
    plt.xlabel('epoch(学習回数)')
    plt.legend(["train", "test"], loc="best")
    if i == 0:
        plt.ylim(0, 1)
    plt.show()
    
    plt.savefig('Chapter4/t3.png')
    
# 予測
pre = model.predict(x_test) # テストデータを使って予測

i = 0 # 予測したい画像のインデックス
plt.imshow(x_test[i], cmap='gray') # 画像を表示
plt.show() # 画像を表示

index = np.argmax(pre[i]) # 予測結果のインデックスを取得
pct = pre[i][index] # 予測結果の確率を取得
print(f"予測結果: {index} ({pct:.2%})") # 予測結果を表示
print(f"正解: {y_test[i]}") # 正解を表示

# 予測(複数)

plt.figure(figsize=(12, 10))

for i in range(10):
    plt.subplot(4, 5, i + 1) 
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap='gray') # 画像を表示
    
    index = np.argmax(pre[i]) # 予測結果のインデックスを取得
    pct = pre[i][index] # 予測結果の確率を取得
    ans=""
    
    if index != y_test[i]:
        ans = "×--o[" + str(y_test[i]) + "]"
        
    lbl = f"予測: {index} ({pct:.2%})\n正解: {y_test[i]} {ans}"
    plt.xlabel(lbl) # ラベルを表示

plt.show() # 画像を表示
plt.savefig('Chapter4/t3_2.png')