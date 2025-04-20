import japanize_matplotlib
import numpy as np
import keras
from keras import layers
from t0 import plot_graph 

hand_name = ["グー", "チョキ", "パー"]
judge_name = ["あいこ", "勝ち", "負け"]

# データの準備
hand_data = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
judge_data = [0, 1, 2, 2, 0, 1, 1, 2, 0]

x_train = x_test = np.array(hand_data)
y_train = y_test = np.array(judge_data)

print("学習データ(問題):")
print(x_train)
print(f"学習データ(答え):{y_train}")

# モデルの構築
model = keras.models.Sequential()
model.add(layers.Dense(8, activation='relu', input_dim=2))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary() # モデルの概要を表示

# モデルのコンパイル
model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',  # 損失関数
            metrics=['accuracy']) # metricsは評価指標

# モデルの学習
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), verbose=0) # verbose=0で学習過程を表示しない

# 学習結果の表示
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"テストデータの損失: {test_loss:.2%}")
print(f"テストデータの精度: {test_acc:.2%}")

plot_graph(history) # 学習結果をグラフで表示


# データの予測
pre = model.predict(x_test) # テストデータを使って予測
print("予測結果:")
for i in range(3):
    print(f"グー(0) : {pre[i][0]:.0%} チョキ(1) : {pre[i][1]:.0%} パー(2) : {pre[i][2]:.0%}")


# 予測結果の表示
for i in range(len(x_test)):
    print(f"手: {hand_name[x_test[i][0]]} {hand_name[x_test[i][1]]} => 判定: {judge_name[np.argmax(pre[i])]}")
    # np.argmax(pre[i])は予測結果の最大値のインデックスを取得