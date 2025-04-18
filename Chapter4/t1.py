import japanize_matplotlib
import numpy as np
import keras
from keras import layers
from t0 import plot_graph  # plot_graph関数をインポート


input_data = [[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]

xor_data = [0, 1, 1, 0]

# Kerasで使うためにはリストをNumPy配列に変換する必要がある
x_train = x_test = np.array(input_data) 
y_train = y_test = np.array(xor_data)

print("学習データ(問題):")
print(x_train)
print(f"学習データ(答え):{y_train}")


"""_summary_
# 3つのニューロンでできた層(活性化関数はReLU, 入力が2つの層付き)
model = keras.models.Sequential() # モデルの入れ物を作成(Sequentialは順番に層を積み重ねる)
model.add(layers.Dense(3, activation='relu', input_dim=2)) 
# layers.Dense(ニューロン数, 活性化関数)は全結合層といい、次の層にあるすべてのニューロンと結合する層のこと
# また、この層ははじめての層なので、input_dimで全結合層の前に入力層を作成している
"""

model = keras.models.Sequential()
model.add(layers.Dense(8, activation='relu', input_dim=2)) # 8個のニューロンを持つ層　, 入力層を追加
model.add(layers.Dense(8, activation='relu')) # 8個のニューロンを持つ層を追加
model.add(layers.Dense(2, activation='sigmoid')) # 出力層を追加, 出力の値を確率に変換するためにsigmoid関数を使用
model.summary() # モデルの概要を表示
# Total paramsは学習するパラメータの総数を表す
# 1層目のパラメータ数は(2+1)*8=24, 2層目は(8+1)*8=72, 出力層は(8+1)*2=18
# 上記はそれぞれ、一つ前の層のニューロン数+1(バイアス, 重み)と着目した層のニューロン数を掛け算したもの

# モデルのコンパイル
model.compile(optimizer='adam', # 最適化手法にadamを指定
                loss='sparse_categorical_crossentropy', # 損失関数にspares_categorical_crossentropyを指定
                metrics=['accuracy']) # 評価指標にaccuracyを指定
# モデルの学習
history = model.fit(x_train, y_train, # 学習データを指定
                    epochs=500, # 学習回数を指定
                    validation_data=(x_test, y_test), # テストデータを指定
                    verbose=0 # 学習の進捗を表示しない
                    ) # model.fitは学習を実行する関数

# 学習結果の表示
test_loss, test_acc = model.evaluate(x_test, y_test) # テストデータでモデルを評価
print(f"テストデータの損失:{test_loss}") 
print(f"テストデータの正解率:{test_acc:.2%}") # テストデータの正解率を表示

plot_graph(history) # 学習結果をグラフで表示

# データの予測
pre = model.predict(x_test) # テストデータを使って予測
print("予測結果:")
print(pre) # 予測結果を表示


# わかりやすくするために、予測結果を0~1の範囲に変換
for i in range(4):
    index = np.argmax(pre[i]) # 予測結果の中で最大値のインデックスを取得
    print(f"入力データ:{x_test[i]}, 予測結果:{index}") # 入力データと予測結果を表示