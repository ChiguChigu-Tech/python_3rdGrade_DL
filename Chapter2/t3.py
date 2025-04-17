import numpy as np
import matplotlib.pyplot as plt
from t0 import *
from t2 import and_test

def fillsclorors(data):
    return "#ffc2c2" if data > 0 else "#c6dcec"
def dotscolors(data):
    return "#ff0e0e" if data > 0 else "#1f77b4"

def plot_perceptron(func, x1, x2, save_path=None):
    plt.figure(figsize=(6, 6))
    
    XX, YY = np.meshgrid(
        np.linspace(-0.25, 1.25, 200),
        np.linspace(-0.25, 1.25, 200))
    
    XX = np.array(XX).flatten()
    YY = np.array(YY).flatten()
    
    fills = []
    colors = []
    for i in range(len(XX)):
        fills.append(func(XX[i], YY[i]))
        colors.append(fillsclorors(func(XX[i], YY[i])))
    plt.scatter(XX, YY, c=colors)
    
    dots = []
    colors = []
    
    for i in range(len(x1)):
        dots.append(func(x1[i], x2[i]))
        colors.append(dotscolors(func(x1[i], x2[i])))
    plt.scatter(x1, x2, c=colors)
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    if save_path:
        plt.savefig(save_path)  # グラフを指定したパスに保存
    else:
        plt.show()  # 保存しない場合は表示 
    
plot_perceptron(and_test, x1, x2, save_path="Chapter2/graph/and_test.png")