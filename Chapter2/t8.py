from t0 import *
from t2 import and_test
from t3 import plot_perceptron
from t4 import or_test
from t5 import nand_test
from t7 import xor_test

plot_perceptron(and_test, x1, x2, save_path="Chapter2/graph/and_test.png")
plot_perceptron(or_test, x1, x2, save_path="Chapter2/graph/or_test.png")
plot_perceptron(nand_test, x1, x2, save_path="Chapter2/graph/nand_test.png")
plot_perceptron(xor_test, x1, x2, save_path="Chapter2/graph/xor_test.png")
