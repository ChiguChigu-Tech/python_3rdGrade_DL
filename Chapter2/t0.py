x1 = [0,1,0,1]
x2 = [0,0,1,1]


def dis_results(func):
    
    for i in range(4):
        Y = func(x1[i], x2[i])
        print(f"X1: {x1[i]}, X2: {x2[i]} => Y: {Y}")

# There are function of display results and Data list