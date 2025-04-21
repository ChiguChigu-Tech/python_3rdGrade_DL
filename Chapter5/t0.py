import matplotlib.pyplot as plt

def plot_graph(history, path=None):
    
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
    
    if path:
        plt.savefig(path)
    else:
        # Save the figure in a subdirectory named "Chapter4"
        plt.savefig('Chapter4/graph.png')
        
        # Uncomment the following line to save the figure in the current directory
                
        