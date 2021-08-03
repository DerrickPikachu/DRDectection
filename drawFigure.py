import matplotlib.pyplot as plt
import numpy as np


def draw_multiline(epochs, train_accuracy, test_accuracy, labels):
    x_axis = np.arange(epochs) + 1
    num_of_line = len(train_accuracy)
    for i in range(num_of_line):
        plt.plot(x_axis, train_accuracy[i], label=f'train({labels[i]})')
        plt.plot(x_axis, test_accuracy[i], label=f'test({labels[i]})', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train = [
        [0.7461119612797609, 0.7769671518559379, 0.7881063383038542, 0.7952596177799922, 0.8058293889462258],
        [0.7169294280935264, 0.7330509982561657, 0.7345813018256877, 0.734866009466529, 0.7349727748318445],
    ]
    test = [
        [0.7646975088967971, 0.7834875444839857, 0.7887544483985766, 0.7864768683274022, 0.7770818505338078],
        [0.7335231316725979, 0.7335231316725979, 0.7335231316725979, 0.7333807829181495, 0.6424199288256228]
    ]
    label = ['pretrain-Resnet50', 'no-pretrain-Resnet50']
    draw_multiline(5, train, test, label)
