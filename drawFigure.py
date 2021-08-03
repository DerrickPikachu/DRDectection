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
        [0.7470016726573899, 0.7825901277625539, 0.7984981671945621, 0.81433502971636, 0.8274315811950603,
         0.8331613224669917, 0.843873447453646, 0.8537670379728816, 0.8618100288266486, 0.8756895263176625],
        [0.7331933520765863, 0.7350439517420548, 0.73507954019716, 0.73507954019716, 0.7350439517420548,
         0.73507954019716, 0.7350439517420548, 0.7350439517420548, 0.7350439517420548, 0.7351151286522652]
    ]
    test = [
        [0.7693950177935943, 0.7890391459074733, 0.7841992882562278, 0.7776512455516015, 0.760711743772242,
         0.7531672597864769, 0.7649822064056939, 0.7279715302491103, 0.7244128113879004, 0.706049822064057],
        [0.7335231316725979, 0.733238434163701, 0.7330960854092526, 0.7319572953736655, 0.7316725978647687,
         0.7293950177935943, 0.726405693950178, 0.7276868327402135, 0.7246975088967972, 0.7255516014234875]
    ]
    label = ['pretrain-Resnet18', 'no-pretrain-Resnet18']
    draw_multiline(10, train, test, label)
