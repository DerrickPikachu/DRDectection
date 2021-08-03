import itertools

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class ConfusionMatrix:
    def __init__(self, pred, target, classes):
        self.cm = confusion_matrix(pred, target)
        self.classes = np.arange(classes)

    def plot(self, normalize=False, title='Normalized Confusion Matrix', cmap=plt.cm.Blues):
        if normalize:
            counter = self.cm.sum(axis=1)
            self.cm = self.cm.astype('float')
            for i in range(len(counter)):
                self.cm[i] = self.cm[i] / counter[i]
        print(self.cm)
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f' if normalize else 'd'
        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, format(self.cm[i][j], fmt), horizontalalignment='center',
                     color='white' if self.cm[i][j] > thresh else 'black')

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusionMat.png', bbox_inches='tight')


if __name__ == "__main__":
    pred = [0, 1, 2, 1, 0, 4, 3, 2, 3, 4]
    target = [0, 1, 0, 1, 0, 1, 3, 2, 1, 1]
    classes = 5
    confusion_mat = ConfusionMatrix(pred, target, classes)
    # plt.figure(figsize=(10, 10))
    plt.figure(figsize=(5, 5))
    confusion_mat.plot(normalize=True)