import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import ResNet
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch
from parameter import *
from confusionMatrix import ConfusionMatrix

hyper_var = {
    'Adam': [0.001],
    'SGD': [0.001],
}


def evaluate(model, loader, loss):
    size = len(loader.dataset)
    model.eval()
    accumulate_loss = 0
    corrects = 0

    pred_list = []
    label_list = []

    for img, label in tqdm(loader):
        img = img.to(device)
        label = label.to(device)

        with torch.set_grad_enabled(False):
            pred = model(img.float())
            loss_val = loss(pred, label)

        for i in range(len(pred)):
            pred_list.append(pred[i].argmax().item())
            label_list.append(label[i].item())

        accumulate_loss += loss_val.item() * img.size(0)
        corrects += (pred.argmax(dim=1) == label).type(torch.long).sum().item()

    accumulate_loss /= size
    corrects /= size

    print(f'Acc: {corrects}, loss: {accumulate_loss}\n')
    return pred_list, label_list


if __name__ == "__main__":
    model = torch.load('model_80p.pth')
    model = model.to(device)
    preds, labels = evaluate(model, loader['test'], loss_fn)
    confusion_mat = ConfusionMatrix(preds, labels, 5)
    plt.figure(figsize=(5, 5))
    confusion_mat.plot(normalize=True)
    # test_data = RetinopathyLoader('data', 'test', ImgToTorch())
    # test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    # for model_type in ['pretrained', 'no_pretrained']:
    #     for opt, lr_list in hyper_var.items():
    #         for lr in lr_list:
    #             filename = f'{model_type}_{opt}_{lr}.pth'
    #             model = torch.load(filename)
    #             model.to(device)
    #
    #             if opt == 'Adam':
    #                 optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #             else:
    #                 optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #
    #             print(f'{model_type} {opt} lr={lr}')
    #             evaluate(model, test_loader, loss_fn)
