import copy
import torch
from parameter import device


def train_model(model, dataloader, loss_fn, optimizer, epochs):
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.

    for e in range(1, epochs + 1):
        print('Epoch: {}/{}'.format(e, epochs))
        print('-' * 10)

        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            accumulate_loss = 0.
            corrects = 0
            size = len(dataloader[mode].dataset)

            for batch, (img, label) in enumerate(dataloader[mode]):
                img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    pred = model(img.float())
                    loss = loss_fn(pred, label)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                accumulate_loss += loss.item()
                corrects += (pred.argmax(dim=1) == label).type(torch.long).sum().item()

                if batch % 100 == 0:
                    print('batch[{}] Loss: {:.4f}'.format(batch, loss.item()))

            epoch_acc = corrects / size
            epoch_mean_loss = accumulate_loss / size

            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(mode, epoch_mean_loss, epoch_acc))

            if mode == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

        print('-' * 10 + '\n')

    print('Finish')
    print('Best accuracy: {:4f}%'.format(best_acc))

    model.load_stat_dict(best_weights)
    return model
