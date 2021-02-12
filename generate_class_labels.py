import os, shutil

import numpy as np
import torch
from torch.nn.functional import softmax
from argparse import ArgumentParser
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cifar10_module import CIFAR10_Module

def main(hparams):
    if not hparams.no_gpu:
        # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
        if type(hparams.gpus) == str:
            if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
                torch.cuda.set_device(int(hparams.gpus[0]))
    else:
        hparams.gpus = None

    module = CIFAR10_Module(hparams, pretrained=True)
    if not hparams.no_gpu:
        model = module.model.cuda()

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform_dataset = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if hparams.all:
        hparams.train = True
        hparams.test = True

    if hparams.probabilities:
        folder = 'probabilities'
    else:
        folder = 'labels'

    if hparams.train:
        train_dataset = CIFAR10(hparams.data_dir, train=True, download=False, transform=transform_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)
        print('Evaluate for train dataset')
        labels = evaluate_for_dataset(module.model, train_dataloader, probabilities=hparams.probabilities, gpu=not hparams.no_gpu)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, '{}_{}.npy'.format(hparams.classifier, 'train'))
        save_labels(file_path, labels, probabilities=hparams.probabilities)

    if hparams.test:
        test_dataset = CIFAR10(hparams.data_dir, train=False, download=False, transform=transform_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)
        print('Evaluate for test dataset')
        labels = evaluate_for_dataset(module.model, test_dataloader, probabilities=hparams.probabilities, gpu=not hparams.no_gpu)
        file_path = os.path.join(folder, '{}_{}.npy'.format(hparams.classifier, 'test'))
        save_labels(file_path, labels, probabilities=hparams.probabilities)

def evaluate_for_dataset(model, dataloader, probabilities=False, gpu=True):
    result = []
    total = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # print 1, 10, 20, ..., last
            if i == 0 or i % 10 == 9 or i == total - 1:
                print('Batch {:5}/{}'.format(i+1, total))
            images, targets = batch
            if gpu:
                images = images.cuda()
            predictions = model(images)
            if probabilities:
                current_result = softmax(predictions, dim=-1).tolist()
            else:
                current_result = predictions.argmax(axis=1).tolist()
            result += current_result
    return result

def save_labels(file_path, labels, probabilities=False):
    with open(file_path, 'wb') as f:
        if probabilities:
            np.save(f, np.array(labels, dtype=np.float32))
        else:
            np.save(f, np.array(labels, dtype=np.int8))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target', type=int, default=-1)
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/data/huy/cifar10/')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--probabilities', action='store_true')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)
