import os, shutil

import numpy as np
from argparse import ArgumentParser
from torchvision.datasets import CIFAR10

def main(hparams):
    if not hparams.no_header:
        print('train acc\ttest acc\ttotal acc')
    train_dataset = CIFAR10(hparams.data_dir, train=True, download=False)
    file_path = os.path.join('labels', '{}_{}.npy'.format(hparams.classifier, 'train'))
    labels = load_labels(file_path)
    targets = np.array(train_dataset.targets, dtype=np.uint8)
    eq = labels == targets
    train_true_preds = np.sum(eq)
    train_total_len = len(targets)
    train_acc = train_true_preds / train_total_len
    print('{:8.2%}'.format(train_acc), end='\t')

    test_dataset = CIFAR10(hparams.data_dir, train=False, download=False)
    file_path = os.path.join('labels', '{}_{}.npy'.format(hparams.classifier, 'test'))
    labels = load_labels(file_path)
    targets = np.array(test_dataset.targets, dtype=np.uint8)
    eq = labels == targets
    test_true_preds = np.sum(eq)
    test_total_len = len(targets)
    test_acc = test_true_preds / test_total_len
    print('{:8.2%}'.format(test_acc), end='\t')

    total_acc = (train_true_preds + test_true_preds) / (
                 train_total_len + test_total_len)
    print('{:8.2%}'.format(total_acc))


def load_labels(file_path):
    with open(file_path, 'rb') as f:
        labels = np.load(f)
    return labels

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/data/huy/cifar10/')
    parser.add_argument('--no_header', action='store_true')
    args = parser.parse_args()
    main(args)
