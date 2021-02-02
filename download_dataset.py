from argparse import ArgumentParser

from torchvision.datasets import CIFAR10

def main(args):
    if args.train:
        train_dataset = CIFAR10(root=args.data_dir, train=True, download=True)

    if args.test:
        test_dataset = CIFAR10(root=args.data_dir, train=False, download=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    args = parser.parse_args()
    main(args)
