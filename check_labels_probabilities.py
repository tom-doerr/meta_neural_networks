import os

import numpy as np
from argparse import ArgumentParser

def main(args):
    if not args.no_header:
        print("  subset\t   equal\t    diff\t   total")

    if args.train or args.all:
        check_files(args, 'train')
    if args.test or args.all:
        check_files(args, 'test')

def check_files(args, dataset_part):
    labels_path = os.path.join(args.labels_dir, '{}_{}.npy'.format(args.classifier, dataset_part))
    probabilities_path = os.path.join(args.probabilities_dir, '{}_{}.npy'.format(args.classifier, dataset_part))
    
    with open(labels_path, 'rb') as f:
        a = np.load(f)
    
    with open(probabilities_path, 'rb') as f:
        b = np.load(f)
    
    a2 = b.argmax(axis=-1)
    eq = a == a2
    eq_count = np.sum(eq)
    total_count = eq.shape[0]
    miss_count = total_count - eq_count
    print("{:>8s}".format(dataset_part), end='\t')
    print("\t".join(map(lambda x: "{:8d}".format(x), [eq_count, miss_count, total_count])))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='mobilenet_v2')
    parser.add_argument('--labels_dir', type=str, default='labels')
    parser.add_argument('--probabilities_dir', type=str, default='probabilities')
    parser.add_argument('--no_header', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    main(args)
