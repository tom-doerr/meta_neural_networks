import os
from collections import defaultdict
import numpy as np
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
from torchvision.datasets import CIFAR10
# from cifar10_module import CIFAR10_Module

def switch_func(x, total_classes, switch_threshold=0.25, **kwargs):
    picked_labels = torch.arange(total_classes).view(1, -1).repeat(x.shape[0], 1)
    picked_labels[x < switch_threshold] = -1  # 0.25 work well
    return picked_labels  # > 0.1

class CIFAR10Class(CIFAR10):
    def __init__(self, *args, labels_file='labels/mobilenet_v2_train.npy', probabilities_file='probabilities/mobilenet_v2_train.npy', target=-1, use_switch_func=False, switch_kwargs=None, **kwargs):
        super(CIFAR10Class, self).__init__(*args, **kwargs)
        if switch_kwargs is None:
            switch_kwargs = {}
        if use_switch_func:
            with open(probabilities_file, 'rb') as f:
                self.probabilities = np.load(f)
            probs = torch.tensor(self.probabilities)
            switch_indicies = switch_func(probs, probs.shape[1], **switch_kwargs)
            self.switch_indicies = (switch_indicies - target == 0).sum(dim=1).bool().numpy()
            self.data_indexes = np.where(self.switch_indicies)[0]
            self.target_data = self.data[self.data_indexes]
            self.target_labels = np.array(self.targets)[self.data_indexes]
        else:
            with open(labels_file, 'rb') as f:
                self.labels = np.load(f)
            self.data_indexes = np.where(self.labels == target)[0]
            self.target_data = self.data[self.data_indexes]
            self.target_labels = np.array(self.targets)[self.data_indexes]

        unique, counts = np.unique(self.target_labels, return_counts=True)
        class_count = defaultdict(int, zip(unique, counts))
        total_classes = 10
        self.classes_count = np.array([class_count[i] for i in range(total_classes)])
        num_dims = len(self.target_data.shape)
        axis = tuple(range(num_dims - 1))
        self.mean = self.target_data.mean(axis=axis) / 255  # all but last dimension
        self.std = self.target_data.std(axis=axis) / 255  # all but last dimension

    def __len__(self):
        return self.data_indexes.shape[0]
        # return super().__len__()

    def __getitem__(self, idx):
        img, target = self.target_data[idx], self.target_labels[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return super().__getitem__(idx)


class ClassLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, target=-1, **kwargs):
        super(ClassLoss, self).__init__(*args, **kwargs)
        self.target = torch.tensor(target, dtype=torch.float)
        self.target_tensor = torch.zeros(10)
        self.target_tensor[target] = 1

    def forward(self, input, target):
        target = torch.min(torch.abs(self.target - target), 1)  # 0 - coincide, 1 - different
        pass


def main(hparams):
    return  # do nothing

    seed_everything(0)

    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))

    # Model
    classifier = CIFAR10_Module(hparams)

    # Trainer
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("logs", name=hparams.classifier)
    trainer = Trainer(callbacks=[lr_logger], gpus=hparams.gpus, max_epochs=hparams.max_epochs,
                      deterministic=True, logger=logger)
    trainer.fit(classifier)

    # Load best checkpoint
    checkpoint_path = os.path.join(os.getcwd(), 'logs', hparams.classifier, 'version_' + str(classifier.logger.version),'checkpoints')
    classifier = CIFAR10_Module.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))

    # Save weights from checkpoint
    statedict_path = os.path.join(os.getcwd(), 'cifar10_models', 'state_dicts', hparams.classifier + '.pt')
    torch.save(classifier.model.state_dict(), statedict_path)

    # Test model
    trainer.test(classifier)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/data/huy/cifar10/')
    parser.add_argument('--gpus', default='0,') # use None to train on CPU
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)
