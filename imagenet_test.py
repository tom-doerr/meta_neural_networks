import os, shutil
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from imagenet_module import ImageNet_Module



def main(hparams):
    if not hparams.no_gpu:
        # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
        if type(hparams.gpus) == str:
            if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
                torch.cuda.set_device(int(hparams.gpus[0]))
    else:
        hparams.gpus = None

    model = ImageNet_Module(hparams, pretrained=True)
    print(model)
    for name, module in model.named_modules():
        print(name)
    trainer = Trainer(gpus=hparams.gpus, default_root_dir=os.path.join(os.getcwd(), 'test_temp'))
    activation = {}
    def hook(model, input_, output):
        activation['output'] = output.detach()
    model.model.features[18][1].register_forward_hook(hook)
    trainer.test(model)
    shutil.rmtree(os.path.join(os.getcwd(), 'test_temp'))
    print(activation)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='mobilenet_v2')
    parser.add_argument('--data_dir', type=str, default='/hdd')
    parser.add_argument('--labels_dir', type=str, default='imagenet_labels')
    parser.add_argument('--probabilities_dir', type=str, default='imagenet_probabilities')
    parser.add_argument('--target', type=int, default=-1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)