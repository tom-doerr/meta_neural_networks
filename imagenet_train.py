import os
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from imagenet_module import ImageNet_Module

def main(hparams):
    
    seed_everything(0)
    
    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))
    
    # Model
    classifier = ImageNet_Module(hparams, pretrained=hparams.pretrained)  # target is passed in hparams to create the right module with loss function
    # IMPORTANT! Be sure to initialize target models with full dataset models on your own, if pretrained=True
    
    # Trainer
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("logs", name=hparams.classifier)
    trainer = Trainer(callbacks=[lr_logger], gpus=hparams.gpus, max_epochs=hparams.max_epochs,
                      deterministic=True, logger=logger)
    trainer.fit(classifier)

    # Load best checkpoint
    checkpoint_path = os.path.join(os.getcwd(), 'logs', hparams.classifier, 'version_' + str(classifier.logger.version),'checkpoints')
    classifier = ImageNet_Module.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    
    # Save weights from checkpoint
    if hparams.target >= 0:
        statedict_path = [os.getcwd(), 'imagenet_models', 'state_dicts', hparams.classifier, str(hparams.target) + '.pt']
        os.makedirs(os.path.join(*statedict_path[:-1]), exist_ok=True)
    else:
        statedict_path = [os.getcwd(), 'imagenet_models', 'state_dicts', hparams.classifier + '.pt']
    statedict_path = os.path.join(*statedict_path)
    torch.save(classifier.model.state_dict(), statedict_path)
    
    # Test model
    trainer.test(classifier)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='/hdd')
    parser.add_argument('--labels_dir', type=str, default='imagenet_labels')
    parser.add_argument('--probabilities_dir', type=str, default='imagenet_probabilities')
    parser.add_argument('--target', type=int, default=-1)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--gpus', default='0,') # use None to train on CPU
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=90)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_switch_func', action='store_true')
    parser.add_argument('--switch_threshold', type=float, default=0.10)
    args = parser.parse_args()
    main(args)
