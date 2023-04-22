# Author: Nguyen Y Hop
import os
import time
import yaml
import math
import torch
import datetime
import torch.optim as optim
from src.loader.processes.img_process import Preproc
from torch.utils.data import DataLoader

from src.loader.processes.generate_box import PriorBox
from src.losses.multibox_loss import MultiBoxLoss
from src.loader.facedet_loader import FaceDataLoader, detection_collate
from src.models.architectures.base_model import RetinaFace

class Trainer:

    def __init__(self, config, **kwargs):
        self.global_config = config['Global']
        self.arch_config = config['Architecture']
        self.optim_config = config['Optimizer']
        self.criterion_config = config['Criterion']
        self.data_config = config['Dataloader']
        self.prior_config = config['PriorBox']
        self.save_config = config['SaveWeight']

        self.init_checkpoint()

        self.build_model()
        self.build_optimizer()
        self.build_criterion()
        self.build_dataloader()
        self.build_prior_box()


    def init_checkpoint(self):
        self.save_iter_dir = os.path.join(self.global_config['checkpoints'], "ITER")
        os.makedirs(self.save_iter_dir, exist_ok=True)
        self.save_epoch_dir = os.path.join(self.global_config['checkpoints'], "EPOCH")
        os.makedirs(self.save_epoch_dir, exist_ok=True)

    def build_model(self):
        self.model = RetinaFace(self.arch_config).cuda()
        if self.global_config['use_pretrain']:
            print('Loading resume network...')
            state_dict = torch.load(self.global_config['pretrain_path'])
            # create new OrderedDict that does not contain `module.`
            self.model.load_state_dict(state_dict)
            print('Loaded pretrain network...')
        return None


    def build_optimizer(self):
        self.lr = self.optim_config['lr']
        self.gamma = self.optim_config['gamma']
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, \
                                   momentum=self.optim_config['momentum'], weight_decay=self.optim_config['weight_decay'])
        
    
    def build_criterion(self):
        self.criterion = MultiBoxLoss(**self.criterion_config)

    def build_dataloader(self):
        
        self.dataset = FaceDataLoader(**self.data_config, preproc=Preproc(**self.data_config['Preproc']))


    def build_prior_box(self):
        img_size = self.data_config['Preproc']['image_size']
        priorbox = PriorBox(self.prior_config, image_size=(img_size, img_size))
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.cuda()


    def adjust_learning_rate(self, epoch, step_index, iteration, epoch_size):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warmup_epoch = -1
        if epoch <= warmup_epoch:
            lr = 1e-6 + (self.lr-1e-6) * iteration / (epoch_size * warmup_epoch)
        else:
            lr = self.lr * (self.gamma ** (step_index))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    
    def train(self):
        epoch = 0
        max_epoch = self.global_config['epoch']
        batch_size = self.global_config['batch_size']
        epoch_size = math.ceil(len(self.dataset) / batch_size)
        max_iter = max_epoch * epoch_size

        stepvalues = (self.optim_config['decay1'] * epoch_size, self.optim_config['decay2'] * epoch_size)

        if self.global_config['resume_epoch'] > 0:
            start_iter = self.global_config['resume_epoch'] * epoch_size
        else:
            start_iter = 0
        step_index = 0


        loc_weight = self.criterion_config['loc_weight']
            
        for iteration in range(start_iter, max_iter):
            if iteration % epoch_size == 0:
        
                batch_iterator = iter(DataLoader(self.dataset, batch_size, shuffle=True, num_workers=1, collate_fn=detection_collate))
                # if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                #     torch.save(self.model.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
                epoch += 1

            load_t0 = time.time()

            if iteration in stepvalues:
                step_index += 1
            lr = self.adjust_learning_rate(epoch, step_index, iteration, epoch_size)

            images, targets = next(batch_iterator)
            
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]


            out = self.model(images, training=True)
            self.optimizer.zero_grad()
            loss_l, loss_c, loss_landm = self.criterion(out, self.priors, targets)
            loss = loc_weight * loss_l + loss_c + loss_landm

            loss.backward()
            self.optimizer.step()
            load_t1 = time.time()

            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if iteration % 1000 == 0:
                print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

            if iteration % self.save_config['iter'] == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_iter_dir , f"iter_{iteration}.pth"))


        # backprop
        torch.save(self.model.state_dict(), os.path.join(self.save_epoch_dir , f"epoch_{epoch}.pth"))

        return None



if __name__ == '__main__':
    config = yaml.load(open('configs/cfg_retinaface_mobilenetv1.yml'))
    trainer = Trainer(config)
    trainer.train()