"""
Copyright 2023 Xinrong Hu et al. https://github.com/xhu248/AutoSAM

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from loss_functions.nt_xent import NTXentLoss
import os
import shutil
import sys
import pickle

from loss_functions.supcon_loss import SupConSegLoss
from models.unet_con import GlobalConUnet, MLP

apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class ContrastExperiment(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config['save_dir'], 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        split_dir = os.path.join(self.config["base_dir"], "splits.pkl")
        data_dir = os.path.join(self.config["base_dir"], "preprocessed")

        with open(split_dir, "rb") as f:
            splits = pickle.load(f)

        fold = config["fold"]
        if fold == 5:
            tr_keys = splits[fold]['train']
            val_keys = splits[fold]['val']
        else:
            tr_keys = splits[fold]['train'] + splits[fold]['val']
            val_keys = splits[fold]['test']
        self.train_loader = NumpyDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["batch_size"],
                                         keys=tr_keys, do_reshuffle=True, mode='simclr', load_pseudo_label=True,
                                         pseudo_label_file=self.config["pseudo_label_file"])
        self.val_loader = NumpyDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["val_batch_size"],
                                       keys=val_keys, do_reshuffle=True, mode='simclr', load_pseudo_label=True,
                                       pseudo_label_file=self.config["pseudo_label_file"])

        print(len(self.train_loader))
        self.model = GlobalConUnet(num_classes=256)
        self.head = MLP(num_class=256)

        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])
        self.criterion = SupConSegLoss(temperature=0.5)

        # dist.init_process_group(backend='nccl')
        if torch.cuda.device_count() > 1:
            print("Let's use %d GPUs" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
            self.head = nn.DataParallel(self.head)

        self.model.to(self.device)
        self.head.to(self.device)

        self.model = self._load_pre_trained_weights(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, yp, n_iter):

        # get the representations and the projections
        zis = model(xis)  # [N,C]
        # zis = head(ris)

        # get the representations and the projections
        zjs = model(xjs)  # [N,C]
        # zjs = head(rjs)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        features = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)
        labels = torch.cat([yp.unsqueeze(1), yp.unsqueeze(1)], dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
                                                                    last_epoch=-1)

        for epoch_counter in range(self.config['epochs']):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, (xis, xjs) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                yp = xis['pseudo_label'][0].long().to(self.device)
                xis = xis['data'][0].float().to(self.device)
                xjs = xjs['data'][0].float().to(self.device)

                loss = self._step(self.model, xis, xjs, yp, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()
                self.optimizer.step()
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                #valid_loss = self._validate(self.val_loader)
                #print("Val:[{0}] loss: {loss:.4f}".format(epoch_counter, loss=valid_loss))
                #if valid_loss < best_valid_loss:
                    # save the model weights
                    # best_valid_loss = valid_loss
                torch.save(self.model.module.state_dict(), os.path.join(self.config['save_dir'],
                            'b_{}_f{}_model.pth'.format(self.config["batch_size"], self.config["fold"])))

                # self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
               scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                yp = xis['pseudo_label'][0].long().to(self.device)
                xis = xis['data'][0].float().to(self.device)
                xjs = xjs['data'][0].float().to(self.device)


                loss = self._step(self.model, xis, xjs, yp, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss
