# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np

from tqdm import tqdm
# from mxnet import nd, autograd, gluon, io
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from core import Compute
# from utils import to_numpy, stack_numpy
# from diff_sample import normal
import os


def mmd_loss(x, y, device, t=0.1, kernel='diffusion'):
    """
    computes the mmd loss with information diffusion kernel
    :param x: batch_size x latent dimension
    :param y:
    :param t:
    :return:
    """
    eps = 1e-6
    n, d = x.shape
    if kernel == 'tv':
        sum_xx = torch.zeros(1, device=device)
        for i in range(n):
            for j in range(i + 1, n):
                sum_xx += torch.norm(x[i] - x[j], p=1)
        sum_xx = sum_xx / (n * (n - 1))

        sum_yy = torch.zeros(1, device=device)
        for i in range(y.shape[0]):
            for j in range(i + 1, y.shape[0]):
                sum_yy += torch.norm(x[i] - x[j], p=1)
        sum_yy = sum_yy / (y.shape[0] * (y.shape[0] - 1))

        sum_xy = torch.zeros(1, device=device)
        for i in range(n):
            for j in range(y.shape[0]):
                sum_xy += torch.norm(x[i] - x[j], p=1)
        sum_yy = sum_yy / (n * y.shape[0])
    else:
        qx = torch.sqrt(torch.clamp(x, eps, 1))
        qy = torch.sqrt(torch.clamp(y, eps, 1))
        xx = torch.mm(qx, qx.T)
        yy = torch.mm(qy, qy.T)
        xy = torch.mm(qx, qy.T)

        def diffusion_kernel(a, tmpt, dim):
            # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
            return torch.exp(- torch.square(torch.acos(a)) / tmpt)

        off_diag = 1 - torch.eye(n, device=device)
        k_xx = diffusion_kernel(torch.clamp(xx, 0, 1 - eps), t, d - 1)
        k_yy = diffusion_kernel(torch.clamp(yy, 0, 1 - eps), t, d - 1)
        k_xy = diffusion_kernel(torch.clamp(xy, 0, 1 - eps), t, d - 1)
        sum_xx = (k_xx * off_diag).sum() / (n * (n - 1))
        sum_yy = (k_yy * off_diag).sum() / (n * (n - 1))
        sum_xy = 2 * k_xy.sum() / (n * n)
    return sum_xx + sum_yy - sum_xy


class Unsupervised(Compute):
    '''
    Class to manage training, testing, and
    retrieving outputs.
    '''

    def __init__(self, data, Enc, Dec, Dis_y, args):
        '''
        Constructor.

        Args
        ----
        Returns
        -------
        Compute object
        '''
        super(Unsupervised, self).__init__(data, Enc, Dec, Dis_y, args)

    def unlabeled_train_op_mmd_combine(self, update_enc=True):
        '''
        Trains the MMD model
        '''
        batch_size = self.args['batch_size']
        device = self.device
        eps = 1e-10

        # Retrieve data
        docs = self.data.get_documents(key='train')
        if self.args['use_kd']:
            split_on = docs.shape[1] // 2
            docs, bert_logits = docs[:, :split_on], docs[:, split_on:]
            t = self.args['kd_softmax_temp']
            kd_docs = F.softmax(bert_logits / t) * torch.sum(docs, dim=1, keepdim=True)
            kd_docs = kd_docs * (kd_docs > self.args['kd_min_count'])

        y_true = np.random.dirichlet(np.ones(self.ndim_y) * self.args['dirich_alpha'], size=batch_size)
        y_true = torch.as_tensor(y_true, device=device)

        # with autograd.record():
        ### reconstruction phase ###
        y_onehot_u = self.Enc(docs)
        y_onehot_u_softmax = F.softmax(y_onehot_u)
        if self.args['latent_noise'] > 0:
            y_noise = np.random.dirichlet(np.ones(self.ndim_y) * self.args['dirich_alpha'], size=batch_size)
            y_noise = torch.as_tensor(y_noise, device=device)
            y_onehot_u_softmax = (1 - self.args['latent_noise']) * y_onehot_u_softmax + self.args[
                'latent_noise'] * y_noise
        x_reconstruction_u = self.Dec(y_onehot_u_softmax)

        if self.args['use_kd']:
            kd_logits = F.log_softmax(x_reconstruction_u / t)
            logits = F.log_softmax(x_reconstruction_u)

            kd_loss_reconstruction = torch.mean(torch.sum(- kd_docs * kd_logits, dim=1))
            loss_reconstruction = torch.mean(torch.sum(- docs * logits, dim=1))

            loss_total = self.args['recon_alpha'] * (
                    self.args['kd_loss_alpha'] * t * t * (kd_loss_reconstruction) +
                    (1 - self.args['kd_loss_alpha']) * loss_reconstruction
            )
        else:
            logits = F.log_softmax(x_reconstruction_u)
            loss_reconstruction = torch.mean(torch.sum(- docs * logits, dim=1))
            loss_total = loss_reconstruction * self.args['recon_alpha']

        ### mmd phase ###
        if self.args['adverse']:
            y_fake = self.Enc(docs)
            y_fake = F.softmax(y_fake)
            loss_mmd = mmd_loss(y_true, y_fake, device=device, t=self.args['kernel_alpha'])
            loss_total = loss_total + loss_mmd

        if self.args['l2_alpha'] > 0:
            loss_total = loss_total + self.args['l2_alpha'] * torch.mean(torch.sum(torch.square(y_onehot_u), dim=1))

        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()

        loss_total.backward()

        self.optimizer_enc.step()
        self.optimizer_dec.step()  # self.m.args['batch_size']

        latent_max = torch.zeros(self.args['ndim_y'], device=device)
        for max_ind in torch.argmax(y_onehot_u, dim=1):
            latent_max[max_ind] += 1.0
        latent_max /= batch_size
        latent_entropy = torch.mean(torch.sum(- y_onehot_u_softmax * torch.log(y_onehot_u_softmax + eps), dim=1))
        latent_v = torch.mean(y_onehot_u_softmax, dim=0)
        dirich_entropy = torch.mean(torch.sum(- y_true * torch.log(y_true + eps), dim=1))

        if self.args['adverse']:
            loss_mmd_return = loss_mmd.item()
        else:
            loss_mmd_return = 0.0
        return torch.mean(loss_reconstruction).item(), loss_mmd_return, latent_max.numpy(), latent_entropy.item(), \
               latent_v.numpy(), dirich_entropy.item()

    def retrain_enc(self, l2_alpha=0.1):
        docs = self.data.get_documents(key='train')
        # with autograd.record():
        ### reconstruction phase ###
        y_onehot_u = self.Enc(docs)
        y_onehot_u_softmax = F.softmax(y_onehot_u)
        x_reconstruction_u = self.Dec(y_onehot_u_softmax)

        logits = F.log_softmax(x_reconstruction_u)
        loss_reconstruction = torch.mean(torch.sum(- docs * logits, dim=1))
        loss_reconstruction = loss_reconstruction + l2_alpha * torch.mean(torch.norm(y_onehot_u, p=1, dim=1))

        self.optimizer_enc.zero_grad()

        loss_reconstruction.backward()

        self.optimizer_enc.step()
        return loss_reconstruction.item()

    def unlabeled_train_op_adv_combine_add(self, update_enc=True):
        '''
        Trains the GAN model
        '''
        batch_size = self.args['batch_size']
        device = self.device
        eps = 1e-10
        ##########################
        ### unsupervised phase ###
        ##########################
        # Retrieve data
        docs = self.data.get_documents(key='train')

        class_true = torch.zeros(batch_size, dtype='int32', device=device)
        class_fake = torch.ones(batch_size, dtype='int32', device=device)
        loss_reconstruction = torch.zeros((1,), device=device)

        ### adversarial phase ###
        discriminator_z_confidence_true = torch.zeros((1,), device=device)
        discriminator_z_confidence_fake = torch.zeros((1,), device=device)
        discriminator_y_confidence_true = torch.zeros((1,), device=device)
        discriminator_y_confidence_fake = torch.zeros((1,), device=device)
        loss_discriminator = torch.zeros((1,), device=device)
        dirich_entropy = torch.zeros((1,), device=device)

        ### generator phase ###
        loss_generator = torch.zeros((1,), device=device)

        ### reconstruction phase ###
        # with autograd.record():
        y_u = self.Enc(docs)
        y_onehot_u_softmax = F.softmax(y_u)
        x_reconstruction_u = self.Dec(y_onehot_u_softmax)

        logits = F.log_softmax(x_reconstruction_u)
        loss_reconstruction = torch.sum(- docs * logits, dim=1)
        loss_total = loss_reconstruction * self.args['recon_alpha']

        if self.args['adverse']:  # and np.random.rand()<0.8:
            y_true = np.random.dirichlet(np.ones(self.ndim_y) * self.args['dirich_alpha'], size=batch_size)
            y_true = torch.as_tensor(y_true, device=device)
            dy_true = self.Dis_y(y_true)
            dy_fake = self.Dis_y(y_onehot_u_softmax)
            discriminator_y_confidence_true = torch.mean(F.softmax(dy_true)[:, 0])
            discriminator_y_confidence_fake = torch.mean(F.softmax(dy_fake)[:, 1])
            softmaxCEL = torch.nn.CrossEntropyLoss()
            loss_discriminator = softmaxCEL(dy_true, class_true) + \
                softmaxCEL(dy_fake, class_fake)
            loss_generator = softmaxCEL(dy_fake, class_true)
            loss_total = loss_total + loss_discriminator + loss_generator
            dirich_entropy = torch.mean(torch.sum(- y_true * torch.log(y_true + eps), dim=1))

        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        self.optimizer_dis_y.zero_grad()

        loss_total.backward()

        self.optimizer_enc.step()
        self.optimizer_dec.step()
        self.optimizer_dis_y.step()

        latent_max = torch.zeros(self.args['ndim_y'], device=device)
        for max_ind in torch.argmax(y_onehot_u_softmax, dim=1):
            latent_max[max_ind] += 1.0
        latent_max /= batch_size
        latent_entropy = torch.mean(torch.sum(- y_onehot_u_softmax * torch.log(y_onehot_u_softmax + eps), dim=1))
        latent_v = torch.mean(y_onehot_u_softmax, dim=0)

        return torch.mean(loss_discriminator).item(), torch.mean(loss_generator).item(), \
            torch.mean(loss_reconstruction).item(), \
            torch.mean(discriminator_z_confidence_true).item(), \
            torch.mean(discriminator_z_confidence_fake).item(), \
            torch.mean(discriminator_y_confidence_true).item(), \
            torch.mean(discriminator_y_confidence_fake).item(), \
            latent_max.numpy(), latent_entropy.item(), latent_v.numpy(), dirich_entropy.item()

    def test_synthetic_op(self):
        batch_size = self.args['batch_size']
        dataset = 'train'
        num_samps = self.data.data[dataset].shape[0]
        batches = int(np.ceil(num_samps / batch_size))
        batch_iter = range(batches)
        enc_out = torch.zeros((batches * batch_size, self.ndim_y))
        for batch in batch_iter:
            # 1. Retrieve data
            if self.args['data_source'] == 'Ian':
                docs = self.data.get_documents(key=dataset)
            # 2. Compute loss
            y_onehot_u = self.Enc(docs)
            y_onehot_softmax = F.softmax(y_onehot_u)
            enc_out[batch * batch_size:(batch + 1) * batch_size, :] = y_onehot_softmax

        return enc_out

    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        '''
        Evaluates the model using num_samples.

        Args
        ----
        num_samples: integer, default None
          The number of samples to evaluate on. This is converted to
          evaluating on (num_samples // batch_size) minibatches.
        num_epochs: integer, default None
          The number of epochs to evaluate on. This used if num_samples
          is not specified. If neither is specified, defaults to 1 epoch.
        reset: bool, default True
          Whether to reset the test data index to 0 before iterating
          through and evaluating on minibatches.
        dataset: string, default 'test':
          Which dataset to evaluate on: 'valid' or 'test'.

        Returns
        -------
        Loss_u: float
          The loss on the unlabeled data.
        Loss_l: float
          The loss on the labeled data.
        Eval_u: list of floats
          A list of evaluation metrics on the unlabeled data.
        Eval_l: list of floats
          A list of evaluation metrics on the labeled data.
        '''
        batch_size = self.args['batch_size']
        device = self.device

        if num_samples is None and num_epochs is None:
            # assume full dataset evaluation
            num_epochs = 1

        if reset:
            # Reset Data to Index Zero
            if self.data.data[dataset] is not None:
                self.data.force_reset_data(dataset)
            if self.data.data[dataset + '_with_labels'] is not None:
                self.data.force_reset_data(dataset + '_with_labels')

        # Unlabeled Data
        u_loss = 'NA'
        u_eval = []
        if self.data.data[dataset] is not None:
            u_loss = 0
            if num_samples is None:
                num_samps = self.data.data[dataset].shape[0] * num_epochs
            else:
                num_samps = num_samples
            batches = int(np.ceil(num_samps / self.args['batch_size']))
            batch_iter = range(batches)
            if batches > 1:
                batch_iter = tqdm(batch_iter, desc='unlabeled')
            for batch in batch_iter:
                # 1. Retrieve data
                docs = self.data.get_documents(key=dataset)
                if self.args['use_kd']:
                    split_on = docs.shape[1] // 2
                    docs, bert_logits = docs[:, :split_on], docs[:, split_on:]
                    # TODO: below is not used, but also may not be necessary
                    t = self.args['kd_softmax_temp']
                    kd_docs = F.softmax(bert_logits / t) * torch.sum(docs, dim=1, keepdim=True)

                # 2. Compute loss
                y_u = self.Enc(docs)
                y_onehot_u_softmax = F.softmax(y_u)
                x_reconstruction_u = self.Dec(y_onehot_u_softmax)

                logits = F.log_softmax(x_reconstruction_u)
                loss_recon_unlabel = torch.sum(- docs * logits, dim=1)

                # 3. Convert to numpy
                u_loss += torch.mean(loss_recon_unlabel).item()
            u_loss /= batches

        # Labeled Data
        l_loss = 0.0
        l_acc = 0.0
        if self.data.data[dataset + '_with_labels'] is not None:
            l_loss = 0
            if num_samples is None:
                num_samps = self.data.data[dataset + '_with_labels'].shape[0] * num_epochs
            else:
                num_samps = num_samples
            batches = int(np.ceil(num_samps / self.args['batch_size']))
            batch_iter = range(batches)
            if batches > 1:
                batch_iter = tqdm(batch_iter, desc='labeled')

            # softmaxCEL = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
            def cross_entropy_soft_targets(pred, soft_targets):
                return torch.sum(- soft_targets * F.log_softmax(pred, dim=1), dim=1)
            softmaxCEL = cross_entropy_soft_targets

            for batch in batch_iter:
                # 1. Retrieve data
                labeled_docs, labels = self.data.get_documents(key=dataset + '_with_labels',
                                                               split_on=self.data.data_dim)
                # 2. Compute loss
                y_u = self.Enc(docs)
                y_onehot_u_softmax = F.softmax(y_u)
                class_pred = torch.argmax(y_onehot_u_softmax, dim=1)
                l_a = labels[list(range(labels.shape[0])), class_pred]
                l_acc += torch.mean(l_a).item()
                labels = labels / torch.sum(labels, dim=1, keepdim=True)
                l_l = softmaxCEL(y_onehot_u_softmax, labels)

                # 3. Convert to numpy
                l_loss += torch.mean(l_l).item()
            l_loss /= batches
            l_acc /= batches

        return u_loss, l_loss, l_acc

    def save_latent(self, saveto):
        before_softmax = True
        try:
            if type(self.data.data['train']) is np.ndarray:
                dataset_train = TensorDataset(self.data.data['train'])
                train_data = DataLoader(dataset_train, self.args['batch_size'], shuffle=False,
                                        drop_last=True)

                dataset_val = TensorDataset(self.data.data['valid'])
                val_data = DataLoader(dataset_val, self.args['batch_size'], shuffle=False,
                                      drop_last=True)

                dataset_test = TensorDataset(self.data.data['test'])
                test_data = DataLoader(dataset_test, self.args['batch_size'], shuffle=False,
                                       drop_last=True)
            else:
                print("Exception in Unsupervised().save_latent() in compute_op.py")
                return
        except:
            print("Loading error during save_latent. Probably caused by not having validation or test set!")
            return

        train_output = np.zeros((self.data.data['train'].shape[0], self.ndim_y))
        # train_label_output = np.zeros(self.data.data['train'].shape[0])
        # for i, (data, label) in enumerate(train_data):
        for i, data in enumerate(train_data):
            data = data.to(self.device)
            if before_softmax:
                output = self.Enc(data)
            else:
                output = F.softmax(self.Enc(data))
            train_output[i * self.args['batch_size']: (i + 1) * self.args['batch_size']] = output.numpy()
            # train_label_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = label.asnumpy()
        train_output = np.delete(train_output, np.s_[(i + 1) * self.args['batch_size']:], 0)
        # train_label_output = np.delete(train_label_output, np.s_[(i+1)*self.args['batch_size']:])
        np.save(os.path.join(saveto, self.args['domain'] + 'train_latent.npy'), train_output)
        # np.save(os.path.join(saveto, self.args['domain']+'train_latent_label.npy'), train_label_output)

        val_output = np.zeros((self.data.data['valid'].shape[0], self.ndim_y))
        # train_label_output = np.zeros(self.data.data['train'].shape[0])
        # for i, (data, label) in enumerate(train_data):
        for i, data in enumerate(val_data):
            data = data.to(self.device)
            if before_softmax:
                output = self.Enc(data)
            else:
                output = F.softmax(self.Enc(data))
            val_output[i * self.args['batch_size']: (i + 1) * self.args['batch_size']] = output.numpy()
            # train_label_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = label.asnumpy()
        val_output = np.delete(val_output, np.s_[(i + 1) * self.args['batch_size']:], 0)
        # train_label_output = np.delete(train_label_output, np.s_[(i+1)*self.args['batch_size']:])
        np.save(os.path.join(saveto, self.args['domain'] + 'val_latent.npy'), val_output)
        # np.save(os.path.join(saveto, self.args['domain']+'train_latent_label.npy'), train_label_output)

        test_output = np.zeros((self.data.data['test'].shape[0], self.ndim_y))
        # test_label_output = np.zeros(self.data.data['test'].shape[0])
        # for i, (data, label) in enumerate(test_data):
        for i, data in enumerate(test_data):
            data = data.to(self.device)
            if before_softmax:
                output = self.Enc(data)
            else:
                output = F.softmax(self.Enc(data))
            test_output[i * self.args['batch_size']:(i + 1) * self.args['batch_size']] = output.numpy()
            # test_label_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = label.asnumpy()
        test_output = np.delete(test_output, np.s_[(i + 1) * self.args['batch_size']:], 0)
        # test_label_output = np.delete(test_label_output, np.s_[(i+1)*self.args['batch_size']:])
        np.save(os.path.join(saveto, self.args['domain'] + 'test_latent.npy'), test_output)
        # np.save(os.path.join(saveto, self.args['domain']+'test_latent_label.npy'), test_label_output)
