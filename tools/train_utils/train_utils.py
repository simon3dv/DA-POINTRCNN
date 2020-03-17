import logging
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import tqdm
import torch.optim.lr_scheduler as lr_sched
import math
import numpy as np

logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=cur_logger):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


def load_part_ckpt(model, filename, logger=cur_logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


class Trainer(object):
    def __init__(self, model, model_fn, optimizer, ckpt_dir, lr_scheduler, bnm_scheduler,
                 model_fn_eval, tb_log, eval_frequency=1, lr_warmup_scheduler=None, warmup_epoch=-1,
                 grad_norm_clip=1.0):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler, self.model_fn_eval = \
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler, model_fn_eval

        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.warmup_epoch = warmup_epoch
        self.grad_norm_clip = grad_norm_clip

    def _train_it(self, batch):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item(), tb_dict, disp_dict

    def eval_epoch(self, d_loader):
        self.model.eval()

        target_eval_dict = {}
        total_loss = count = 0.0

        # eval one epoch
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()

            loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)

            total_loss += loss.item()
            count += 1
            for k, v in tb_dict.items():
                target_eval_dict[k] = target_eval_dict.get(k, 0) + v

        # statistics this epoch
        for k, v in target_eval_dict.items():
            target_eval_dict[k] = target_eval_dict[k] / max(count, 1)

        cur_performance = 0
        if 'recalled_cnt' in target_eval_dict:
            target_eval_dict['recall'] = target_eval_dict['recalled_cnt'] / max(target_eval_dict['gt_cnt'], 1)
            cur_performance = target_eval_dict['recall']
        elif 'iou' in target_eval_dict:
            cur_performance = target_eval_dict['iou']

        return total_loss / count, target_eval_dict, cur_performance

    """
    def collate_batch(self, cfg, batch):
        batch_size = batch.__len__()
        ans_dict = {}
        import ipdb
        ipdb.set_trace()

        for key in batch[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
    """
    def train(self, cfg, start_it, start_epoch, n_epochs, source_train_loader, source_test_loader,
              target_train_loader, target_test_loader, ckpt_save_interval=5,
              lr_scheduler_each_iter=False):
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1

        it = start_it
        with tqdm.trange(start_epoch, n_epochs, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(source_train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:
                if self.lr_scheduler is not None and self.warmup_epoch <= epoch and (not lr_scheduler_each_iter):
                    self.lr_scheduler.step(epoch)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)
                    self.tb_log.add_scalar('bn_momentum', self.bnm_scheduler.lmbd(epoch), it)

                # train one epoch
                for cur_it, (source_batch, target_batch) in enumerate(zip(source_train_loader, target_train_loader)):

                    batch = [source_batch, target_batch]
                    import ipdb
                    batch = {}
                    for key, value in source_batch.items():
                        if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                                (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                            max_gt = 0
                            batch_size = value.shape[0]
                            for k in range(batch_size):
                                max_gt = max(max_gt, source_batch[key][k].__len__())
                                max_gt = max(max_gt, target_batch[key][k].__len__())
                            batch_gt_boxes3d = np.zeros((batch_size*2, max_gt, 7), dtype=np.float32)
                            for i in range(batch_size):
                                batch_gt_boxes3d[i, :source_batch[key][i].__len__(), :] = source_batch[key][i]
                            for i in range(batch_size,batch_size*2):
                                batch_gt_boxes3d[i, :target_batch[key][i-batch_size].__len__(), :] = target_batch[key][i-batch_size]
                            batch[key] = batch_gt_boxes3d
                        elif type(value) == np.ndarray:
                            batch[key] = np.concatenate([source_batch[key], target_batch[key]], 0)
                        elif type(value) == list:
                            batch[key] = source_batch[key] + target_batch[key]
                        else:
                            ipdb.set_trace()

                    if lr_scheduler_each_iter:
                        self.lr_scheduler.step(it)
                        cur_lr = float(self.optimizer.lr)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                    else:
                        if self.lr_warmup_scheduler is not None and epoch < self.warmup_epoch:
                            self.lr_warmup_scheduler.step(it)
                            cur_lr = self.lr_warmup_scheduler.get_lr()[0]
                        else:
                            cur_lr = self.lr_scheduler.get_lr()[0]

                    loss, tb_dict, disp_dict = self._train_it(batch)
                    it += 1

                    disp_dict.update({'loss': loss, 'lr': cur_lr})

                    # log to console and tensorboard
                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()

                    if self.tb_log is not None:
                        self.tb_log.add_scalar('train_loss', loss, it)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                        for key, val in tb_dict.items():
                            self.tb_log.add_scalar('train_' + key, val, it)

                # save trained model
                trained_epoch = epoch + 1
                if trained_epoch % ckpt_save_interval == 0:
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(self.model, self.optimizer, trained_epoch, it), filename=ckpt_name,
                    )

                # eval one epoch
                if (epoch % eval_frequency) == 0:
                    pbar.close()
                    if target_test_loader is not None:
                        with torch.set_grad_enabled(False):
                            target_val_loss, target_eval_dict, cur_performance = self.eval_epoch(target_test_loader)

                        if self.tb_log is not None:
                            self.tb_log.add_scalar('target_val_loss', target_val_loss, it)
                            for key, val in target_eval_dict.items():
                                self.tb_log.add_scalar('target_val_' + key, val, it)

                pbar.close()
                pbar = tqdm.tqdm(total=len(source_train_loader), leave=False, desc='train')
                pbar.set_postfix(dict(total_it=it))

        return None
