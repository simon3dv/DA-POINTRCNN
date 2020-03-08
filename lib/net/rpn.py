import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.rpn.proposal_layer import ProposalLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
import importlib


class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')

        MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        self.backbone_net = MODEL.get_model(input_channels=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)

        # classification branch
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.Sequential(*cls_layers)

        # regression branch
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        if cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer = nn.Sequential(*reg_layers)

        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0],
                                                                               gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError

        self.proposal_layer = ProposalLayer(mode=mode)
        self.init_weights()

    def init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))

        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input']
        backbone_xyz, backbone_features = self.backbone_net(pts_input)  # (B, N, 3), (B, C, N)

        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}

        return ret_dict

if __name__ == '__main__':
    from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
    cfg_file = 'tools/cfgs/default.yaml'
    cfg_from_file(cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(cfg_file))[0]

    train_mode = 'rcnn'
    if train_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
    elif train_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    elif train_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    else:
        raise NotImplementedError


    mode = 'TRAIN'
    DATA_PATH = os.path.join('data')

    import logging
    def create_logger(log_file):
        log_format = '%(asctime)s  %(levelname)5s  %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger(__name__).addHandler(console)
        return logging.getLogger(__name__)

    root_result_dir = os.path.join('output', 'rcnn', cfg.TAG)
    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    """
    dataset = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=False,
                                rcnn_eval_roi_dir=None,
                                rcnn_eval_feature_dir=None,
                                classes=cfg.CLASSES,
                                logger=logger)
    dataset[0]
    """
    from lib.datasets.kitti_rcnn_dataset import  KittiRCNNDataset
    train_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.SPLIT, mode='TRAIN',
                                 logger=logger,
                                 classes=cfg.CLASSES,
                                 rcnn_training_roi_dir=None,
                                 rcnn_training_feature_dir=None,
                                 gt_database_dir='tools/gt_database/train_gt_database_3level_Car.pkl')

    import ipdb
    ipdb.set_trace()
    train_set[0]


    import torch
    from lib.net.rcnn_net import RCNNNet
    import ipdb
    training = True
    output = {}
    rpn = RPN(use_xyz=True, mode=mode).cuda()
    rcnn_net = RCNNNet(num_classes=2, input_channels=128, use_xyz=True)

    pts = torch.zeros((2,1000,3)).float().cuda()
    gt_boxes = torch.zeros((2,cfg.RPN.NUM_POINTS,8)).float().cuda()
    input_data = {'pts_input':pts,
                  'gt_boxes3d':gt_boxes}

    rpn_output = rpn(input_data)

    output.update(rpn_output)

    with torch.no_grad():
        rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
        backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

        rpn_scores_raw = rpn_cls[:, :, 0]
        rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
        seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
        pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

        # proposal layer
        rois, roi_scores_raw = rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
        output['rois'] = rois
        output['roi_scores_raw'] = roi_scores_raw
        output['seg_result'] = seg_mask

    rcnn_input_info = {'rpn_xyz': backbone_xyz,#B,N,3
                       'rpn_features': backbone_features.permute((0, 2, 1)),#B,N,128
                       'seg_mask': seg_mask,#B,N
                       'roi_boxes3d': rois,#B,M,7
                       'pts_depth': pts_depth}#B,N
    if training:
        rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']
    rcnn_output = rcnn_net(rcnn_input_info)
    ipdb.set_trace()

    output.update(rcnn_output)