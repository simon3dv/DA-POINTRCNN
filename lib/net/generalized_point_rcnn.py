import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg
import torch.nn.functional as F
import ipdb
class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply

class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr


class DAImgHead(nn.Module):
    def __init__(self, in_channels):
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv1d(in_channels, 128, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv1d(128, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        img_features = self.conv2_da(t) #B, 1, N
        return img_features

class DAInsHead(nn.Module):
    def __init__(self, in_channels):
        super(DAInsHead, self).__init__()

        self.conv1_da = nn.Conv1d(in_channels, 256, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.conv3_da = nn.Conv1d(256, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da, self.conv3_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        t = F.relu(self.conv2_da(t))
        ins_features = self.conv3_da(t)
        return ins_features

class da_rpn(torch.nn.Module):
    def __init__(self, cfg):
        super(da_rpn, self).__init__()

        self.cfg = cfg

        """
        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith(
            'V') else res2_out_channels * stage2_relative_factor
        """

        #self.img_weight = 1.0#cfg.DA.DA_IMG_LOSS_WEIGHT
        #self.ins_weight = cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        #self.cst_weight = cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0 * cfg.DA.DA_IMG.GRL_WEIGHT) #self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        #self.grl_ins = GradientScalarLayer(-1.0 * self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        #self.grl_img_consist = GradientScalarLayer(1.0 * self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        #self.grl_ins_consist = GradientScalarLayer(1.0 * self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)

        in_channels = 128#cfg.MODEL.BACKBONE.OUT_CHANNELS
        if cfg.DA.DA_IMG.RESHAPE:
            in_channels *= cfg.RPN.NUM_POINTS
        self.imghead = DAImgHead(in_channels)
        #self.inshead = DAInsHead(num_ins_inputs)
        #self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

        self.avgpool = torch.nn.AvgPool1d(16384, 1)
    def forward(self, img_features):
        """
        Arguments:
            img_features : B, 128, N

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if cfg.DA.DA_IMG.POOL:
            img_features = self.avgpool(img_features) # B,128,1
        if cfg.DA.DA_IMG.RESHAPE:
            img_features = img_features.reshape(img_features.shape[0], -1) #B, 128*N
        img_grl_fea = self.grl_img(img_features)
        da_img_features = self.imghead(img_grl_fea)

        #if self.resnet_backbone:
        #    da_ins_feature = self.avgpool(da_ins_feature)
        #da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
        #img_grl_fea = self.grl_img(img_features) #B, 128, N
        #ins_grl_fea = self.grl_ins(da_ins_feature)
        #img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        #ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        #da_img_features = self.imghead(img_grl_fea) #B, 1, N
        #da_ins_features = self.inshead(ins_grl_fea)
        #da_img_consist_features = self.imghead(img_grl_consist_fea)
        #da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        #da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        #da_ins_consist_features = da_ins_consist_features.sigmoid()
        """
        if self.training:
            da_img_loss = F.binary_cross_entropy_with_logits(
                da_img_features, da_img_labels
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            return losses
        return {}
        """
        output = {'da_img': da_img_features}
        return output

class da_rcnn(torch.nn.Module):
    def __init__(self, cfg):
        super(da_rcnn, self).__init__()

        self.cfg = cfg
        #self.ins_weight = 1.0#cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT

        self.grl_ins = GradientScalarLayer(-1.0 * cfg.DA.DA_INS.GRL_WEIGHT)

        num_ins_inputs = 512
        if cfg.DA.DA_INS.RESHAPE:
            num_ins_inputs *= 64
        self.inshead = DAInsHead(num_ins_inputs)

    def forward(self, ins_features):
        """
        Arguments:
            ins_features : B*64, 512, 1

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        ins_grl_fea = self.grl_ins(ins_features) #(B*64,512,1)
        if cfg.DA.DA_INS.RESHAPE:
            B = round(ins_grl_fea.shape[0]/64)
            ins_grl_fea = ins_grl_fea.reshape(B, -1, 1)
        da_ins_features = self.inshead(ins_grl_fea) #[B*64, 1, 1]
        """
        if self.training:
            da_img_loss = F.binary_cross_entropy_with_logits(
                da_img_features, da_img_labels
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            return losses
        return {}
        """
        output = {'da_ins': da_ins_features}
        return output

class GeneralizedPointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

        if cfg.DA.ENABLED:
            self.da_rpn = da_rpn(cfg)
            self.da_rcnn = da_rcnn(cfg)
    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
                if cfg.DA.ENABLED and cfg.DA.DA_IMG.ENABLED:
                    if cfg.RPN.FIXED:
                        self.da_rpn.eval()
                    da_rpn_output = self.da_rpn(rpn_output['backbone_features'])
                    output.update(da_rpn_output)
            """
            rpn_output:
                rpn_cls: B,N,1(Foreground Mask)
                rpn_reg: B,N,76(3D RoIs)
                backbone_xyz B,N,3(Point Coords)
                backbone_features: B,128,N(Semantic Features)
            """
            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)).contiguous(),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                
                if self.training:
                    rcnn_input_info['is_source'] = input_data['is_source'] # To avoid using gt when sampling
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

                if cfg.DA.ENABLED and cfg.DA.DA_INS.ENABLED and self.training:
                    da_rcnn_output = self.da_rcnn(output['l_features']) # l_features: [B*64, 512, 133]
                    output.update(da_rcnn_output)


        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
            ipdb.set_trace()
        else:
            raise NotImplementedError
        return output
