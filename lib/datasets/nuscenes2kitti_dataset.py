import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.nuscenes_calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import ipdb

class nuscenes2kittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='v1.0-trainval', sensor='CAM_FRONT'):
        self.split = split
        is_test = self.split == 'test'
        img_sets = split
        self.imageset_dir = os.path.join(root_dir, 'nuScenes2KITTI', 'v1.0-test' if is_test else 'v1.0-trainval')

        split_dir = os.path.join(root_dir, 'nuScenes2KITTI', 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_' + sensor)
        self.lidar_dir = os.path.join(self.imageset_dir, 'LIDAR_TOP')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_' + sensor)
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')
    def get_image(self, idx):
        assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.jpg' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.jpg' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
