import numpy as np
import os

'''
def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
'''

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class Calibration(object):
    def __init__(self, calib_file, sensor_list = ['CAM_FRONT']):
        if isinstance(calib_file, str):
            calibs = read_calib_file(calib_file)
            #calib = get_calib_from_file(calib_file)
        else:
            calibs = calib_file

        self.sensor_list = sensor_list
        if 'CAM_FRONT' in self.sensor_list:
            self.CAM_FRONT = np.reshape(calibs['CAM_FRONT'], [3, 3])
            self.P2 = self.CAM_FRONT
        # self.P2 = calib['P2']  # 3 x 4
        #self.R0 = calib['R0']  # 3 x 3
        #self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = 0
        self.ty = 0
        #self.tx = self.P2[0, 3] / (-self.fu)
        #self.ty = self.P2[1, 3] / (-self.fv)

        self.lidar2ego_translation = np.reshape(calibs['lidar2ego_translation'], [3, 1])
        self.lidar2ego_rotation = np.reshape(calibs['lidar2ego_rotation'], [3, 3])
        self.ego2global_translation = np.reshape(calibs['ego2global_translation'], [3, 1])
        self.ego2global_rotation = np.reshape(calibs['ego2global_rotation'], [3, 3])
        for sensor in self.sensor_list:
            for m in [ 'cam2ego_translation','ego2global_translation']:
                attrt = sensor + '_'+ m
                exec('self.'+attrt+' = np.reshape(calibs["'+attrt+'"],[3,1])')
            for m in ['cam2ego_rotation','ego2global_rotation']:
                attrt = sensor + '_'+ m
                exec('self.'+attrt+' = np.reshape(calibs["'+attrt+'"],[3,3])')

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def view_points(self, points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
        """
        This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
        orthographic projections. It first applies the dot product between the points and the view. By convention,
        the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
        normalization along the third dimension.

        For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
        For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
        For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
         all zeros) and normalize=False

        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
            The projection should be such that the corners are projected onto the first 2 axis.
        :param normalize: Whether to normalize the remaining coordinate (along the third axis).
        :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def translate(self, points, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        pts = points.copy()
        for i in range(3):
            pts[i, :] = pts[i, :] + x[i]
        return pts

    def rotate(self, points, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        return np.dot(rot_matrix, points[:, :])

    # ====lidar - ego(lidar) - global - ego_cam - cam====
    def project_lidar_to_ego(self, pts_3d_velo):
        pts_3d_ego = self.rotate(pts_3d_velo, getattr(self, 'lidar2ego_rotation'))
        pts_3d_ego = self.translate(pts_3d_ego, getattr(self, 'lidar2ego_translation'))
        return pts_3d_ego

    def project_ego_to_lidar(self, pts_3d_ego):
        pts_3d_velo = self.translate(pts_3d_ego, -getattr(self, 'lidar2ego_translation'))
        pts_3d_velo = self.rotate(pts_3d_velo, getattr(self, 'lidar2ego_rotation').T)
        return pts_3d_velo

    def project_ego_to_global(self, pts_3d_ego):
        pts_3d_global = self.rotate(pts_3d_ego, getattr(self, 'ego2global_rotation'))
        pts_3d_global = self.translate(pts_3d_global, getattr(self, 'ego2global_translation'))
        return pts_3d_global

    def project_global_to_ego(self, pts_3d_global):
        pts_3d_ego = self.translate(pts_3d_global, -getattr(self, 'ego2global_translation'))
        pts_3d_ego = self.rotate(pts_3d_ego, getattr(self, 'ego2global_rotation').T)
        return pts_3d_ego

    def project_cam_to_ego(self, pts_3d_cam, sensor):
        pts_3d_ego_cam = self.rotate(pts_3d_cam, getattr(self, sensor + '_' + 'cam2ego_rotation'))
        pts_3d_ego_cam = self.translate(pts_3d_ego_cam, getattr(self,sensor+'_'+'cam2ego_translation'))
        return pts_3d_ego_cam

    def project_ego_to_cam(self, pts_3d_ego_cam, sensor):
        pts_3d_cam = self.translate(pts_3d_ego_cam, -getattr(self,sensor+'_'+'cam2ego_translation'))
        pts_3d_cam = self.rotate(pts_3d_cam, getattr(self, sensor + '_' + 'cam2ego_rotation').T)
        return pts_3d_cam

    def project_ego_to_global_cam(self, pts_3d_ego_cam, sensor):
        pts_3d_global_cam = self.rotate(pts_3d_ego_cam, getattr(self, sensor + '_' + 'ego2global_rotation'))
        pts_3d_global_cam = self.translate(pts_3d_global_cam, getattr(self,sensor+'_'+'ego2global_translation'))
        return pts_3d_global_cam

    def project_global_to_ego_cam(self, pts_3d_global_cam, sensor):
        pts_3d_ego_cam = self.translate(pts_3d_global_cam, -getattr(self,sensor+'_'+'ego2global_translation'))
        pts_3d_ego_cam = self.rotate(pts_3d_ego_cam, getattr(self, sensor + '_' + 'ego2global_rotation').T)
        return pts_3d_ego_cam

    # ====lidar - global - cam====
    def project_global_to_lidar(self, pts_3d_global):
        pts_3d_ego = self.project_global_to_ego(pts_3d_global)
        pts_3d_velo = self.project_ego_to_lidar(pts_3d_ego)
        return pts_3d_velo

    def project_lidar_to_global(self, pts_3d_velo):
        pts_3d_ego = self.project_lidar_to_ego(pts_3d_velo)
        pts_3d_global = self.project_ego_to_global(pts_3d_ego)
        return pts_3d_global

    def project_cam_to_global(self, pts_3d_cam, sensor):
        pts_3d_ego_cam = self.project_cam_to_ego(pts_3d_cam, sensor)
        pts_3d_global_cam = self.project_ego_to_global_cam(pts_3d_ego_cam, sensor)
        return pts_3d_global_cam

    def project_global_to_cam(self, pts_3d_global_cam, sensor):
        pts_3d_ego_cam = self.project_global_to_ego_cam(pts_3d_global_cam, sensor)
        pts_3d_cam = self.project_ego_to_cam(pts_3d_ego_cam, sensor)
        return pts_3d_cam


    #=========intrinsic=========#
    def project_image_to_cam(self, uv_depth, sensor):
        ''' Input: 3xn first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: 3xn points in (rect) camera coord.
        '''
        # Camera intrinsics and extrinsics
        c_u = getattr(self,sensor)[0,2]
        c_v = getattr(self,sensor)[1,2]
        f_u = getattr(self,sensor)[0,0]
        f_v = getattr(self,sensor)[1,1]
        n = uv_depth.shape[1]
        x = ((uv_depth[0,:]-c_u)*uv_depth[2,:])/f_u
        y = ((uv_depth[1,:]-c_v)*uv_depth[2,:])/f_v
        pts_3d_cam = np.zeros((3,n))
        pts_3d_cam[0,:] = x
        pts_3d_cam[1,:] = y
        pts_3d_cam[2,:] = uv_depth[2,:]
        return pts_3d_cam

    def project_cam_to_image(self, pts_3d_cam, sensor, return_depth=False):
        pts_2d = self.view_points(pts_3d_cam[:3, :], getattr(self,sensor), normalize=True)#(3,n)
        depth = pts_3d_cam[2,:]
        return pts_2d, depth


    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs


    def corners3d_to_img_boxes(self, corners3d, sensor='CAM_FRONT'):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        #corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        #img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)
        img_pts = np.empty((sample_num, 8, 3))
        for n in range(sample_num):
            pts_2d_cam = self.view_points(corners3d[n,:,:3].T, getattr(self, sensor), normalize=True)
            img_pts[n,:,:] = pts_2d_cam.T
        x, y = img_pts[:, :, 0], img_pts[:, :, 1]
        #x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu)**2 + (v - self.cv)**2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect
