import os
import cv2
import json
from copy import deepcopy
import torch
import numpy as np
from scene.dataset_readers import convert_to_scene_cameras
from model import libcore
from model.smplx_utils import smplx_utils
import pytorch3d.structures.meshes as py3d_meshes

def read_instant_avatar_frameset(dat_dir, frm_idx, cam, extension='.png'):
    image_path = os.path.join(dat_dir, f'images/image_{frm_idx:04d}.png')
    mask_path = os.path.join(dat_dir, f'masks/mask_{frm_idx:04d}.png')
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    image = np.concatenate([image, mask[:, :, None]], axis=-1)

    color_frames = libcore.DataVec()
    color_frames.cams = [cam]
    color_frames.frames = [image]
    color_frames.images_path = [image_path]
    return color_frames

class InstantAvatarSmplxDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        self.dat_dir = config.dat_dir
        self.cameras_extent = config.get('cameras_extent', 1.0)

        self.load_config_file()
        self.load_camera_file()
        self.load_pose_file()
        self.num_frames = len(self.frm_list)
        print(f'[InstantAvatarDataset][{self.split}] num_frames = {self.num_frames}')

    ##################################################
    # load config.json
    def load_config_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'config.json')):
            raise NotImplementedError
        
        with open(os.path.join(self.dat_dir, 'config.json'), 'r') as fp:
            contents = json.load(fp)

        self.smplx_config = {
            'model_type': 'smplx',
            'gender': contents['gender'],
            'use_pca': False,
            'num_betas': 10
        }

        self.start_idx = contents[self.split]['start']
        self.end_idx = contents[self.split]['end']
        self.step = contents[self.split]['skip']
        self.frm_list = [i for i in range(self.start_idx, self.end_idx+1, self.step)]

    # load cameras.npz
    def load_camera_file(self):
        with open(os.path.join(self.dat_dir, 'camera.json')) as f:
            contents = json.load(f)
            
        K = np.array(contents["intrinsic"])
        c2w = np.linalg.inv(contents["extrinsic"])

        self.cam_intrinsic = torch.tensor(contents["intrinsic"])
        self.cam_extrinsic = torch.tensor(contents["extrinsic"])
        
        height = contents["height"]
        width = contents["width"]
        w2c = np.linalg.inv(c2w)

        R = w2c[:3,:3]
        T = w2c[:3, 3]

        cam = libcore.Camera()
        cam.h, cam.w = height, width
        cam.set_K(K)
        cam.R = R
        cam.setTranslation(T)
        # print(cam)
        self.cam = cam

    # load poses.npz
    def load_pose_file(self):
        self.smplx_model = smplx_utils.create_smplx_model(**self.smplx_config)

        smplx_output_fpath = os.path.join(self.dat_dir, 'smplx_params.pt')
        if not os.path.exists(smplx_output_fpath):
            raise FileNotFoundError

        data = torch.load(smplx_output_fpath)
        
        self.smplx_beta = torch.tensor(data[0]['shape'][None,...].cpu())
        
        vert_fpath = os.path.join(self.dat_dir, 'vertices.pt')
        if not os.path.exists(vert_fpath):
            raise FileNotFoundError

        self.smplx_verts = torch.load(vert_fpath)[self.frm_list].cpu() #N_framesx10475x3
        
        # verts_normals_padded is not updated except for the first batch
        # so init with only one batch of verts and use update_padded later
        face_fpath = os.path.join(self.dat_dir, 'faces.pt')
        if not os.path.exists(vert_fpath):
            raise NotImplementedError

        smplx_faces = torch.load(face_fpath) 

        #print(self.smplx_verts[:1].shape, smplx_faces[None, ...].astype(int).shape)
        
        self.mesh_py3d = py3d_meshes.Meshes(
            self.smplx_verts[0:1], 
            torch.tensor(smplx_faces[None, ...].astype(int)),
        )

        print("self.smplx_verts.shape ", self.smplx_verts.shape)


    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = self.frm_list[idx]

        # frames
        color_frames = read_instant_avatar_frameset(self.dat_dir, frm_idx, self.cam)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)
        
        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
            'color_frames': color_frames,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
            "cam_intrinsic": self.cam_intrinsic,
            "cam_extrinsic": self.cam_extrinsic,
        }

        # mesh
        batch['mesh_info'] = self.get_smplx_mesh(idx)
        
        return batch

    def get_smplx_mesh(self, idx):
        frame_mesh = self.mesh_py3d.update_padded(self.smplx_verts[idx:idx+1])
        return {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }
