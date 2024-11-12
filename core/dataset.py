from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
import math
import h5py
import numpy as np
import torch

torch.set_default_dtype(torch.float)

class BaseH5Dataset(Dataset):

    def __init__(self, h5_path, N_samples=96, patch_size=1, split='full',
                 N_nms=0, subject=None, mask_img=False):
        '''
        Base class for multi-proc h5 dataset

        args
        ----
        h5_path (str): path to .h5 file
        N_samples (int): number of pixels to sample from each image
        patch_size (int): sample patches of rays of this size.
        split (str): split to use. splits are defined in a dataset-specific manner
        N_nms (float): number of pixel samples to sample from out-of-mask regions (in a bounding box).
        subject (str): name of the dataset subject
        mask_img (bool): replace background parts with estimated background pixels
        '''
        self.h5_path = h5_path
        self.split = split
        self.dataset = None
        self.subject = subject
        self.mask_img = mask_img

        self.N_samples = N_samples
        self.patch_size = patch_size
        self.N_nms = int(math.floor(N_nms)) if N_nms >= 1.0 else float(N_nms)
        self._idx_map = None # map queried idx to predefined idx
        self._render_idx_map = None # map idx for render

        self.init_meta()
        self.init_len()

        self.render_skip = 1
        self.N_render = 15

    def __getitem__(self, q_idx):
        '''
        q_idx: index queried by sampler, should be in range [0, len(dataset)].
        Note - self._idx_map maps q_idx to indices of the sub-dataset that we want to use.
               therefore, self._idx_map[q_idx] may not lie within [0, len(dataset)]
        '''

        if self._idx_map is not None:
            idx = self._idx_map[q_idx]
        else:
            idx = q_idx

        # TODO: map idx to something else (e.g., create a seq of img idx?)
        # or implement a different sampler
        # as we may not actually use the idx here

        if self.dataset is None:
            self.init_dataset()

        # get camera information
        c2w, focal, center, cam_idxs = self.get_camera_data(idx, q_idx, self.N_samples)

        # get kp index and kp, skt, bone, cyl
        kp_idxs, kps, skts = self.get_pose_data(idx, q_idx, self.N_samples)

        # sample pixels
        pixel_idxs = self.sample_pixels(idx, q_idx)

        # print("sample pixel ids: ", pixel_idxs)

        # maybe get a version that computes only sampled points?
        rays_o, rays_d = self.get_rays(c2w, focal, pixel_idxs, center)

        # load the image, foreground and background,
        # and get values from sampled pixels
        rays_rgb, fg, bg = self.get_img_data(idx, pixel_idxs)


        return_dict = {'rays_o': rays_o,
                       'rays_d': rays_d,
                       'target_s': rays_rgb,
                       'kp_idx': kp_idxs,
                       'kp3d': kps,
                       'skts': skts,
                       'cam_idxs': cam_idxs,
                       'fgs': fg,
                       'bgs': bg,
                       }

        return return_dict
    
    def __len__(self):
        return self.data_len
    
    def init_len(self):
        if self._idx_map is not None:
            self.data_len = len(self._idx_map)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_len = len(f['imgs'])

    def init_dataset(self):

        if self.dataset is not None:
            return
        print('init dataset')

        self.dataset = h5py.File(self.h5_path, 'r')

    def init_meta(self):
        '''
        Init properties that can be read directly into memory (as they are small)
        '''
        print('init meta')

        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        self.dataset_keys = [k for k in dataset.keys()]

        # initialize some attributes
        self.has_bg = 'bkgds' in self.dataset_keys
        self.centers = None

        if 'centers' in dataset:
            self.centers = dataset['centers'][:]

        # precompute mesh (for ray generation) to reduce computational cost
        img_shape = dataset['img_shape'][:]
        self._N_total_img = img_shape[0]
        self.HW = img_shape[1:3]
        mesh = np.meshgrid(np.arange(self.HW[1], dtype=np.float32),
                           np.arange(self.HW[0], dtype=np.float32),
                           indexing='xy')
        self.mesh = mesh

        i, j = mesh[0].reshape(-1), mesh[1].reshape(-1)

        if self.centers is None:
            offset_y, offset_x = self.HW[0] * 0.5, self.HW[1] * 0.5
        else:
            # have per-image center. apply those during runtime
            offset_y = offset_x = 0.

        # pre-computed direction, the first two cols
        # need to be divided by focal
        self._dirs = np.stack([ (i-offset_x),
                              -(j-offset_y),
                              -np.ones_like(i)], axis=-1)

        # pre-computed pixel indices
        self._pixel_idxs = np.arange(np.prod(self.HW)).reshape(*self.HW)

        # store pose and camera data directly in memory (they are small)
        self.gt_kp3d = dataset['gt_kp3d'][:] if 'gt_kp3d' in self.dataset_keys else None
        self.kp_map, self.kp_uidxs = None, None 
        self.kp3d, self.skts = self._load_pose_data(dataset)

        self.focals, self.c2ws = self._load_camera_data(dataset)

        self.temp_validity = None

        if self.has_bg:
            self.bgs = dataset['bkgds'][:].reshape(-1, np.prod(self.HW), 3)
            self.bg_idxs = dataset['bkgd_idxs'][:].astype(np.int64)

        dataset.close()
        
    def get_near_far(self, idx):
        """
        Computes near and far distances for ray sampling based on the 3D keypoints and camera position.
        
        Args:
          idx: Index of the sample for which to calculate near and far values.
        
        Returns:
          near, far: Calculated near and far distances for ray sampling.
        """
        # Retrieve camera transformation and position
        c2w, _, _, _ = self.get_camera_data(idx, idx, self.N_samples)
        camera_position = c2w[:3, 3]  # Camera position in world coordinates

        # Retrieve 3D keypoints for this sample
        kp3d = self.kp3d[idx]  # (N_joints, 3), keypoints in world coordinates

        # Calculate distances from the camera to each keypoint
        distances = np.linalg.norm(kp3d - camera_position, axis=1)

        # Define near and far with a margin
        margin = 0.1  # Adjust this as needed
        near = max(0.1, distances.min() - margin)  # near must be positive
        far = distances.max() + margin

        return near, far

    def _load_pose_data(self, dataset):
        '''
        read pose data from .h5 file
        '''
        kp3d, skts = dataset['gt_kp3d'][:], dataset['skts'][:]

        return kp3d, skts
    
    def _load_camera_data(self, dataset):
        '''
        read camera data from .h5 file
        '''
        return dataset['focals'][:] * 1000, dataset['c2ws'][:]

    def get_camera_data(self, idx, q_idx, N_samples):
        '''
        get camera data
        '''
        real_idx, cam_idx = self.get_cam_idx(idx, q_idx)

        # Since we only have one camera, we just use the first one
        focal = self.focals[0] 
        c2w = self.c2ws[0].astype(np.float32)

        center = None
        if self.centers is not None:
            center = self.centers[0]

        cam_idx = np.array(cam_idx).reshape(-1, 1).repeat(N_samples, 1).reshape(-1)

        return c2w, focal, center, cam_idx
    
    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx, q_idx
    
    def get_img_data(self, idx, pixel_idxs):
        '''
        get image data (in np.uint8)
        '''
        fg = self.dataset['masks'][idx].reshape(-1)[pixel_idxs].astype(np.float32)
        img = self.dataset['imgs'][idx].reshape(-1, 3)[pixel_idxs].astype(np.float32) / 255.

        bg = None
        if self.has_bg:
            bg_idx = self.bg_idxs[idx]
            bg = self.bgs[bg_idx, pixel_idxs].astype(np.float32) / 255.

            if self.mask_img:
                img = img * fg + (1. - fg) * bg

        return img, fg, bg
    
    def sample_pixels(self, idx, q_idx):
        '''
        return sampled pixels (in (H*W,) indexing, not (H, W))
        '''
        p = self.patch_size
        N_rand = self.N_samples // int(p**2)
        H, W = self.HW
        sampling_mask = self.dataset['masks'][idx].reshape(H, W)

        valid_idxs = np.where(sampling_mask > 0)
        num_valid = len(valid_idxs[0])
        if num_valid < N_rand:
            selected = np.random.choice(num_valid, N_rand, replace=True)
        else:
            selected = np.random.choice(num_valid, N_rand, replace=False)
        
        hs, ws = valid_idxs[0][selected], valid_idxs[1][selected]

        if self.patch_size > 1:
            hs = np.clip(hs, a_min=0, a_max=H-p)
            ws = np.clip(ws, a_min=0, a_max=W-p)
            _s = []
            for h, w in zip(hs, ws):
                patch = np.array([(h+i)*W + (w+j) for i in range(p) for j in range(p)])
                _s.append(patch)
            sampled_idxs = np.array(_s).reshape(-1)
        else:
            sampled_idxs = hs * W + ws

        if isinstance(self.N_nms, int):
            N_nms = self.N_nms
        else:
            N_nms = int(self.N_nms > np.random.random())

        if N_nms > 0:
            nms_idxs = self._sample_in_box2d(idx, q_idx, sampling_mask, N_nms)
            sampled_idxs[np.random.choice(len(sampled_idxs), size=(N_nms,), replace=False)] = nms_idxs

        # Ensure all indices are within bounds
        sampled_idxs = np.clip(sampled_idxs, 0, H*W - 1)

        return np.sort(sampled_idxs)
    
    def _sample_in_box2d(self, idx, q_idx, fg, N_samples):

        H, W = self.HW
        # get bounding box
        real_idx, _ = self.get_cam_idx(idx, q_idx)
        tl, br = self.box2d[real_idx].copy()

        fg = fg.reshape(H, W)
        cropped = fg[tl[1]:br[1], tl[0]:br[0]]
        vy, vx = np.where(cropped < 1)

        # put idxs from cropped ones back to the non-cropped ones
        vy = vy + tl[1]
        vx = vx + tl[0]
        idxs = vy * W + vx

        #selected_idxs = np.random.choice(idxs, size=(N_samples,), replace=False)
        # This is faster for small N_samples
        selected_idxs = np.random.default_rng().choice(idxs, size=(N_samples,), replace=False)

        return selected_idxs
    
    def get_rays(self, c2w, focal, pixel_idxs, center=None):

        dirs = self._dirs[pixel_idxs].copy()

        if center is not None:
            center = center.copy()
            center[1] *= -1
            dirs[..., :2] -= center

        dirs[:, :2] /= focal

        I = np.eye(3)

        if np.isclose(I, c2w[:3, :3]).all():
            rays_d = dirs # no rotation required if rotation is identity
        else:
            rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        return rays_o.copy(), rays_d.copy()


    def get_pose_data(self, idx, q_idx, N_samples):

        # real_idx: the real data we want to sample from
        # kp_idx: for looking up the optimized kp in poseopt layer (or other purpose)
        real_idx, kp_idx = self.get_kp_idx(idx, q_idx)

        kp = self.kp3d[real_idx:real_idx+1].astype(np.float32)
        skt = self.skts[real_idx:real_idx+1].astype(np.float32)

        kp_idx = np.array([kp_idx]).repeat(N_samples, 0)
        kp = kp.repeat(N_samples, 0)
        skt = skt.repeat(N_samples, 0)

        return kp_idx, kp, skt
    
    
    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx, q_idx
    
    def _get_subset_idxs(self, render=False):
        '''return idxs for the subset data that you want to train on.
        Returns:
        k_idxs: idxs for retrieving pose data from .h5
        c_idxs: idxs for retrieving camera data from .h5
        i_idxs: idxs for retrieving image data from .h5
        kq_idxs: idx map to map k_idxs to consecutive idxs for rendering
        cq_idxs: idx map to map c_idxs to consecutive idxs for rendering
        '''
        if self._idx_map is not None:
            # queried_idxs
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))

        else:
            # queried == actual index
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(len(self.kp3d))
            _c_idxs = _cq_idxs = np.arange(len(self.c2ws))

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs
    
    def get_meta(self):
        '''
        return metadata needed for other parts of the code.
        '''

        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        # get idxs to retrieve the correct subset of meta-data

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs()

        # prepare HWF
        H, W = self.HW
        if not np.isscalar(self.focals):
            H = np.repeat([H], len(c_idxs), 0)
            W = np.repeat([W], len(c_idxs), 0)

        hwf = (H, W, self.focals[c_idxs])

        # prepare center if there's one
        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()

        data_attrs = {
            'hwf': hwf,
            'center': center,
            'c2ws': self.c2ws[c_idxs],
            'near': 0.8, 'far': 2.5, # don't really need this
            'n_views': self.data_len,
            # skeleton-related info
            'gt_kp3d': self.gt_kp3d[k_idxs] if self.gt_kp3d is not None else None,
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
        }

        dataset.close()

        return data_attrs
    
    def get_render_data(self):

        dataset = h5py.File(self.h5_path, 'r', swmr=True)

        # get the subset idxs to collect the right data
        k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs = self._get_subset_idxs(render=True)

        # grab only a subset (15 images) for rendering
        kq_idxs = kq_idxs[::self.render_skip][:self.N_render]
        cq_idxs = cq_idxs[::self.render_skip][:self.N_render]
        i_idxs = i_idxs[::self.render_skip][:self.N_render]
        k_idxs = k_idxs[::self.render_skip][:self.N_render]
        c_idxs = c_idxs[::self.render_skip][:self.N_render]

        # get images if split == 'render'
        # note: needs to have self._idx_map
        H, W = self.HW
        render_imgs = dataset['imgs'][i_idxs].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][i_idxs].reshape(-1, H, W, 1)
        render_bgs = self.bgs.reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_bg_idxs = self.bg_idxs[i_idxs]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, self.focals[c_idxs])

        center = None
        if self.centers is not None:
            center = self.centers[c_idxs].copy()

        hwf = (int(hwf[0][0]), int(hwf[1][0]), hwf[2][0] )

        render_data = {
            'imgs': render_imgs,
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(self.bgs),
            # camera data
            'cam_idxs': c_idxs,
            'cam_idxs_len': len(self.c2ws),
            'c2ws': [self.c2ws[0] for _ in range(len(render_imgs))],
            'hwf': hwf,
            'center': center,
            # keypoint data
            'kp_idxs': k_idxs,
            'kp_idxs_len': len(self.kp3d),
            'kp3d': self.kp3d[k_idxs],
            'skts': self.skts[k_idxs],
            'bones':self.kp3d[k_idxs],
        }

        dataset.close()

        return render_data
    

    
if __name__ == "__main__":
    dataset = BaseH5Dataset("data/rat7mdata.h5")
    
    # 1. Check metadata
    meta = dataset.get_meta()
    print("Dataset metadata:")
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{key}: {value}")
    
    # Ensure dataset is initialized
    dataset.init_dataset()
    
    # 2. Check a few samples
    print("\nChecking a few samples:")
    for i in range(3):  # Check first 3 samples
        print(f"\nSample {i}:")
        
        try:
            # Print pixel_idxs
            pixel_idxs = dataset.sample_pixels(i, i)
            print(f"pixel_idxs: min={pixel_idxs.min()}, max={pixel_idxs.max()}, shape={pixel_idxs.shape}")
            
            # Print mask shape
            mask_shape = dataset.dataset['masks'].shape
            print(f"Mask shape: {mask_shape}")
            
            sample = dataset[i]
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
    
    # 3. Verify data consistency
    print("\nVerifying data consistency:")
    assert len(dataset) == dataset.data_len, "Dataset length mismatch"
    print(f"Dataset length: {len(dataset)}")
    
    print("\nBasic verification complete. Please review the output for any inconsistencies.")