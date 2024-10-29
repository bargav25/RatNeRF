import h5py
import numpy as np
import pickle as pkl
from tqdm import tqdm
import cv2
from utils.skeleton_utils import *
import matplotlib.pyplot as plt

DATA_PATH = "../data/rats/"

H5_PATH = "../data/rats/rat7mdata.h5"
PKL_PATH = "../data/rats/rat7m_s1_s2_s3_s4.pkl"
MASK_SEGPATH = "../data/rats/masks_rat7m.npy"

def camera_to_world(camera_coords, extrinsic_matrix):
    """Transform 3D camera coordinates to 3D world coordinates."""
    camera_coords = np.atleast_2d(camera_coords)
    camera_coords_homogeneous = np.hstack((camera_coords, np.ones((camera_coords.shape[0], 1))))
    extrinsic_inverse = np.linalg.inv(extrinsic_matrix)
    world_coords_homogeneous = np.dot(camera_coords_homogeneous, extrinsic_inverse.T)
    return world_coords_homogeneous[:, :3] / world_coords_homogeneous[:, 3:]

def extract_focal_and_c2w(camera_params):
    """Extract focal length and camera-to-world matrix from camera parameters."""
    fx, fy = camera_params['IntrinsicMatrix'][0][0], camera_params['IntrinsicMatrix'][1][1]
    focal = (fx + fy) / 2

    extrinsic = camera_params['extrinsic']
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    R_inv, t_inv = R.T, -R.T @ t

    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R_inv, t_inv

    return focal, c2w

def apply_bounding_box_to_masks(masks, padding=2):
    """
    Apply a bounding box to each mask in the given array and return a new array
    where each mask is replaced by its corresponding bounding box.

    Parameters:
        masks (numpy.ndarray): Array of binary masks (shape: [num_masks, height, width]).
        padding (int): Amount of extra space around the bounding box.

    Returns:
        numpy.ndarray: Array with bounding boxes applied to each mask.
    """
    bounding_box_masks = np.zeros_like(masks)
    

    for i, mask in tqdm(enumerate(masks)):
        # Find the non-zero (object) pixels in the mask
        y_indices, x_indices = np.where(mask > 0)

        if len(x_indices) == 0 or len(y_indices) == 0:
            # Skip if no object is present in the mask
            continue

        # Calculate the bounding box coordinates with extra space
        x_min = max(0, np.min(x_indices) - padding)
        x_max = min(mask.shape[1], np.max(x_indices) + padding)
        y_min = max(0, np.min(y_indices) - padding)
        y_max = min(mask.shape[0], np.max(y_indices) + padding)

        # Create a new mask with the bounding box set to 1 (binary)
        bounding_box_masks[i, y_min:y_max, x_min:x_max] = 1

    return bounding_box_masks


def process_data(data):
    """Filter and process the data records."""
    filtered_data = [x[5] for x in data if 's3-d1' in x[5]['image_path']]
    processed_data = filtered_data
    
    for item in tqdm(processed_data, desc="Processing data"):
        item['image_path'] = item['image_path'].replace("s3-d1/Camera6/", DATA_PATH)
    
    return processed_data

def filter_nan_data(data):
    """Filter data on nan."""
    valid_indices = []
    filtered_data = []
    for i, item in enumerate(data):
        if not np.any(np.isnan(item['pose_3d'])):
            valid_indices.append(i)
            filtered_data.append(item)
    
    print(f"Filtered {len(data) - len(filtered_data)} frames with NaN values")
    return filtered_data, valid_indices

def load_images(data, valid_indices):
    """Load images from the processed data and apply masks to remove the background."""
    images = []

    masks = np.load(MASK_SEGPATH)
    masks = masks[valid_indices]

    for i, item in enumerate(tqdm(data, desc="Loading images")):
        # Load the image
        img = cv2.imread(item['image_path'])

        # Apply the corresponding mask
        mask = masks[i]
        
        # Ensure the mask is binary (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Create a white background
        white_background = np.full_like(img, 255)
        
        # Apply the mask: keep the rat region, and set the rest to white
        masked_img = np.where(binary_mask[:, :, None], img, white_background)
        
        images.append(masked_img)
    
    return images

def main():
    # Load and process data
    with open(PKL_PATH, 'rb') as f:
        raw_data = pkl.load(f)
    
    processed_data = process_data(raw_data)
    print("Sample data keys:", processed_data[0].keys())

    processed_data, valid_indices = filter_nan_data(processed_data)

    # Extract camera parameters
    focal, c2w = extract_focal_and_c2w(processed_data[0]['camera_matrices'])
    extrinsic_matrix = processed_data[0]['camera_matrices']['extrinsic']

    # Prepare data for H5 file
    focals = [focal]
    c2ws = [c2w]
    gt_kp3d_world = [camera_to_world(item['pose_3d'], extrinsic_matrix) for item in tqdm(processed_data,  desc="Converting Camera Coordinates to World Coordinates")]
    
    imgs = load_images(processed_data, valid_indices)

    num_imgs = len(processed_data)
    img_shape = [num_imgs, *imgs[0].shape]
    img_shape = list(map(lambda x : int(x), img_shape))

    masks = np.load(MASK_SEGPATH)
    masks = apply_bounding_box_to_masks(masks)

    masks = masks[valid_indices]

    bkgd_idxs = [0 for _ in range(num_imgs)]
    bkgds = np.full((1, imgs[0].shape[0] * imgs[0].shape[1], 3), 255)

    rat_skt = RatSkeleton()
    parent_child_relationships = rat_skt.get_parent_child_relationships()

    skts = [get_rat_skeleton_transformation(kps, parent_child_relationships)[0] for kps in tqdm(gt_kp3d_world)]



    print(f"Masks shape: {masks.shape}")
    print(f"Skts shape: {np.array(skts).shape}")

    # Save data to H5 file
    with h5py.File(H5_PATH, 'w') as file:
        for name, data in [
            ('c2ws', c2ws),
            ('focals', focals),
            ('gt_kp3d', gt_kp3d_world),
            ('imgs', imgs),
            ('img_shape', img_shape),
            ('masks', masks),
            ('bkgd_idxs', bkgd_idxs),
            ('bkgds', bkgds),
            ('skts', skts)
        ]:
            file.create_dataset(name, data=data, compression="gzip")

    print(f"Data successfully saved to {H5_PATH}")

if __name__ == "__main__":
    main()