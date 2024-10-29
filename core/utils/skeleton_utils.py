import numpy as np

rat7m_joints = ['HeadF', 'HeadB', 'HeadL', 'SpineF', 'SpineM', 'SpineL', 'Offset1', 'Offset2', 
                'HipL', 'HipR', 'ElbowL', 'ArmL', 'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 
                'KneeR', 'KneeL', 'ShinL', 'ShinR']


class RatSkeleton:
    # 0-based links between joints
    links = np.array([
        [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 7], [4, 13], [4, 14], 
        [5, 6], [6, 8], [6, 9], [6, 10], [7, 8], [9, 18], [10, 17], [18, 19], 
        [17, 20], [11, 12], [13, 11], [14, 15], [15, 16]
    ]) - 1

    @staticmethod
    def get_parent_child_relationships(root=3):
        '''
        Return the parent-child relationships for the skeleton.
        Args:
          root (int): Index of the root joint. Default is 3 (SpineF).
        Returns:
          parent_child (list): A list of (parent, child) pairs.
        '''
        parent_child = {}
        visited = set()
        to_visit = [root]  # Start with the root joint

        # As long as there are joints to visit
        while to_visit:
            current_joint = to_visit.pop(0)
            visited.add(current_joint)

            # Iterate over the links
            for link in RatSkeleton.links:
                # If current_joint is in the link
                if current_joint in link:
                    # Find the other joint connected to it
                    child_joint = link[0] if link[1] == current_joint else link[1]
                    
                    # If the child_joint hasn't been visited yet, it is a child of current_joint
                    if child_joint not in visited:
                        parent_child[child_joint] = current_joint
                        to_visit.append(child_joint)

        return parent_child




def create_local_coord(vec):
    '''
    Creates a local coordinate system where the z-axis is aligned with the input vector `vec`.
    
    Args:
      vec: A 3D vector (numpy array) representing the direction to align the z-axis with.
    
    Returns:
      coord: A 4x4 matrix representing the local coordinate system where the z-axis is aligned with `vec`.
    '''
    # Default global coordinate axes (in world coordinates)
    x_axis = np.array([1., 0., 0.], dtype=np.float32)  # Standard x-axis
    y_axis = np.array([0., 1., 0.], dtype=np.float32)  # Standard y-axis
    z_axis = np.array([0., 0., 1.], dtype=np.float32)  # Standard z-axis

    # If the input vector is close to zero (no direction), return identity matrix
    if np.isclose(np.linalg.norm(vec), 0.):
        return np.eye(4)

    # --- Step 1: Align the z-axis with the input vector in the xz-plane ---
    vec_xz = vec[[0, 2]] / np.linalg.norm(vec[[0, 2]])  # Normalize projection on xz-plane
    theta = arccos_safe(vec_xz[-1]) * np.sign(vec_xz[0])  # Angle between input vector and z-axis
    rot_y = rotate_y(theta)  # Rotation matrix around y-axis by theta

    # Rotate the vector to align with the xz-plane
    rotated_y = rot_y[:3, :3] @ vec

    # --- Step 2: Further align the vector in the yz-plane ---
    vec_yz = rotated_y[1:3] / np.linalg.norm(rotated_y[1:3])  # Normalize projection on yz-plane
    psi = arccos_safe(vec_yz[-1]) * np.sign(vec_yz[0])  # Angle in yz-plane
    rot_x = rotate_x(psi)  # Rotation matrix around x-axis by psi

    # --- Step 3: Construct the final rotation matrix ---
    rot = np.linalg.inv(rot_x @ rot_y)  # Combine the rotations
    coord = np.eye(4)  # Start with identity matrix
    coord[:3, :3] = rot[:3, :3]  # Set rotation part
    return coord

def arccos_safe(a):
    ''' 
    Safely computes arccos with clipping to avoid numerical issues outside the range [-1, 1]. 
    '''
    clipped = np.clip(a, -1. + 1e-8, 1. - 1e-8)
    return np.arccos(clipped)

def rotate_x(phi):
    ''' Creates a 4x4 rotation matrix to rotate around the x-axis by an angle phi (in radians). '''
    cos, sin = np.cos(phi), np.sin(phi)
    return np.array([[1, 0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0, 0,    0, 1]], dtype=np.float32)

def rotate_y(theta):
    ''' Creates a 4x4 rotation matrix to rotate around the y-axis by an angle theta (in radians). '''
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, 0, sin, 0],
                     [0,   1, 0,  0],
                     [-sin, 0, cos, 0],
                     [0,   0, 0,  1]], dtype=np.float32)

def sort_joints_by_hierarchy(parent_child_relationships, root_joint):
    """
    Sorts joints by hierarchy based on parent-child relationships, ensuring that the root joint is processed first.
    
    Args:
      parent_child_relationships: (dict) where each key represents the child joint and value represents the parent joint.
      root_joint: Index of the root joint to start from.
    
    Returns:
      sorted_indices: Sorted indices of joints based on hierarchy.
    """
    visited, sorted_indices = set(), []

    # Helper function to recursively visit joints in the hierarchy
    def visit(joint):
        if joint not in visited:
            parent = parent_child_relationships.get(joint, None)
            if parent is not None:  # Visit parent first if the joint has a parent
                visit(parent)
            visited.add(joint)
            sorted_indices.append(joint)

    # Start by visiting the root joint first
    visit(root_joint)

    # Visit remaining joints if they haven't been visited already
    for joint in parent_child_relationships:
        visit(joint)

    return sorted_indices

def get_rat_skeleton_transformation(kps, parent_child_relationships, root_joint=3):
    """
    Computes local-to-world transformation matrices for each joint.
    
    Args:
      kps: (N_joints, 3) array of 3D keypoints for each joint
      parent_child_relationships: (dict) mapping each child joint to its parent joint.
      root_joint: Index of the root joint.
    
    Returns:
      l2ws: Local-to-world transformation matrices for each joint.
    """
    N_joints = kps.shape[0]
    l2ws = [None] * N_joints  # Initialize the list for storing transformations

    # Sort joints to ensure parents are processed before children
    sorted_indices = sort_joints_by_hierarchy(parent_child_relationships, root_joint)

    # Process each joint in sorted order
    for i in sorted_indices:
        if i == root_joint:
            # Root joint transformation (identity transformation, positioned at root)
            root_T = np.eye(4)
            root_T[:3, 3] = kps[i]  # Set root position in world space
            l2ws[i] = root_T
        else:
            parent = parent_child_relationships[i]
            vec = kps[i] - kps[parent]  # Vector from parent to child
            local_rotation = create_local_coord(vec)[:3, :3]  # Compute local rotation matrix

            # Build the local transformation matrix
            T = np.eye(4)
            T[:3, :3] = local_rotation  # Add rotation part
            T[:3, 3] = vec  # Add translation part (relative position from parent to child)

            # Compute the global transformation (parent's world transform @ local transform)
            l2ws[i] = l2ws[parent] @ T

    # Compute world-to-local transformations by inverting local-to-world
    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])

    return skts, np.array(l2ws)

def get_kp_bounding_cylinder(kp, skel_type=None, ext_scale=0.0001,
                             extend_mm=50, top_expand_ratio=0.5,
                             bot_expand_ratio=0.2, head=None):
    '''
    Defines a bounding cylinder for a rat model, encompassing all keypoints.
    - kp: Keypoints as a NumPy array, expected shape [num_keypoints, 3] or [batch_size, num_keypoints, 3].
    - skel_type: Type of skeleton, if applicable; if not, determined from `kp`.
    - ext_scale: Scale factor for cylinder radius extension.
    - extend_mm: Millimeter extension to base radius.
    - top_expand_ratio, bot_expand_ratio: Ratios for extending the top and bottom bounds of the cylinder.
    - head: Specifies the ground plane direction (e.g., '-y' for SPIN, 'z' for SURREAL datasets).
    '''
    # Ensure head direction is specified
    assert head is not None, "Specify the direction of the ground plane for the rat's body orientation."
    print(f'Head direction: {head}')

    # Define ground and vertical axes based on head orientation
    if head.endswith('z'):
        g_axes = [0, 1]
        h_axis = 2
    elif head.endswith('y'):
        g_axes = [0, 2]
        h_axis = 1
    else:
        raise NotImplementedError(f'Head orientation "{head}" is not implemented for rats!')

    # Flip height if the head is in the negative direction
    flip = 1 if not head.startswith('-') else -1

    n_dim = len(kp.shape)
    # Root location is usually the pelvis or torso region
    root_loc = kp[..., 3, :]  # assuming skel_type.root_id = 3

    # Calculate distance to center line (horizontal distance from root)
    if n_dim == 2:
        dist = np.linalg.norm(kp[:, g_axes] - root_loc[g_axes], axis=-1)
        max_height = (flip * kp[:, h_axis]).max()
        min_height = (flip * kp[:, h_axis]).min()
        max_dist = dist.max()
    elif n_dim == 3:  # for batches
        dist = np.linalg.norm(kp[..., g_axes] - root_loc[:, None, g_axes], axis=-1)
        max_height = (flip * kp[..., h_axis]).max(axis=1)
        min_height = (flip * kp[..., h_axis]).min(axis=1)
        max_dist = dist.max(axis=1)

    # Set cylinder radius and height bounds with additional adjustments for rat proportions
    extension = extend_mm * ext_scale
    radius = max_dist + extension
    top = flip * (max_height + extension * top_expand_ratio)  # extend head region a bit
    bot = flip * (min_height - extension * bot_expand_ratio)  # limit tail extension

    # Expand dimensions to make sure shapes are consistent for stacking
    root_loc = root_loc[..., g_axes]  # [batch, 2] or [2]
    radius = np.expand_dims(radius, -1)
    top = np.expand_dims(top, -1)
    bot = np.expand_dims(bot, -1)

    # Stack cylinder parameters: center coordinates, radius, top, and bottom
    cylinder_params = np.concatenate([root_loc, radius, top, bot], axis=-1)

    return cylinder_params

def focal_to_intrinsic_np(focal):
    if isinstance(focal, float) or (len(focal.reshape(-1)) < 2):
        focal_x = focal_y = focal
    else:
        focal_x, focal_y = focal
    return np.array([[focal_x,      0, 0, 0],
                     [     0, focal_y, 0, 0],
                     [     0,       0, 1, 0]],
                    dtype=np.float32)


def cylinder_to_box_2d(cylinder_params, hwf, w2c=None, scale=1.0,
                       center=None, make_int=True):

    H, W, focal = hwf

    root_loc, radius = cylinder_params[..., :2], cylinder_params[..., 2:3]
    top, bot = cylinder_params[..., 3:4], cylinder_params[..., 4:5]

    rads = np.linspace(0., 2 * np.pi, 50)

    if len(root_loc.shape) == 1:
        root_loc = root_loc[None]
        radius = radius[None]
        top = top[None]
        bot = bot[None]
    N = root_loc.shape[0]

    x = root_loc[..., 0:1] + np.cos(rads)[None] * radius
    z = root_loc[..., 1:2] + np.sin(rads)[None] * radius

    y_top = top * np.ones_like(x)
    y_bot = bot * np.ones_like(x)
    w = np.ones_like(x) # to make homogenous coord

    top_cap = np.stack([x, y_top, z, w], axis=-1)
    bot_cap = np.stack([x, y_bot, z, w], axis=-1)

    cap_pts = np.concatenate([top_cap, bot_cap], axis=-2)
    cap_pts = cap_pts.reshape(-1, 4)

    intrinsic = focal_to_intrinsic_np(focal)

    if w2c is not None:
        cap_pts = cap_pts @ w2c.T
    cap_pts = cap_pts @ intrinsic.T
    cap_pts = cap_pts.reshape(N, -1, 3)
    pts_2d = cap_pts[..., :2] / cap_pts[..., 2:3]

    max_x = pts_2d[..., 0].max(-1)
    min_x = pts_2d[..., 0].min(-1)
    max_y = pts_2d[..., 1].max(-1)
    min_y = pts_2d[..., 1].min(-1)

    if make_int:
        max_x = np.ceil(max_x).astype(np.int32)
        min_x = np.floor(min_x).astype(np.int32)
        max_y = np.ceil(max_y).astype(np.int32)
        min_y = np.floor(min_y).astype(np.int32)

    tl = np.stack([min_x, min_y], axis=-1)
    br = np.stack([max_x, max_y], axis=-1)

    if center is None:
        offset_x = int(W * .5)
        offset_y = int(H * .5)
    else:
        offset_x, offset_y = int(center[0]), int(center[1])


    tl[:, 0] += offset_x
    tl[:, 1] += offset_y

    br[:, 0] += offset_x
    br[:, 1] += offset_y

    # scale the box
    if scale != 1.0:
        box_width = (max_x - min_x) * 0.5 * scale
        box_height = (max_y - min_y) * 0.5 * scale
        center_x = (br[:, 0] + tl[:, 0]).copy() * 0.5
        center_y = (br[:, 1] + tl[:, 1]).copy() * 0.5

        tl[:, 0] = center_x - box_width
        br[:, 0] = center_x + box_width
        tl[:, 1] = center_y - box_height
        br[:, 1] = center_y + box_height

    tl[:, 0] = np.clip(tl[:, 0], 0, W-1)
    br[:, 0] = np.clip(br[:, 0], 0, W-1)
    tl[:, 1] = np.clip(tl[:, 1], 0, H-1)
    br[:, 1] = np.clip(br[:, 1], 0, H-1)

    if N == 1:
        tl = tl[0]
        br = br[0]
        pts_2d = pts_2d[0]

    return tl, br, pts_2d
    
def coord_to_homogeneous(c):
    assert c.shape[-1] == 3

    if len(c.shape) == 2:
        h = np.ones((c.shape[0], 1)).astype(c.dtype)
        return np.concatenate([c, h], axis=1)
    elif len(c.shape) == 1:
        h = np.array([0, 0, 0, 1]).astype(c.dtype)
        h[:3] = c
        return h
    else:
        raise NotImplementedError(f"Input must be a 2-d or 1-d array, got {len(c.shape)}")

def swap_mat(mat):

    return np.concatenate([
        mat[..., 0:1], -mat[..., 1:2], -mat[..., 2:3], mat[..., 3:]
        ], axis=-1)

def nerf_c2w_to_extrinsic(c2w):
    return np.linalg.inv(swap_mat(c2w))

def world_to_cam(pts, extrinsic, H, W, focal, center=None):

    if center is None:
        offset_x = W * .5
        offset_y = H * .5
    else:
        offset_x, offset_y = center

    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)

    intrinsic = focal_to_intrinsic_np(focal)

    cam_pts = pts @ extrinsic.T @ intrinsic.T
    cam_pts = cam_pts[..., :2] / cam_pts[..., 2:3]
    cam_pts[cam_pts == np.inf] = 0.
    cam_pts[..., 0] += offset_x
    cam_pts[..., 1] += offset_y
    return cam_pts

def calculate_angle(a, b=None):
    if b is None:
        b = torch.Tensor([0., 0., 1.]).view(1, 1, -1)
    dot_product = (a * b).sum(-1)
    norm_a = torch.norm(a, p=2, dim=-1)
    norm_b = torch.norm(b, p=2, dim=-1)
    cos = dot_product / (norm_a * norm_b)
    cos = torch.clamp(cos, -1. + 1e-6, 1. - 1e-6)
    angle = torch.acos(cos)
    assert not torch.isnan(angle).any()

    return angle - 0.5 * np.pi

if __name__ == "__main__":
    # Example 3D keypoints for the rat skeleton (randomly generated for testing)
    kps = np.random.rand(20, 3)
    
    # Initialize the RatSkeleton and get parent-child relationships
    rat_skt = RatSkeleton()
    parent_child_relationships = rat_skt.get_parent_child_relationships()

    # Compute the local-to-world transformations
    skts, l2ws = get_rat_skeleton_transformation(kps, parent_child_relationships)

    # Output the resulting transformation matrices
    print(l2ws.shape)
    print(skts.shape)
    