import numpy as np
from Rat7mSkeleton import RatSkeleton

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

    return np.array(l2ws)

if __name__ == "__main__":
    # Example 3D keypoints for the rat skeleton (randomly generated for testing)
    kps = np.random.rand(20, 3)
    
    # Initialize the RatSkeleton and get parent-child relationships
    rat_skt = RatSkeleton()
    parent_child_relationships = rat_skt.get_parent_child_relationships()

    # Compute the local-to-world transformations
    l2ws = get_rat_skeleton_transformation(kps, parent_child_relationships)

    # Output the resulting transformation matrices
    print(l2ws)