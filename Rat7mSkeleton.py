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


if __name__ == "__main__":
    # Get parent-child relationships starting from SpineF (index 3)
    parent_child_rels = RatSkeleton.get_parent_child_relationships(root=3)

    # Print the parent-child relationships
    for parent, child in parent_child_rels.items():
        print(f"Parent: {rat7m_joints[parent]}, Child: {rat7m_joints[child]}")