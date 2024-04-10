import numpy as np
from pymotion.io.bvh import BVH
import os

bvh = BVH()
train_dir = './nodummy_train'
save_dir_train = './train'
eval_dir = './nodummy_eval'
save_dir_eval = './eval'


def addDummyJoint(filename,dirpath,savepath): 

    file_path = os.path.join(dirpath,filename)
    print(file_path)
    bvh.load(dirpath+"/"+filename) 
    bvh.data["names"].append("Dummy")
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

    # Get the root joint data
    root_rotation = local_rotations[:, 0:1, :]
    root_position = local_positions[:, 0:1, :]

    # Create new arrays with one additional joint
    n_frames, n_joints, _ = local_rotations.shape
    new_n_joints = n_joints + 1
    new_rotations = np.zeros((n_frames, new_n_joints, 4))
    new_positions = np.zeros((n_frames, new_n_joints, 3))
    new_offsets = np.vstack([offsets, [0, 0, 0]]) 
    new_end_sites = np.vstack([end_sites, [0, 0, 0]])  
    new_end_sites_parents = end_sites_parents[:]  

    # Set data for the new joint
    new_rotations[:, :-1, :] = local_rotations
    new_positions[:, :-1, :] = local_positions
    new_rotations[:, -1, :] = root_rotation.squeeze()
    new_positions[:, -1, :] = root_position.squeeze()
    new_end_sites_parents.append(n_joints)
    new_rot_order_row = bvh.data["rot_order"][-1:, :]   # Duplicate the last row of rot_order to create a new row
    new_rot_order = np.concatenate((bvh.data["rot_order"], new_rot_order_row), axis=0) # Concatenate the new row to the existing rot_order array

    bvh.data["offsets"]=new_offsets
    bvh.data["end_sites"]=new_end_sites
    bvh.data["end_sites_parents"]=new_end_sites_parents
    bvh.data["rot_order"] = new_rot_order
    bvh.data["parents"].append(0)

    # Set and save
    bvh.set_data(new_rotations,new_positions)
    bvh.save(savepath+"/"+filename)


def main():

    ## Adding dummy joint to all files in database ##

    files_train = os.listdir(train_dir)
    files_eval = os.listdir(eval_dir)

    for filename in files_train:
        addDummyJoint(filename,train_dir,save_dir_train)

    for filename in files_eval:
        addDummyJoint(filename,eval_dir,save_dir_eval)
    


if __name__ == "__main__":
    main()
