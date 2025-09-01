import torch
import os
from tqdm import tqdm
import pandas as pd
import argparse
import pickle
import numpy as np
import trimesh
import glob
def generate_obj_rgb_paths(root, df, frame_num=120):
    """
    生成 obj_rgb 图片路径，读取 color 文件夹下的所有帧，并根据帧数填充 0。

    参数:
        root (str): 数据集根目录。
        df (pd.DataFrame): 包含路径数据框，'path' 列为视频文件夹名。
        frame_num (int): 定长帧数，默认为 120。

    返回:
        obj_rgb_paths (list): 生成的图片路径列表。
    """
    obj_rgb_paths = []

    for i in tqdm(range(len(df))):
        video_name = df.at[i, 'path']  # 'path' 存的是 Video_files 下的某个 video_name

        # 搜索对应 color 文件夹下的所有 color_xxxx.jpeg 文件
        color_folder = os.path.join(root, "Video_files", video_name, "color")
        frame_pattern = os.path.join(color_folder, "color_*.jpeg")
        frame_files = sorted(glob.glob(frame_pattern))  # 按文件名排序

        # 如果帧数不足，填充 0
        frame_files = pad_image_paths(frame_files, max_frame_num=frame_num)

        obj_rgb_paths.append(frame_files)

    return obj_rgb_paths


def save_to_pkl(root, dest):
    train_df,test_df,_ = read_split(root)
    split = ['train','test']
    for i in split:
        if i == 'train':
            data = generate_obj_rgb_paths(root, train_df, frame_num=120)
        if i == 'test':
            data = generate_obj_rgb_paths(root,test_df,frame_num=120)
        """将数据保存为 .pkl 文件"""
        with open(os.path.join(dest,i,'obj_rgb_paths.pkl'), 'wb') as f:
            pickle.dump(data, f)




def get_object_keypoints_from_ply(mesh, object_pose):
    """
    从物体 .ply 模型中提取 21 个关键点（角点、边中心、中心点），并应用 object pose。

    参数:
    - ply_path: .ply 文件路径
    - object_pose: (4, 4) 的位姿矩阵，numpy 格式

    返回:
    - (21, 3) numpy 数组，变换后关键点
    """
    verts = np.array(mesh.vertices) * 1000  # m → mm

    # 得到 bbox 8 个角点
    bbox = mesh.bounding_box_oriented.vertices  # (8, 3)
    bbox = bbox * 1000  # m → mm
    corners = np.unique(bbox, axis=0)

    # 12 条边的中心点
    edge_indices = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                    (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    edge_centers = [(corners[i] + corners[j]) / 2.0 for i, j in edge_indices]

    # 中心点
    center = np.mean(corners, axis=0, keepdims=True)

    keypoints = np.concatenate([corners, edge_centers, center], axis=0)  # (21, 3)

    # 变换
    hom_kps = np.concatenate([keypoints, np.ones((21, 1))], axis=1)  # (21, 4)
    transformed = object_pose @ hom_kps.T  # (4, 21)
    transformed = transformed.T[:, :3]  # (21, 3)

    return transformed


# read one split
def read_split(root):
    path = os.path.join(root, 'data_split_action_recognition.txt')
    action_object_info_path = os.path.join(root, 'action_object_info.txt')

    train_data = []
    test_data = []

    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Training') or line.startswith('Test'):
            current_split, count = line.split()
            count = int(count)
            subset = lines[i + 1:i + 1 + count]
            for entry in subset:
                video_path, label = entry.strip().split()
                entry_dict = {
                    'path': video_path,
                    'action_label': int(label)
                }
                if current_split.lower() == 'training':
                    train_data.append(entry_dict)
                elif current_split.lower() == 'test':
                    test_data.append(entry_dict)
            i += count + 1
        else:
            i += 1

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # 读取 action-object info
    action_object_info_df = pd.read_csv(action_object_info_path, delimiter=' ', header=0)

    return train_df, test_df, action_object_info_df


# get data in single frame
def read_single_frame(root,path, max_frame_num, action_object_info_df,action_id,type='hands'):
      # 确保你本地装了 trimesh
    if type == 'hands':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')
        num_frames = df.shape[0]
        assert df.shape[1] == 64, f"Expected 64 columns (1 frame_id + 63 coords), got {df.shape[1]}"
        temp_tensor = torch.zeros((num_frames, 1, 21, 3))  # (T, M, V, C)
        for t in range(num_frames):
            row = df.iloc[t, 1:]
            coords = row.values.reshape(21, 3)
            temp_tensor[t, 0] = torch.tensor(coords, dtype=torch.float)
        return temp_tensor, num_frames
    elif type == 'object':
        if path == '0' or not os.path.exists(path):
            return torch.zeros((max_frame_num, 1, 21, 3)), -1
        try:
            assert action_object_info_df is not None and action_id is not None
            object_name = action_object_info_df[
                action_object_info_df['action_id'] == action_id + 1  # +1 因为文件中从1开始
                ]['object_name'].values[0]
            # 构造 .ply 路径
            print('object_name is {}'.format(object_name))
            ply_path = os.path.join(root,'Object_models', f'{object_name}_model', f'{object_name}_model.ply')
        except Exception as e:
            print(f"[错误] 获取 object_name 失败: {e}")
        # 生成关键点
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')
        num_frames = df.shape[0]
        assert df.shape[1] == 17
        temp_tensor = torch.zeros((num_frames, 1, 21, 3), dtype=torch.float32)

        mesh = trimesh.load(ply_path)

        for t in range(num_frames):
            pose_vals = df.iloc[t, 1:].values.astype(np.float32)
            trans_mat = pose_vals.reshape(4, 4).T  # (4, 4)
            keypoints = get_object_keypoints_from_ply(mesh, trans_mat)  # (21, 3)
            temp_tensor[t, 0] = torch.tensor(keypoints, dtype=torch.float32)
        return temp_tensor, None

# pad tensor according to args.frames
def pad_tensor(sample_tensor, max_frame_num=120):
    if sample_tensor.size(0) < max_frame_num:
        zero_tensor = torch.zeros((max_frame_num - sample_tensor.size(0), sample_tensor.size(1), sample_tensor.size(2),
                                   sample_tensor.size(3)))
        sample_tensor = torch.cat([sample_tensor, zero_tensor], dim=0)

    if sample_tensor.size(0) > max_frame_num:
        st = (sample_tensor.size(0) - max_frame_num) // 2
        sample_tensor = sample_tensor[st:st + max_frame_num, :, :, :]

    return sample_tensor


def pad_image_paths(paths, max_frame_num=120):
    """填充图像路径，与pad_tensor逻辑一致"""
    if len(paths) < max_frame_num:
        # 用最后一个有效路径填充
        paths.extend([0] * (max_frame_num - len(paths)))
    elif len(paths) > max_frame_num:
        # 中心裁剪
        st = (len(paths) - max_frame_num) // 2
        paths = paths[st:st + max_frame_num]
    return paths


# get data of one sample
def read_sample(root,path,frame_num,action_object_info_df,action_id, type='hands'):
    frame_list = []

    frame_tensor, current_frame_num = read_single_frame(root,path,frame_num,action_object_info_df ,action_id,type)
    frame_list.append(frame_tensor)
    sample = torch.stack(frame_list).squeeze(0)
    return sample ,current_frame_num


# main func
def get_fpha(root, split, frame_num=120):
    tqdm_desc = 'Get fpha dataset(s)'
    # create train subset
    train_df, test_df, action_object_info_df = read_split(root)

    sample_list = []
    obj_label_list = []
    action_label_list = []
    verb_label_list = []

    with tqdm(total=len(train_df), desc=tqdm_desc, ncols=100) as pbar:
        for i in range(len(train_df)):
            action_id = train_df.at[i, 'action_label']
            path = train_df.at[i, 'path']
            p_hands = os.path.join(root,'Hand_pose_annotation_v1', train_df.at[i, 'path'], 'skeleton.txt')
            # 判断一下这个动作是否有obj pose 没有填0
            object_pose_flag = action_object_info_df[action_object_info_df['action_id'] == action_id+1]['object_pose'].values[0]

            if object_pose_flag == 0:
                p_obj = '0'
            else:
                p_obj = os.path.join(root, 'Object_6D_pose_annotation_v1_1', path, 'object_pose.txt')
                if os.path.exists(p_obj) is False:
                    p_obj = '0'

            hand_sample,current_frame_num = read_sample(root,p_hands,frame_num,action_object_info_df,action_id, 'hands')
            obj_sample,_ = read_sample(root,p_obj,current_frame_num ,action_object_info_df,action_id,'object')
            temp = torch.cat([hand_sample, obj_sample], dim=1)

            sample_list.append(pad_tensor(temp, frame_num))
            action_label_list.append(action_id)
            pbar.update(1)

    # N, T, M, V, C
    train_data = torch.stack(sample_list, dim=0)
    train_gt = torch.tensor(action_label_list)
    # N, T, M, V, C ->  N, C, T, V, M
    train_data = train_data.permute(0, 4, 1, 3, 2)

    # create test subset
    sample_list = []
    obj_label_list = []
    action_label_list = []
    verb_label_list = []
    with tqdm(total=len(test_df), desc=tqdm_desc, ncols=100) as pbar:
        for i in range(len(test_df)):
            action_id = test_df.at[i, 'action_label']
            path = test_df.at[i, 'path']
            p_hands = os.path.join(root, 'Hand_pose_annotation_v1', test_df.at[i, 'path'], 'skeleton.txt')
            # 判断一下这个动作是否有obj pose 没有填0
            object_pose_flag = action_object_info_df[action_object_info_df['action_id'] == action_id + 1]['object_pose'].values[0]

            if object_pose_flag == 0:
                p_obj = '0'
            else:
                p_obj = os.path.join(root, 'Object_6D_pose_annotation_v1_1', path, 'object_pose.txt')
                if os.path.exists(p_obj) is False:
                    p_obj = '0'

            hand_sample, current_frame_num = read_sample(root,p_hands, frame_num,action_object_info_df,action_id, 'hands')
            obj_sample ,_= read_sample(root,p_obj, current_frame_num, action_object_info_df,action_id,'object')
            temp = torch.cat([hand_sample, obj_sample], dim=1)

            sample_list.append(pad_tensor(temp, frame_num))
            action_label_list.append(action_id)
            pbar.update(1)

    # N, T, M, V, C
    test_data = torch.stack(sample_list, dim=0)
    # N
    test_gt = torch.tensor(action_label_list)

    # N, T, M, V, C ->  N, C, T, V, M
    test_data = test_data.permute(0, 4, 1, 3, 2)
    return train_data,test_data, train_gt,test_gt
    raise NotImplementedError('data split only supports train/val/test/all')

    # N, C, T, V, M


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate FPHA pth files")
    parser.add_argument('--root', type=str, help='Path to downloaded files.', default='downloads')
    parser.add_argument('--dest', type=str, help='Destination path to save pth files.', default='fpha_pth')
    parser.add_argument('--frames', type=int, help='Input frame numbers', default=120)#暂定120

    args = parser.parse_args()
    #保存场景RGB图路径的函数
    save_to_pkl(args.root, args.dest)

    # 保存val
    #TODO fpha dataset lack of val subset moreover the split file consist of train an test file path
    #保存train
    # train_data,test_data, train_gt,test_gt = get_fpha(args.root, args.frames)
    # torch.save(train_data.clone(), os.path.join(args.dest, 'train', 'data.pth'))
    # torch.save(train_gt.clone(), os.path.join(args.dest, 'train', 'gt.pth'))
    # torch.save(test_data.clone(), os.path.join(args.dest, 'test', 'data.pth'))
    # torch.save(test_gt.clone(), os.path.join(args.dest, 'test', 'gt.pth'))