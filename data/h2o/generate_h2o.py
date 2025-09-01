import torch
import os
from tqdm import tqdm
import pandas as pd
import argparse
import pickle


def generate_obj_rgb_paths(root, df, frame_num=120):
    """
    生成 obj_rgb 图片路径，并根据帧数填充 0。

    参数:
        root (str): 数据集根目录。
        df (pd.DataFrame): 包含路径和帧范围的数据框。
        frame_num (int): 定长帧数，默认为 120。

    返回:
        obj_rgb_paths (list): 生成的图片路径列表。
    """
    obj_rgb_paths = []

    for i in tqdm(range(len(df))):
        path = df.at[i, 'path']
        start_act = df.at[i, 'start_act']
        end_act = df.at[i, 'end_act']

        # 生成图片路径
        frame_paths = []
        for frame_idx in range(start_act, end_act + 1):
            frame_id = f"{frame_idx:06d}.png"
            frame_path = os.path.join(root, path, "cam4", "rgb", frame_id)
            frame_paths.append(frame_path)

        # 如果帧数不足，填充 0
        frame_paths = pad_image_paths(frame_paths,max_frame_num=frame_num)

        obj_rgb_paths.append(frame_paths)

    return obj_rgb_paths
def save_to_pkl(root,split,save_path):
    df = read_split(root, split, False)
    data = generate_obj_rgb_paths(root,df,frame_num=120)
    """将数据保存为 .pkl 文件"""
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

# find the acton label of one clip in txt
def find_action_label(root, sample_short_path, vaild_start_frame):
    id = (6-len(str(vaild_start_frame))) * '0' + str(vaild_start_frame)
    with open(os.path.join(root, sample_short_path, 'cam4', 'action_label', id+'.txt'), 'r') as f:
        res = f.readline().strip()
    
    return int(res)

# read one split
def read_split(root, split, with_label=True):
    if split == 'all':
        df = pd.concat([read_split(root, 'train', False), read_split(root, 'val', False), read_split(root, 'test', False)])
    elif split == 'train' or split == 'val' or split == 'test':
        path = os.path.join(root, 'label_split', 'action_'+split+'.txt')
        # action_train.txt
        df = pd.read_csv(path, delimiter=' ', header=0)
        
        if split == 'train' and with_label:
            if not df[df['action_label'].isin([0])].empty:
                print(df[df['action_label'].isin([0])])
            # df['verb_label'] = df.apply(lambda df: action2verb(df['action_label']), axis=1)#action_label->verb_label
        elif split == 'val' and with_label:
            df['action_label'] = df.apply(lambda df: find_action_label(root, df['path'], df['start_act']), axis=1)
            # df['verb_label'] = df.apply(lambda df: action2verb(df['action_label']), axis=1)

        df['valid_frame_len'] = df['end_act'] - df['start_act'] + 1#有效动作帧长
    else:
        raise NotImplementedError('data split only supports train/val/test/all')

    return df

# get data in single frame
def read_single_frame(path, type='hands'):
    if type=='hands':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')

        assert df.shape == (1, 128)

        temp_tensor = torch.zeros((2, 21, 3)) # M, V, C

        for i in range(64):
            if i == 0:
                if df.iloc[0,i] == 0:
                    break
            else:
                temp_tensor[0][(i-1) // 3][(i-1) % 3] = df.iloc[0,i]

        for i in range(64, 128):
            if i == 64:
                if df.iloc[0,i] == 0:
                    break
            else:
                k = i - 64
                temp_tensor[1][(k-1) // 3][(k-1) % 3] = df.iloc[0,i]
        return temp_tensor,None  # M, V, C

    elif type=='object':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')

        assert df.shape == (1, 64)

        temp_tensor = torch.zeros((1, 21, 3)) # M, V, C

        for i in range(64):
            if i == 0:
                continue
            else:
                temp_tensor[0][(i-1) // 3][(i-1) % 3] = df.iloc[0,i]
        obj_label = df.iloc[0, 0]
        return temp_tensor,obj_label


# pad tensor according to args.frames
def pad_tensor(sample_tensor, max_frame_num=120):
    if sample_tensor.size(0) < max_frame_num:
        zero_tensor = torch.zeros((max_frame_num-sample_tensor.size(0), sample_tensor.size(1), sample_tensor.size(2), sample_tensor.size(3)))
        sample_tensor = torch.cat([sample_tensor, zero_tensor], dim=0)

    if sample_tensor.size(0) > max_frame_num:
        st = (sample_tensor.size(0)-max_frame_num)//2
        sample_tensor = sample_tensor[st:st+max_frame_num,:,:,:]
    
    return sample_tensor

def pad_image_paths(paths,max_frame_num=120):
    """填充图像路径，与pad_tensor逻辑一致"""
    if len(paths) < max_frame_num:
        # 用最后一个有效路径填充
        paths.extend([0] * (max_frame_num - len(paths)))
    elif len(paths) >max_frame_num:
        # 中心裁剪
        st = (len(paths) - max_frame_num) // 2
        paths = paths[st:st + max_frame_num]
    return paths

# get data of one sample
def read_sample(path, start, end, type='hands'):
    frame_list = []
    obj_label = None

    for i in range(start, end+1):
        id = (6-len(str(i))) * '0' + str(i)
        frame_tensor, current_obj_label = read_single_frame(os.path.join(path, id + '.txt'), type)
        frame_list.append(frame_tensor)
        if current_obj_label is not None:
            obj_label = current_obj_label
    sample = torch.stack(frame_list, dim=0)
    if type == 'object':
        return sample, torch.tensor(obj_label)  # Return both pose tensor and obj_labels
    else:
        return sample, None  # For hands, obj_labels is None # T, M, V, C

# main func
def get_H2O(root, split, frame_num=120):
    tqdm_desc = 'Get H2O '+split+' set(s)'
    if split == 'test' or split == 'all':
        # test split has no label
        df = read_split(root, split, False)

        sample_list = []
        obj_label_list = []

        with tqdm(total=len(df), desc=tqdm_desc, ncols=100) as pbar:
            for i in range(len(df)):
                p_hands = os.path.join(root, df.at[i, 'path'], 'cam4', 'hand_pose')
                p_obj = os.path.join(root, df.at[i, 'path'], 'cam4', 'obj_pose')
                st = df.at[i, 'start_act']
                end = df.at[i, 'end_act']
                hand_sample, _ = read_sample(p_hands, st, end, 'hands')
                obj_sample, obj_label = read_sample(p_obj, st, end, 'object')
                temp = torch.cat([hand_sample, obj_sample], dim=1)
                sample_list.append(pad_tensor(temp, frame_num))
                obj_label_list.append(obj_label)
                pbar.update(1)

        # N, T, M, V, C
        data = torch.stack(sample_list, dim=0)
        ground_truth = None

        # N, T, M, V, C ->  N, C, T, V, M
        data = data.permute(0, 4, 1, 3, 2)
        obj_labels_tensor = torch.stack(obj_label_list, dim=0)  # N, T
        return data, ground_truth, obj_labels_tensor

    elif split == 'train' or split == 'val':
        df = read_split(root, split, True)

        sample_list = []
        label_list = []
        obj_label_list = []
        with tqdm(total=len(df), desc=tqdm_desc, ncols=100) as pbar:
            for i in range(len(df)):
                p_hands = os.path.join(root, df.at[i, 'path'], 'cam4', 'hand_pose')
                p_obj = os.path.join(root, df.at[i, 'path'], 'cam4', 'obj_pose')
                st = df.at[i, 'start_act']
                end = df.at[i, 'end_act']
                # Read hand and object data
                hand_sample, _ = read_sample(p_hands, st, end, 'hands')
                obj_sample, obj_labels = read_sample(p_obj, st, end, 'object')
                # Concatenate hand and object data
                temp = torch.cat([hand_sample, obj_sample], dim=1)
                sample_list.append(pad_tensor(temp, frame_num))
                label_list.append(df.at[i, 'action_label'])
                obj_label_list.append(obj_labels)
                pbar.update(1)

        # N, T, M, V, C
        data = torch.stack(sample_list, dim=0)
        # N
        ground_truth = torch.tensor(label_list)

        # N, T, M, V, C ->  N, C, T, V, M
        data = data.permute(0, 4, 1, 3, 2)
        obj_labels_tensor = torch.stack(obj_label_list, dim=0)  # N, T
        return data, ground_truth, obj_labels_tensor
    else:
        raise NotImplementedError('data split only supports train/val/test/all')

    # N, C, T, V, M

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate H2O pth files")
    parser.add_argument('--root', type=str, help = 'Path to downloaded files.', default='downloads')
    parser.add_argument('--dest', type=str, help = 'Destination path to save pth files.', default='h2o_pth')
    parser.add_argument('--frames', type=int, help = 'Input frame numbers', default=120)

    args = parser.parse_args()
    save_to_pkl(args.root,'test',os.path.join(args.dest, 'test', 'obj_rgb_paths.pkl'))
    # a, b, obj_labels_val = get_H2O(args.root, 'val', args.frames)
    # print(a.shape)
    # print(b.shape)
    # print(obj_labels_val.shape)
    #
    # torch.save(a.clone(), os.path.join(args.dest, 'val', 'data.pth'))
    # torch.save(b.clone(), os.path.join(args.dest, 'val', 'gt.pth'))
    # torch.save(obj_labels_val.clone(), os.path.join(args.dest, 'val', 'obj_labels.pth'))
    #
    # c, d, obj_labels_train = get_H2O(args.root, 'train', args.frames)
    # print(c.shape, d.shape, obj_labels_train.shape)
    # torch.save(c.clone(), os.path.join(args.dest, 'train', 'data.pth'))
    # torch.save(d.clone(), os.path.join(args.dest, 'train', 'gt.pth'))
    # torch.save(obj_labels_train.clone(), os.path.join(args.dest, 'train', 'obj_labels.pth'))
    #
    # # Generate and save test set
    # e, _, obj_labels_test = get_H2O(args.root, 'test', args.frames)
    # print(e.shape, obj_labels_test.shape)
    # torch.save(e.clone(), os.path.join(args.dest, 'test', 'data.pth'))
    # torch.save(obj_labels_test.clone(), os.path.join(args.dest, 'test', 'obj_labels.pth'))