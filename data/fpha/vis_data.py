import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==== 关键点连线结构 ====
# 手部骨架连线结构（基于 FPHA 的手部 21 点拓扑结构）
hand_edges = [
    # 拇指 (Wrist → TMCP → TPIP → TDIP → TTIP)
    (0, 1), (1, 6), (6, 7), (7, 8),

    # 食指 (Wrist → IMCP → IPIP → IDIP → ITIP)
    (0, 2), (2, 9), (9, 10), (10, 11),

    # 中指 (Wrist → MMCP → MPIP → MDIP → MTIP)
    (0, 3), (3, 12), (12, 13), (13, 14),

    # 无名指 (Wrist → RMCP → RPIP → RDIP → RTIP)
    (0, 4), (4, 15), (15, 16), (16, 17),

    # 小指 (Wrist → PMCP → PPIP → PDIP → PTIP)
    (0, 5), (5, 18), (18, 19), (19, 20)
]

# 物体结构（单位立方体关键点）连接：先连立方体 12 条边，其他随你定义
object_edges = [
    (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6),
    (3,7), (4,5), (4,6), (5,7), (6,7)
] + [(i, 20) for i in range(20)]  # 所有点连向中心点（20号）

# ==== 可视化函数 ====
def plot_hand_object_frame(joint_3d, object_3d):
    """
    可视化单帧的手部 + 物体关键点及其连接结构
    - joint_3d: (21, 3)
    - object_3d: (21, 3)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 手关键点（蓝）
    ax.scatter(joint_3d[:, 0], joint_3d[:, 1], joint_3d[:, 2], c='blue', label='Hand', s=20)
    for i, j in hand_edges:
        ax.plot(
            [joint_3d[i, 0], joint_3d[j, 0]],
            [joint_3d[i, 1], joint_3d[j, 1]],
            [joint_3d[i, 2], joint_3d[j, 2]],
            c='blue', linewidth=1.5
        )

    # 物体关键点（红）
    ax.scatter(object_3d[:, 0], object_3d[:, 1], object_3d[:, 2], c='red', label='Object', s=20)
    for i, j in object_edges:
        ax.plot(
            [object_3d[i, 0], object_3d[j, 0]],
            [object_3d[i, 1], object_3d[j, 1]],
            [object_3d[i, 2], object_3d[j, 2]],
            c='red', linewidth=1.2, linestyle='--'
        )

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Hand and Object Keypoints with Structure')
    ax.view_init(elev=20, azim=130)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':

    data = torch.load('./fpha_pth/val/data.pth')
    gt = torch.load('./fpha_pth/val/gt.pth')
    data = data.clone()

    sample_idx = 3
    frame_idx = 10

    data_sample = data[sample_idx].permute(1, 2, 0, 3)  # T, V, C, M
    frame = data_sample[frame_idx].permute(2, 0, 1)     # M, V, C

    hand_kps = frame[0].cpu().numpy()    # (21, 3)
    object_kps = frame[1].cpu().numpy()  # (21, 3)

    plot_hand_object_frame(hand_kps, object_kps)