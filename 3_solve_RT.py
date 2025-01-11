import json
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_dataset(json_path):
    """
    读取 calibrated_dataset.json 文件并返回数据列表。
    """
    if not os.path.exists(json_path):
        print(f"找不到文件 {json_path}。请确保文件存在。")
        return None

    with open(json_path, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"读取 JSON 文件时出错：{e}")
            return None

def extract_rt_matrices(dataset):
    """
    从数据集中提取手臂（arm）和相机（eye）的旋转矩阵和平移向量。
    
    参数:
        dataset (list): 数据集列表，每个元素包含 "RT" 和 "calibration_RT"。

    返回:
        R_gripper2base (list of np.ndarray): 手臂相对于基座的旋转矩阵列表。
        T_gripper2base (list of np.ndarray): 手臂相对于基座的平移向量列表。
        R_target2cam (list of np.ndarray): 目标相对于相机的旋转矩阵列表。
        T_target2cam (list of np.ndarray): 目标相对于相机的平移向量列表。
    """
    R_gripper2base = []
    T_gripper2base = []
    R_target2cam = []
    T_target2cam = []

    for idx, frame_data in enumerate(dataset):
        # 提取并逆变换手臂的 RT 矩阵
        RT_arm = np.array(frame_data["RT"])
        if RT_arm.shape != (4, 4):
            print(f"第 {idx} 帧的 'RT' 矩阵形状不正确: {RT_arm.shape}")
            continue
        # RT_arm_inv = np.linalg.inv(RT_arm)
        RT_arm_inv = RT_arm
        R_arm = RT_arm_inv[:3, :3]
        T_arm = RT_arm_inv[:3, 3]
        R_gripper2base.append(R_arm)
        T_gripper2base.append(T_arm)

        # 提取相机的 RT 矩阵
        RT_eye = np.array(frame_data["calibration_RT"])
        if RT_eye.shape != (4, 4):
            print(f"第 {idx} 帧的 'calibration_RT' 矩阵形状不正确: {RT_eye.shape}")
            continue
        R_eye = RT_eye[:3, :3]
        T_eye = RT_eye[:3, 3]
        R_target2cam.append(R_eye)
        T_target2cam.append(T_eye)

    return R_gripper2base, T_gripper2base, R_target2cam, T_target2cam

def perform_hand_eye_calibration(R_gripper2base, T_gripper2base, R_target2cam, T_target2cam):
    """
    执行手眼标定，并返回标定结果的旋转矩阵和平移向量。
    
    参数:
        R_gripper2base (list of np.ndarray): 手臂相对于基座的旋转矩阵列表。
        T_gripper2base (list of np.ndarray): 手臂相对于基座的平移向量列表。
        R_target2cam (list of np.ndarray): 目标相对于相机的旋转矩阵列表。
        T_target2cam (list of np.ndarray): 目标相对于相机的平移向量列表。
    
    返回:
        R (np.ndarray): 手眼变换的旋转矩阵。
        t (np.ndarray): 手眼变换的平移向量。
    """
    # 选择标定方法，可以根据需要更改
    method = cv2.CALIB_HAND_EYE_TSAI

    # 执行手眼标定
    R, t = cv2.calibrateHandEye(
        R_gripper2base,
        T_gripper2base,
        R_target2cam,
        T_target2cam,
        method=method
    )

    # print(f"标定结果质量度量 (retval): {retval}")
    print("标定结果旋转矩阵 R:\n", R)
    print("标定结果平移向量 t:\n", t)

    return R, t

def main():
    calibrated_dataset_json_path = "calibrated_dataset.json"  # 更新为您的 calibrated_dataset.json 文件路径

    # 读取 calibrated_dataset.json
    dataset = load_dataset(calibrated_dataset_json_path)
    if dataset is None:
        return

    # 提取旋转矩阵和平移向量
    R_gripper2base, T_gripper2base, R_target2cam, T_target2cam = extract_rt_matrices(dataset)

    # 检查数据量
    num_pairs = min(len(R_gripper2base), len(R_target2cam))
    if num_pairs < 3:
        print(f"数据对数量不足（至少需要 3 对），当前有 {num_pairs} 对。")
        return

    print(f"准备进行手眼标定，使用 {num_pairs} 对姿态。")

    # 执行手眼标定
    R, t = perform_hand_eye_calibration(
        R_gripper2base[:num_pairs],
        T_gripper2base[:num_pairs],
        R_target2cam[:num_pairs],
        T_target2cam[:num_pairs]
    )

    # 构造齐次变换矩阵
    hand_eye_transform = np.eye(4)
    hand_eye_transform[:3, :3] = R
    hand_eye_transform[:3, 3] = t.flatten()  # 将 t 转换为一维数组
    print("手眼变换矩阵 (从手到眼):\n", hand_eye_transform)

    # （可选）保存标定结果到文件
    np.savez("hand_eye_calibration_result.npz", R=R, t=t, hand_eye_transform=hand_eye_transform)
    print("标定结果已保存到 'hand_eye_calibration_result.npz'")

    # （可选）可视化标定结果
    # visualize_hand_eye_transform(hand_eye_transform)

def visualize_hand_eye_transform(transform):
    """
    可视化手眼变换矩阵的旋转和位置。
    
    参数:
        transform (np.ndarray): 4x4 齐次变换矩阵。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 手眼坐标系的三个轴
    x_axis = transform[:3, 0]
    y_axis = transform[:3, 1]
    z_axis = transform[:3, 2]
    position = transform[:3, 3]

    # 绘制坐标系
    ax.quiver(*position, *x_axis, length=0.1, color='r', label='X轴')
    ax.quiver(*position, *y_axis, length=0.1, color='g', label='Y轴')
    ax.quiver(*position, *z_axis, length=0.1, color='b', label='Z轴')

    # 设置图形属性
    ax.set_xlim([position[0]-0.5, position[0]+0.5])
    ax.set_ylim([position[1]-0.5, position[1]+0.5])
    ax.set_zlim([position[2]-0.5, position[2]+0.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('手眼变换坐标系')

    plt.show()

if __name__ == "__main__":
    main()
