import numpy as np
import open3d as o3d
import json
import cv2

FX = 607.88
FY = 607.966
CX = 641.875
CY = 365.585

DEPTH_SCALE = 1000.0

DEPTH_TRUNC = 2.0

with open("calibrated_dataset.json") as file:
    calibrated_data = json.load(file)

def load_hand_eye_calibration(filename="hand_eye_calibration_result.npz"):
    data = np.load(filename)
    T_G2C = data["hand_eye_transform"]  # shape (4,4)
    return T_G2C

def get_robot_pose_for_frame(i):
    """
    根据帧号 i 获取末端相对于基座的 4x4 矩阵.
    这里仅举例，实际应从机器人接口/记录数据中得到.
    """
    # 假设都已事先存好或者实时获取
    tmp = np.array(calibrated_data[i]["RT"]).reshape(4, 4)
    return tmp

def create_pcd_from_rgbd(color_image, depth_image, fx, fy, cx, cy):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=DEPTH_SCALE,     # 根据你的深度图单位适当调整
        depth_trunc=DEPTH_TRUNC,       # 2米截断，可根据需求修改
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd

def load_rgbd(index):
    # 1) 取出彩色图路径
    color_img_path = calibrated_data[index]["image_path"]
    # 2) 替换字符串得到深度图路径
    depth_img_path = color_img_path.replace("rgb_images", "depth_images")

    # 3) 用 OpenCV 读取彩色图 (默认是 BGR 格式)
    color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)
    # 如果后续需要 RGB 顺序，可以再转换
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # 4) 用 OpenCV 读取深度图
    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)

    return color_img, depth_img

def main():
    # 1) 加载手眼标定矩阵 (gripper->camera)
    T_G2C = load_hand_eye_calibration("hand_eye_calibration_result.npz")

    # 2) 遍历每一帧 RGBD
    pcd_global = o3d.geometry.PointCloud()
    num_frames = 219  # 假设你有 10 帧数据
    for i in range(num_frames):
        color_image, depth_image = load_rgbd(i)
        pcd_camera = create_pcd_from_rgbd(color_image, depth_image, FX, FY, CX, CY)

        # 3) 从机器人获取本帧末端姿态 (base->gripper)
        T_B2G_i = get_robot_pose_for_frame(i)

        # 4) 计算 (base->camera)
        T_B2C_i = T_B2G_i @ T_G2C

        # 5) 将该帧点云变换到基座坐标系
        pcd_camera.transform(T_B2C_i)

        # 6) 累加到全局点云
        pcd_global += pcd_camera

    # 7) 保存最终点云
    o3d.io.write_point_cloud("final_map.pcd", pcd_global)

    # 8) 交互式查看最终融合后的点云
    o3d.visualization.draw_geometries(
        [pcd_global],
        window_name="Fused Point Cloud Viewer",
        width=1280,
        height=720
    )

if __name__ == "__main__":
    main()
