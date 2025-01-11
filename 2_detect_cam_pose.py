import json
import cv2
import numpy as np
import os

def camera_calibrate(image_paths, save_visualizations=False, output_dir="calibrations_visualizations"):
    print("++++++++++开始相机标定++++++++++++++")
    
    # 角点的个数以及棋盘格间距
    XX = 11  # 标定板的水平角点个数
    YY = 8  # 标定板的垂直角点个数
    L = 0.03  # 标定板一格的长度，单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.01)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 世界坐标系
    objp *= L  # 以实际尺寸缩放

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    valid_image_paths = []  # 存储有效图片路径

    if save_visualizations:
        os.makedirs(output_dir, exist_ok=True)

    for idx, image_path in enumerate(image_paths):
        print(f"正在处理第 {idx+1}/{len(image_paths)} 张图片：{image_path}")

        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片：{image_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:
                print(f"找到棋盘格角点，开始优化角点位置。")

                # 在找到的角点基础上寻找亚像素角点
                corners2 = cv2.cornerSubPix(
                    gray, corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
                )

                obj_points.append(objp)
                img_points.append(corners2)
                valid_image_paths.append(image_path)

                if save_visualizations:
                    # 绘制角点并保存图像
                    img_with_corners = cv2.drawChessboardCorners(img.copy(), (XX, YY), corners2, ret)
                    visualization_path = os.path.join(output_dir, f"corners_{idx+1}.png")
                    cv2.imwrite(visualization_path, img_with_corners)
                    print(f"角点可视化已保存到：{visualization_path}")
            else:
                print(f"未在图片 {image_path} 中找到棋盘格角点。")
        else:
            print(f"图片路径不存在：{image_path}")

    if save_visualizations:
        print(f"所有角点可视化已保存到目录：{output_dir}")

    if not valid_image_paths:
        print("未检测到任何有效的棋盘格角点，无法进行相机标定。")
        return None, None, None, None, []

    # 标定相机
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    if ret:
        print("相机标定成功！")
        print("内参矩阵 (Camera Matrix):\n", mtx)
        print("畸变系数 (Distortion Coefficients):\n", dist)
    else:
        print("相机标定失败。")

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs, mtx, dist, valid_image_paths

def rvec_tvec_to_RT(rvec, tvec):
    """
    将旋转向量和位移向量转换为4x4的RT矩阵。
    """
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = tvec.flatten()
    return RT.tolist()

def main():
    dataset_json_path = "dataset.json"

    # 检查 dataset.json 是否存在
    if not os.path.exists(dataset_json_path):
        print(f"找不到文件 {dataset_json_path}。请确保文件存在。")
        return

    # 读取 dataset.json
    with open(dataset_json_path, "r") as f:
        try:
            dataset = json.load(f)
        except json.JSONDecodeError as e:
            print(f"读取 JSON 文件时出错：{e}")
            return

    # 提取所有 image_path
    image_paths = [entry["image_path"] for entry in dataset]

    # 验证所有图片路径是否存在
    missing_images = [path for path in image_paths if not os.path.exists(path)]
    if missing_images:
        print("以下图片路径不存在，请检查路径是否正确：")
        for path in missing_images:
            print(path)
        # 继续处理存在的图片
        image_paths = [path for path in image_paths if os.path.exists(path)]
        if not image_paths:
            print("没有可用的图片进行相机标定。")
            return

    # 调用 camera_calibrate 函数
    # 如果需要保存角点可视化图像，请将 save_visualizations 设置为 True
    rvecs, tvecs, mtx, dist, valid_image_paths = camera_calibrate(image_paths, save_visualizations=False)

    if rvecs is not None and tvecs is not None:
        # 保存标定结果到 JSON 文件
        calibration_result = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "rotation_vectors": [r.tolist() for r in rvecs],
            "translation_vectors": [t.tolist() for t in tvecs]
        }

        calibration_result_path = "calibration_result.json"
        with open(calibration_result_path, "w") as f:
            json.dump(calibration_result, f, indent=4)
        print(f"相机标定结果已保存到 '{calibration_result_path}'。")

        # 将 rvec 和 tvec 转换为 RT 矩阵，并更新 dataset.json
        print("++++++++++将标定的RT矩阵写入dataset.json++++++++++++++")
        filtered_dataset = []
        for image_path, rvec, tvec in zip(valid_image_paths, rvecs, tvecs):
            # 从原始 dataset 中找到匹配的条目
            entry = next((item for item in dataset if item["image_path"] == image_path), None)
            if entry:
                RT = rvec_tvec_to_RT(rvec, tvec)
                entry["calibration_RT"] = RT
                filtered_dataset.append(entry)

        # 将更新后的数据写回 dataset.json
        with open("calibrated_dataset.json", "w") as f:
            json.dump(filtered_dataset, f, indent=4)
        print(f"已将标定的RT矩阵写入到 'calibrated_dataset.json' 中。")
    else:
        print("相机标定未成功，未保存标定结果。")

if __name__ == "__main__":
    main()
