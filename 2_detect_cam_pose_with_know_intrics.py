import json
import cv2
import numpy as np
import os

def estimate_extrinsics_with_known_intrinsics(
    image_paths,
    camera_matrix,
    dist_coeffs,
    save_visualizations=False,
    output_dir="calibrations_visualizations"
):
    """
    使用已知的相机内参 (camera_matrix, dist_coeffs)，通过棋盘格进行外参估计。
    返回每张图像对应的旋转向量和平移向量列表，以及有效图像路径。
    """
    print("++++++++++开始相机外参估计++++++++++++++")

    XX = 11  # 棋盘格水平角点个数
    YY = 8   # 棋盘格垂直角点个数
    L = 0.03 # 每个方格的物理尺寸（单位：米）

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.01)

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp *= L

    all_rvecs = []
    all_tvecs = []
    valid_image_paths = []

    if save_visualizations:
        os.makedirs(output_dir, exist_ok=True)

    for idx, image_path in enumerate(image_paths):
        print(f"正在处理第 {idx+1}/{len(image_paths)} 张图片：{image_path}")

        if not os.path.exists(image_path):
            print(f"图片路径不存在：{image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片：{image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

        if ret:
            print("找到棋盘格角点，开始优化角点位置。")
            corners2 = cv2.cornerSubPix(
                gray, corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
            )

            success, rvec, tvec = cv2.solvePnP(
                objp,
                corners2,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                all_rvecs.append(rvec)
                all_tvecs.append(tvec)
                valid_image_paths.append(image_path)

                if save_visualizations:
                    img_with_corners = cv2.drawChessboardCorners(
                        img.copy(), (XX, YY), corners2, ret
                    )
                    visualization_path = os.path.join(output_dir, f"corners_{idx+1}.png")
                    cv2.imwrite(visualization_path, img_with_corners)
                    print(f"角点可视化已保存到：{visualization_path}")
            else:
                print("solvePnP 求解失败。")
        else:
            print(f"未找到棋盘格角点，跳过图片：{image_path}")

    if save_visualizations:
        print(f"所有角点可视化结果已保存到目录：{output_dir}")

    print(f"有效的棋盘格图像数量：{len(valid_image_paths)}")
    print("++++++++++相机外参估计完成++++++++++++++")

    return all_rvecs, all_tvecs, valid_image_paths

def rvec_tvec_to_RT(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = tvec.flatten()
    return RT.tolist()

def main():
    cx = 641.875
    cy = 365.585
    fx = 607.88
    fy = 607.966

    camera_matrix = np.array([
        [fx,   0.,  cx],
        [ 0.,  fy,  cy],
        [ 0.,   0.,  1.]
    ], dtype=np.float64)

    k1 = 0.458331
    k2 = -2.87024
    k3 = 1.74446
    k4 = 0.331466
    k5 = -2.66947
    k6 = 1.65473
    p1 = 0.00048617
    p2 = -9.82262e-05

    dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

    dataset_json_path = "dataset.json"
    if not os.path.exists(dataset_json_path):
        print(f"找不到文件 {dataset_json_path}。请确保文件存在。")
        return

    try:
        with open(dataset_json_path, "r") as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        print(f"读取 JSON 文件时出错：{e}")
        return

    image_paths = [entry["image_path"] for entry in dataset]

    rvecs, tvecs, valid_image_paths = estimate_extrinsics_with_known_intrinsics(
        image_paths,
        camera_matrix,
        dist_coeffs,
        save_visualizations=False
    )

    filtered_dataset = []
    for image_path, rvec, tvec in zip(valid_image_paths, rvecs, tvecs):
        # 从原始 dataset 中找到匹配的条目
        entry = next((item for item in dataset if item["image_path"] == image_path), None)
        if entry:
            RT = rvec_tvec_to_RT(rvec, tvec)
            entry["calibration_RT"] = RT
            filtered_dataset.append(entry)

    output_json_path = "calibrated_dataset.json"
    with open(output_json_path, "w") as f:
        json.dump(filtered_dataset, f, indent=4)
    print(f"已将外参写入到 '{output_json_path}' 中。")

if __name__ == "__main__":
    main()
