import json

dataset_path = "./dataset/calibrate/calibrate1_8_2"

IMAGE_COUNT = 222

rgb_path = f"{dataset_path}/rgb_images"

RT_path = f"{dataset_path}/pose_rt.txt"

with open(RT_path, "r") as file:
    rt_content = file.read()
    rt_content = rt_content.splitlines()

output_object = []

for i in range(0, IMAGE_COUNT):
    image_name = f"{i}.png"
    image_path = f"{rgb_path}/{image_name}"
    id = int(image_name.replace(".png", ""))
    rt_numbers = rt_content[id].split()
    if len(rt_numbers) != 16:
        raise ValueError("")
    float_numbers = [float(num) for num in rt_numbers]
    array_4x4 = [float_numbers[i:i+4] for i in range(0, 16, 4)]
    output_object.append({
        "image_path": image_path,
        "id": id,
        "RT": array_4x4
    })

with open("dataset.json", "w") as f:
    json.dump(output_object, f, indent=4)