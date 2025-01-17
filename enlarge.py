from PIL import Image
import os

def crop_images_in_folder(input_folder, output_folder, crop_area):
    """
    裁剪指定文件夹中的所有图片并保存到输出文件夹。

    :param input_folder: 输入图片文件夹路径
    :param output_folder: 输出图片文件夹路径
    :param crop_area: 裁剪区域 (left, top, right, bottom)
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"输入文件夹 {input_folder} 不存在，请检查路径！")
        return

    total_files = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    count = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            count += 1
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size

                # 验证裁剪区域是否超出图片范围
                left, top, right, bottom = crop_area
                if left < 0 or top < 0 or right > img_width or bottom > img_height:
                    print(f"跳过文件 {filename}，裁剪区域超出图片范围：{crop_area}")
                    continue

                # 裁剪图片
                cropped_img = img.crop(crop_area)

                # 保存裁剪后的图像
                output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_cropped{os.path.splitext(filename)[1]}")
                cropped_img.save(output_path)
                print(f"[{count}/{total_files}] 已裁剪并保存图片：{filename} 到 {output_path}")
            except Exception as e:
                print(f"跳过文件 {filename}，因为加载出错：{e}")
        else:
            print(f"跳过非图片文件：{filename}")

def resize_image_to_size(image_path, output_path, target_width, target_height):
    """
    将指定图像等比放大或缩小到目标尺寸。

    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    :param target_width: 目标宽度
    :param target_height: 目标高度
    """
    try:
        img = Image.open(image_path)
        # 获取原始尺寸
        original_width, original_height = img.size

        # 按较长边等比缩放到目标尺寸
        if original_width / original_height > target_width / target_height:
            new_width = target_width
            new_height = int(original_height * (target_width / original_width))
        else:
            new_height = target_height
            new_width = int(original_width * (target_height / original_height))

        # 缩放图像
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # 将图像填充到目标尺寸（居中）
        final_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))  # 白色背景
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        final_img.paste(resized_img, (offset_x, offset_y))

        # 保存结果
        final_img.save(output_path)
        print(f"已将图像调整为指定尺寸并保存：{output_path}")
    except Exception as e:
        print(f"处理图像时出错：{e}")

# 设置输入和输出文件夹路径
input_folder = "images/input"  # 请替换为你的输入图片文件夹路径
output_folder = "images/output"  # 裁剪后的图片保存路径
resize_output_folder = "images/resized_output"  # 调整尺寸后的图片保存路径

# 确保输出文件夹存在
if not os.path.exists(resize_output_folder):
    os.makedirs(resize_output_folder)

# 自定义裁剪区域 (left, top, right, bottom)
crop_area = (16, 0, 100, 50)
crop_images_in_folder(input_folder, output_folder, crop_area)

# 等比调整裁剪后的图片尺寸
target_width = 200  # 目标宽度
target_height = 100  # 目标高度

for filename in os.listdir(output_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        cropped_image_path = os.path.join(output_folder, filename)
        resized_image_path = os.path.join(resize_output_folder, filename)
        resize_image_to_size(cropped_image_path, resized_image_path, target_width, target_height)
