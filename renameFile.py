import os


def batch_rename_files(folder_path):
    """
    批量重命名文件，将文件名第一个下划线后的字符删除。

    :param folder_path: 文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"路径不存在: {folder_path}")
        return

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否是文件
        if os.path.isfile(file_path):
            # 找到第一个下划线的位置
            underscore_index = filename.find("_")

            # 如果找到下划线
            if underscore_index != -1:
                # 提取下划线前的部分和文件扩展名
                new_name = filename[:underscore_index] + os.path.splitext(filename)[1]
                new_path = os.path.join(folder_path, new_name)

                # 重命名文件
                os.rename(file_path, new_path)
                print(f"重命名: {filename} -> {new_name}")


if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ").strip()
    batch_rename_files(folder_path)
