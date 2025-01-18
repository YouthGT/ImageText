import os
import torch
from torchvision import transforms
from PIL import Image
import shutil
from Binary_Classification import SimpleCNN

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("binary_classification_model.pth", map_location=device))
model.eval()  # 设置为评估模式

# 图像变换 (与训练时保持一致)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize([0.5], [0.5])  # 归一化
])

# 需要分类的文件夹路径
input_folder = 'images/resized_output'
output_folder_1 = 'images/F-M/F'
output_folder_2 = 'images/F-M/M'

# 创建输出文件夹
os.makedirs(output_folder_1, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

# 分类函数
def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")  # 确保图像有3个通道
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# 批量处理文件夹中的所有图片
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    if os.path.isfile(image_path):  # 确保是文件而不是文件夹
        predicted_class = classify_image(image_path)
        if predicted_class == 0:
            shutil.move(image_path, os.path.join(output_folder_1, image_name))
        else:
            shutil.move(image_path, os.path.join(output_folder_2, image_name))

print("分类完成！")

