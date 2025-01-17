def predict(image_path, model, transform):
    model.eval()
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# 加载模型
model.load_state_dict(torch.load("binary_classification_model.pth"))

# 单张图片预测
image_path = "path/to/image.jpg"
predicted_class = predict(image_path, model, transform)
print(f"预测类别：{predicted_class}")
