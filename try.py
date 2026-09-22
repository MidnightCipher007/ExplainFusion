from torchvision.datasets import ImageFolder
data = ImageFolder("dataset/train")
print(data.classes)
print(data.class_to_idx)