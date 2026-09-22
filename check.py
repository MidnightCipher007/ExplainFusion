from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("dataset/test/NORMAL/IM-0039-0001.jpeg")
plt.imshow(img, cmap='gray')
plt.title("Check if truly normal")
plt.show()