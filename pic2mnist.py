from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import sys
import serial

# 读取图片转成灰度格式
img = Image.open(sys.argv[1]).convert('L')
# resize的过程
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))

# 暂存像素值的一维数组
arr = []

for i in range(28):
    for j in range(28):
        # mnist 里的颜色是0代表白色（背景），1.0代表黑色
        pixel = int((1.0 - float(img.getpixel((j, i)))/255.0)*127)
        #pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
        arr.append(pixel)
arr1 = np.array(arr).reshape((1, 28, 28, 1))
f = open('content.txt', 'w')
#f.write('#define IMG%d {'% (1))

np.round(arr1.flatten()).tofile(f, sep=" ", format="%02x")
#f.write('} \n')
f.close()
#print(arr)
#print(arr1)

