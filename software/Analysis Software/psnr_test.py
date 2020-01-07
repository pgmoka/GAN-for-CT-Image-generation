
# Made by Pedro Goncalves Mokarzel
# while attending UW Bothell Student ID# 1576696
# Made in 12/09/2019
# Based on instruction in CSS 490, 
# taught by professor Dong Si

from utils import *
# These are tests used for the comparison of Case 3 and case 4. It prints in the console
# x3 and x4 are the original case 3 and case 4
# x3_c and x4_c are the created case 3 and created case 4

x3 = load_float("./Case 3/00003680_img.flt")
x3_c = load_float("./Case 3/Data/00003680_img_validate_e200_b52_img1.flt")
x4 = load_float("./Case 4/00002662_img.flt")
x4_c = load_float("./Case 4/Data/00002662_img_validate_e200_b22_img0.flt")

scalef= np.amax(x3)
x3_1 = np.clip(255 * x3/scalef, 0, 255).astype('uint8')
x3_1c = np.clip(255 * x3_c/scalef, 0, 255).astype('uint8')
print("x3->x3c: %4f"% cal_PSNR(x3_1,x3_1c))

scalef= np.amax(x4)
x4_2 = np.clip(255 * x4/scalef, 0, 255).astype('uint8')
x4_2c = np.clip(255 * x4_c/scalef, 0, 255).astype('uint8')
print("x4->x4c: %4f"% cal_PSNR(x4_2,x4_2c))

scalef= np.amax(x3)
x3_3 = np.clip(255 * x3/scalef, 0, 255).astype('uint8')
x4_3 = np.clip(255 * x4/scalef, 0, 255).astype('uint8')
print("x3->x4: %4f"% cal_PSNR(x3_3,x4_3))

scalef= np.amax(x4_c)
x3_4 = np.clip(255 * x3_c/scalef, 0, 255).astype('uint8')
x4_4 = np.clip(255 * x4_c/scalef, 0, 255).astype('uint8')
print("x4_c->x3_c: %4f"% cal_PSNR(x4_4,x3_4))

scalef= np.amax(x3)
x3_5 = np.clip(255 * x3/scalef, 0, 255).astype('uint8')
x4_5c = np.clip(255 * x4_c/scalef, 0, 255).astype('uint8')
print("x3->x4_c: %4f"% cal_PSNR(x3_5,x4_5c))

scalef= np.amax(x4)
x3_5 = np.clip(255 * x4/scalef, 0, 255).astype('uint8')
x4_5c = np.clip(255 * x3_c/scalef, 0, 255).astype('uint8')
print("x4->x3_c: %4f"% cal_PSNR(x3_5,x4_5c))





print("Shmuck")