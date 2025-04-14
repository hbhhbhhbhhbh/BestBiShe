import imgaug.augmenters as iaa  # 导入iaa
import cv2
import glob
import os
import numpy as np
 
if __name__ == '__main__':
    dataname="data1_resized"
    filedir="test"
    img_dir = f'{dataname}/imgs/{filedir}'	# 图片文件路径
    msk_dir = f'{dataname}/masks/{filedir}'	# 标签文件路径
    #img_type = '.png'
    img_tmp_dir = f'{dataname}-enhanced/imgs/{filedir}'	# 输出图片文件路径
    msk_tmp_dir = f'{dataname}-enhanced/masks/{filedir}'
    if not os.path.exists(img_tmp_dir):
            os.makedirs(img_tmp_dir)
    if not os.path.exists(msk_tmp_dir):
            os.makedirs(msk_tmp_dir)
    img_list = os.listdir(img_dir)
    msk_list = os.listdir(msk_dir)
    print(img_list)
    print(msk_list)
    for i in range(len(img_list)):
        img_name = img_list[i]
        if(img_name.endswith(".jpg")):
            pass
        else: 
            continue
        msk_name = msk_list[i]
        if(msk_name.endswith("png")):
            pass
        else:
            continue
        img = cv2.imread(filename=img_dir + "/" + img_name)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        msk = cv2.imread(filename=msk_dir + "/" + msk_name)
        msk = np.expand_dims(msk, axis=0).astype(np.int32)
        # 定义数据增强策略
        # 每次选择一个翻转方式
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),    # 水平翻转
            iaa.Flipud(0.5),    # 垂直翻转
            # iaa.GaussianBlur(sigma=(0, 3.0)),   # 高斯模糊
            iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1)),  # 锐化处理
            iaa.Affine(scale=(0.9, 1), translate_percent=(0, 0.1), rotate=(-40, 40), cval=0, mode='constant'),   # 仿射变换
            # iaa.CropAndPad(px=(-10, 0), percent=None, pad_mode='constant', pad_cval=0, keep_size=True), # 裁剪缩放
            # iaa.PiecewiseAffine(scale=(0, 0.05), nb_rows=4, nb_cols=4, cval=0),     # 以控制点的方式随机形变
            iaa.ContrastNormalization((0.75, 1.5), per_channel=True),  # 对比度增强，0.75-1.5随机数值为alpha，该alpha应用于每个通道
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # 高斯噪声
            iaa.Multiply((0.8, 1.2), per_channel=0.2),  # 20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
        ])
        # 同时对原图和分割进行数据增强
        for j in range(6):
            img_aug, msk_aug = seq(images=img, segmentation_maps=msk)
            img_out = img_tmp_dir+"/" + img_name.split(".")[0] + "_" + str(j) + '.jpg'
            msk_out = msk_tmp_dir +"/"+ msk_name.split(".")[0] + "_" + str(j) + '.png'
            cv2.imwrite(img_out, img_aug[0])
            cv2.imwrite(msk_out, msk_aug[0,:,:,0])
        print("正在进行数据增强"+img_name+" "+msk_name)
 