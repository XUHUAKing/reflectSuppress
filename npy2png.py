'''
convert .npy to .png (non-hierachical format)

steps:
    1. download ./crop_npy folder
    2. run this script python npy2png.py
    3. output will be stored in ./save_png
    4. run matlab script test.m
'''
import numpy as np
from glob import glob
import cv2
import os

datadir = './crop_npy'
savedir = './save_png'
val_items = []
folders1 = glob(datadir+'/*')
folders2 = []
for folder1 in folders1:
    folders2 = folders2 + glob(folder1+'/Indoor/*') + glob(folder1+'/Outdoor/*')
folders2.sort()
for folder2 in folders2[::5]:
    folder = folder2
    imgs = glob(folder + '/*.npy')
    imgs.sort()
    print(folder, len(imgs))
    for idx in range(len(imgs)//2):
        tmp_M = imgs[2*idx+1]
        val_items.append(tmp_M)
        print(tmp_M)
print('Data load succeed!')

for m_name in val_items:
    import pdb; pdb.set_trace()
    m_img = 0.5*np.load(m_name)[0,:,:,-1]
    h,w = m_img.shape[:2]
    h = h // 32 *32
    w = w // 32 *32
    m_img = m_img[:h:2, :w:2]
    # crop between 0, 1
    m_img[m_img<0] = 0
    m_img[m_img>1] = 1

    # gamma correction
    m_img = np.power(m_img[:,:,np.newaxis], 1/2.2)
    m_img = np.tile(m_img,[1,1,3])
    # new png name
    # save_path = m_name.replace(datadir, savedir).replace('.npy', '.png')
    save_path = m_name.replace('.npy', '.png')
    path_list = save_path.split('/')
    img_name = path_list[-4]+'-'+path_list[-3]+'-'+path_list[-2]+'-'+path_list[-1]
    save_path = savedir + '/' + img_name
    # scale back to 65535
    # save_img = cv2.cvtColor((m_img * 255.).clip(0,65535), cv2.COLOR_BGR2GRAY).astype(np.uint16) # np.uint16 for save, np.uint8 for visualize
    save_img = (m_img * 65535.).clip(0,65535).astype(np.uint16) # save as RGB
    # if not os.path.exists(save_path[:-12]):
    #     os.makedirs(save_path[:-12])
    cv2.imwrite(save_path, save_img)
