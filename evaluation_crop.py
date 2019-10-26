'''
eval script for reflectSuppress results
steps:
    1. download ./crop_npy
    2. run python npy2png.py --> convert .npy to .png
    3. run test.m in matlab --> generate results
    4. run python res2folder.py --> reorganize results for evaluation
    5. cd results/ && rm *.png --> remove .png files after reorganize
    6. run evaluation_crop.py (this script) --> evaluation
'''
from imageio import imread, imsave
from glob import glob
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import cv2

def prepare_data_npy(data_path='./crop_npy/'):
    train_items, val_items = [], []
    folders1 = glob(data_path+'/*')
#    print(folders1)
    folders2 = []
    for folder1 in folders1:
        folders2 = folders2 + glob(folder1+'/Indoor/*') + glob(folder1+'/Outdoor/*')
#    print(folders2)
    folders2.sort()
    for folder2 in folders2[1::5] + folders2[2::5]+folders2[3::5]+folders2[4::5]:
        folder = folder2
        imgs = glob(folder + '/*.npy')
        imgs.sort()
        print(folder, len(imgs))
        for idx in range(len(imgs)//2):
            tmp_M = imgs[2*idx+1]
            tmp_R = imgs[2*idx]
            train_items.append([tmp_M,tmp_R])
            print(tmp_R, tmp_M)

    for folder2 in folders2[::5]:
        folder = folder2
        imgs = glob(folder + '/*.npy')
        imgs.sort()
        print(folder, len(imgs))
        for idx in range(len(imgs)//2):
            tmp_M = imgs[2*idx+1]
            tmp_R = imgs[2*idx]
            val_items.append([tmp_M,tmp_R])
            print(tmp_R, tmp_M)


    return train_items, val_items[::3]

def prepare_results(data_path='./results/'):
    pred_images = []
    folders1 = glob(data_path+'/*')
#    print(folders1)
    folders2 = []
    for folder1 in folders1:
        folders2 = folders2 + glob(folder1+'/Indoor/*') + glob(folder1+'/Outdoor/*')
#    print(folders2)
    folders2.sort()
    for folder2 in folders2:
        folder = folder2
        imgs = glob(folder + '/*.png')
        imgs.sort()
        print(folder, len(imgs))
        pred_images.extend(imgs)
    return pred_images

train_names, val_names = prepare_data_npy()
print('Data load succeed!')
print(len(train_names), len(val_names))
num_train, num_test = len(train_names), len(val_names)
def prepare_item(item):
    M_name, R_name = item
    tmp_M = np.load(M_name)
    tmp_R = np.load(R_name)
    tmp_M =0.5* np.load(M_name)[:,:,:,[4,4,4]]
    tmp_R =0.5* np.load(R_name)[:,:,:,[4,4,4]]
    #tmp_M = np.load(M_name)[:,:,:,[4,4,4]]
    #tmp_R = np.load(R_name)[:,:,:,[4,4,4]]
    print("M_mean", np.mean(tmp_M), np.max(tmp_M),tmp_M.shape)
    tmp_T = tmp_M - tmp_R
    tmp_T[tmp_T>1] = 1
    tmp_T[tmp_T<0] = 0
    return np.power(tmp_M,1/2.2), np.power(tmp_T,1/2.2), np.power(tmp_R,1/2.2)
#    return np.power(tmp_M,1), np.power(tmp_T,1), np.power(tmp_R,1)

pred_images = prepare_results()
pred_images.sort()
all_ssim, all_psnr = 0,0
for idx in range(len(pred_images)):
	all, gt, R = prepare_item(val_names[idx])
	pred = cv2.imread(pred_images[idx],-1)/65535.
    # duplicate along RGB
	pred = pred[:,:,np.newaxis]
	pred = np.tile(pred,[1,1,3])
	print(gt.shape, pred.shape)
#	gt = gt[0,::2,::2,:]
#	all = all[0,::2,::2,:]
	gt = gt[0,:,:,:]
	all = all[0,:,:,:]

	h1,w1 = all.shape[:2]
	h2,w2 = pred.shape[:2]
	h, w = min(h1,h2), min(w1, w2)
	h = h // 32 * 32
	w = w // 32 * 32
	pred = pred[:h,:w,:]
	gt = gt[:h,:w,:]
	all =all[:h,:w,:]
	print(gt.shape, pred.shape)
	print(np.mean(gt), np.mean(pred))
	'''with gamma correction inside model'''
	ssim = compare_ssim(np.power(np.mean(pred,axis=2),2.2), np.power(gt[:,:,0],2.2))
	psnr = compare_psnr(np.power(np.mean(pred,axis=2),2.2), np.power(gt[:,:,0],2.2),1)
	all_ssim += ssim
	all_psnr += psnr
	imsave('evaluation/IMG_%04d.jpg'%idx,np.concatenate([all, pred, gt],axis=1))
	print(ssim, psnr)
print(all_ssim*1.0/(idx+1), all_psnr*1.0/(idx+1))
