from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
import matplotlib.pyplot as plt

# filename = '/media/guan/My Passport/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz'
# img = nib.load(filename)
# print(img)
# print(img.dataobj.shape)
# width, height, queue = img.dataobj.shape
# OrthoSlicer3D(img.dataobj).show() # show the 3d image
# x = int((queue/10)**0.5) + 1
# num = 1
# plt.figure()
# for i in range(0,queue,10):
#     img_arr = img.dataobj[:,:,i]
#     plt.subplot(x,x,num)
#     plt.imshow(img_arr,cmap='gray')
#     num += 1

filename2 = 'data/dataset/BRATS2018/training/HGG/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz'
seg = nib.load(filename2)
width,height,queue = seg.dataobj.shape
x2 = int((queue/10)**0.5) + 1
num = 1
plt.figure()
for i in range(0,queue,10):
    img_arr = seg.dataobj[:,:,i]
    plt.subplot(x2,x2,num)
    plt.imshow(img_arr)
    num += 1
plt.show()
seg


