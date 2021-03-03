import numpy as np
from sklearn.metrics import pairwise_distances
import nibabel as nib
from copy import deepcopy

## Seperate unique values into own arrays
def separate_unique(array, binarize=True):
    arrays = []
    for j in np.unique(array[array.nonzero()]):
        a1 = array.copy()
        a1[a1 != j] = 0 
        arrays.append(a1)
        
    arrays = np.vstack(arrays)
    
    if binarize:
        arrays[arrays != 0] = 1
    return arrays

def binarize_nib(img):
    img = deepcopy(img)
    img.get_data()[img.get_data() != 0] = 1
    return img

# Calculate distance between each cluster
def pairwise_unique(a_img, b_img, metric='dice'):
    a_img = separate_unique(a_img)
    b_img = separate_unique(b_img)
    return pairwise_distances(a_img, b_img, metric=metric)

# Given a dictionary, map values in an array 
def convert_array(data, replace):
    mp = np.arange(0,data.max()+1)
    mp[np.array(replace.keys())] = replace.values()
    return mp[data]

def get_common_voxels(nib1, nib2):
    data1 = nib1.get_data().squeeze()
    data2 = nib2.get_data().squeeze()
    ix = (data1 != 0) & (data2 != 0)
    return data1[ix], data2[ix]

def match_parcellations(nib1, nib2):
	""" Match parcells in nib2 to nib1, and return converted nib2 """

	data1 = nib1.get_data().flatten()
	data2 = nib2.get_data().flatten()
	distances = pairwise_unique(data1, data2)

	min_match = np.apply_along_axis(lambda x: np.where(x == x.min())[0][0], 0, distances)
	conversion_index = dict(zip(range(1, np.nonzero(np.unique(data1))[0].shape[0]+1), min_match+1))

	converted_nib2 = nib.Nifti1Image(
	    convert_array(nib2.get_data(), conversion_index), nib2.get_affine(), nib2.get_header())

	distances = pairwise_unique(data1.flatten(), converted_nib2.get_data().flatten())

	return distances, conversion_index, converted_nib2
