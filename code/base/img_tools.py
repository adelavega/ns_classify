from neurosynth.base import imageutils 
from neurosynth.base import mask 
import numpy as np
import pandas as pd
from copy import deepcopy
import nibabel as nib

def binarize_level(img, level, output='np'):
    new_img = deepcopy(img)

    if isinstance(new_img, nib.Nifti1Image):
	    data = new_img.get_data()

    data[data != level] = 0
    data[data == level] = 1

    if output == 'np':
        return data
    elif output == 'nib':
        return new_img

def threshold_nib(img, threshold, binarize=True, output='np'):
	new_img = deepcopy(img)

	if isinstance(new_img, nib.Nifti1Image):
		data = new_img.get_data()
	else:
		data = new_img
		
	data[data < threshold] = 0

	if binarize:
		data[data >= threshold] = 1

	if output == 'np':
		return data
	elif output == 'nib':
		return new_img


def convert_nib(in_nib, dtype='int16'):
    header = in_nib.get_header()
    data = in_nib.get_data()
    header['cal_max'] = data.max()
    header['cal_min'] = data.min()
    
    data = np.round(data).astype('int16')
    return nib.nifti1.Nifti1Image(data, in_nib.get_affine(), header)

def remove_value(infile, vals_rm, outfile):
	masker = mask.Masker(infile)
	img = imageutils.load_imgs(infile, masker)
	img = np.round(img)

	# Remove value
	for val in vals_rm:
		np.place(img, img == val, [0])

	# Save
	imageutils.save_img(img, outfile, masker)

def extract_roi(mask_img, data_img, masker):
	""" Exctract values from an image for each region in an atlas
	Args:
	    mask_img: Mask image or atlas image with multiple integer values denoting different regions
	    data_img: Image to be extract_roi
	    masker: A masker instance

	Returns: A list of tuples containing the region number mean value within each ROI,
	"""

	mask = imageutils.load_imgs(mask_img, masker)

	data = imageutils.load_imgs(data_img, masker)

	mask_values = np.unique(mask)[1:]

	mean_vals = []
	for region in mask_values:
		mean_vals.append((region, data[np.where(mask == region)[0]].mean()))

	return pd.DataFrame(mean_vals, columns = ['roi', 'value'])

def compress_values(array):
    unique = np.unique(array)
    d = dict(zip(unique, np.arange(0, unique.shape[0])))
    
    for k, v in d.iteritems(): array[array==k] = v
    return array

def mask_clusters(clustering, mask, amount = .3, method='percentage', compress=True):
    from copy import deepcopy
    
    clustering_copy = deepcopy(clustering)

    clustering = clustering.get_data()
    mask = mask.get_data()

    unique_values = np.unique(clustering)
    unique_values = unique_values[unique_values.nonzero()]

    def amnt_inmask(level, clustering, mask, method='percentage'):
        # Make cluster mask
        cluster_mask = clustering == level

        if method == 'percentage':
        	results = mask[cluster_mask].mean()
        elif method == 'sum':
        	results = mask[cluster_mask].sum()

        return results

    cluster_perc_in = np.array([amnt_inmask(level, clustering, mask, method=method) for level in unique_values])

    values_in = unique_values[cluster_perc_in >= amount]
    
    in_mask = np.in1d(clustering, values_in).reshape(clustering.shape)
    
    clustering_copy.get_data()[in_mask == False] = 0

    if compress is True:
    	_ = compress_values(clustering_copy.get_data())

    return in_mask, clustering_copy