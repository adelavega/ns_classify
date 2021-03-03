from neurosynth.base import imageutils 
from neurosynth.base import mask 
import numpy as np
import pandas as pd

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