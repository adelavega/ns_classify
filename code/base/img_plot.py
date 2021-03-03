from nilearn import plotting as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import nibabel as nib
from surfer import project_volume_data
from copy import deepcopy
import tempfile

def mask_nifti(nifti, mask):
    masked_nib = deepcopy(nifti)
    masked_nib.get_data()[mask.get_data() == 0] = 0
    return masked_nib

from img_tools import compress_values
def display_bilateral(brain, nifti, colormap=None, spatial_mask=None, level_mask=None, discrete=True, **kwargs):
    args = {'thresh' : 0.001, 'alpha' : 0.33, 'colorbar' : False, 'remove_existing' : True, 'min' : 1}
    if kwargs != {}:
        args.update(kwargs)

    if spatial_mask is None:
        args['alpha'] = .8
    
    if colormap is None:
        n_clusters = int(nifti.get_data().max())
        colormap = sns.color_palette('husl', n_clusters)

    if level_mask is not None:
        nifti = deepcopy(nifti)
        data = nifti.get_data()
        unique = np.unique(data[data.nonzero()])

        for val in unique:
            if not val in level_mask: 
                data[data == val] = float(0)

        unique = np.unique(data[data.nonzero()])
        colormap = [v for i, v in enumerate(colormap) if i+1 in unique]

        compress_values(nifti.get_data())
        
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        nib.save(nifti, f.name)
         
        l_roi_surf = project_volume_data(f.name, "lh",
                            subject_id="fsaverage", projsum='max', smooth_fwhm=0)
        r_roi_surf = project_volume_data(f.name, "rh",
                            subject_id="fsaverage",  projsum='max', smooth_fwhm=0)

        if discrete == True:
            l_cols = [colormap[int(c-1)] for c in np.unique(l_roi_surf)[1:]]
            if len(l_cols) < 2:
                l_cols = l_cols + [(0, 0, 0)] 
            r_cols = [colormap[int(c-1)] for c in np.unique(r_roi_surf)[1:]]
            if len(r_cols) < 2:
                r_cols = r_cols + [(0, 0, 0)] 
        else:
            l_cols = colormap
            r_cols = colormap

        brain.add_data(l_roi_surf, hemi='lh', colormap=colormap, **args)
        brain.add_data(r_roi_surf, hemi='rh', colormap=r_cols, **args)

    if spatial_mask is not None:
        spatial_masked_nifti = mask_nifti(nifti, spatial_mask)
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
            nib.save(spatial_masked_nifti, f.name)

            args['remove_existing'] = False
            args['alpha'] = .7

            l_roi_surf = project_volume_data(f.name, "lh",
                                subject_id="fsaverage", projsum='max', smooth_fwhm=0)
            r_roi_surf = project_volume_data(f.name, "rh",
                                subject_id="fsaverage", projsum='max', smooth_fwhm=0)

            l_cols = [colormap[int(c-1)] for c in np.unique(l_roi_surf)[1:]]
            if len(l_cols) < 2:
                l_cols = l_cols + [(0, 0, 0)] 
            r_cols = [colormap[int(c-1)] for c in np.unique(r_roi_surf)[1:]]
            if len(r_cols) < 2:
                r_cols = r_cols + [(0, 0, 0)] 

            brain.add_data(l_roi_surf, hemi='lh', colormap=l_cols, **args)
            brain.add_data(r_roi_surf, hemi='rh', colormap=r_cols, **args)

def display_coactivation(brain, niftis, colormap=None, reduce_alpha_step = 0, **kwargs):
    args = {'thresh' : 0.001, 'alpha' : 0.85, 'colorbar' : False, 'min' : 0}
    if kwargs != {}:
        args.update(kwargs)

    if colormap is None:
        colormap = sns.color_palette('Set1', len(niftis))
    
    for i, image in enumerate(niftis):      
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
            nib.save(image, f.name)
             
            l_roi_surf = project_volume_data(f.name, "lh",
                                subject_id="fsaverage", smooth_fwhm=2)
            r_roi_surf = project_volume_data(f.name, "rh",
                                subject_id="fsaverage", smooth_fwhm=2)

            args['remove_existing'] = i == 0

            color = sns.light_palette(colormap[i], n_colors=10)[5:]
            if l_roi_surf.sum() > 0:
                brain.add_data(l_roi_surf, hemi='lh', colormap=color, **args)
            if r_roi_surf.sum() > 0:
                brain.add_data(r_roi_surf, hemi='rh', colormap=color, **args)
                
            args['alpha'] -= reduce_alpha_step

def make_thresholded_slices(regions, colors, display_mode='z', overplot=True, binarize=True, **kwargs):
    """ Plots on axial slices numerous images
    regions: Nibabel images
    colors: List of colors (rgb tuples)
    overplot: Overlay images?
    binarize: Binarize images or plot full stat maps
    """             
    if binarize:
        for reg in regions:
             reg.get_data()[reg.get_data().nonzero()] = 1
                                   
    for i, reg in enumerate(regions):
        reg_color = LinearSegmentedColormap.from_list('reg1', [colors[i], colors[i]])
        if i == 0:
            plot = plt.plot_stat_map(reg, draw_cross=False,  display_mode=display_mode, cmap = reg_color, alpha=0.9, colorbar=False, **kwargs)
        else:
            if overplot:
                plot.add_overlay(reg, cmap = reg_color, alpha=.72)
            else:
                plt.plot_stat_map(reg, draw_cross=False,  display_mode=display_mode, cmap = reg_color, colorbar=False, **kwargs)
    
    return plot

import seaborn as sns

def plot_subset(nifti, layers, colors = None, **kwargs):
    if not isinstance(nifti, nib.Nifti1Image):    
        nifti = nib.load(nifti)
    data = nifti.get_data()

    if colors is None:
        colors = sns.color_palette('Set1', len(layers) + 1)

    for value in np.unique(np.round(data)):
        if np.in1d(value, layers)[0] == False:
            data[np.round(data) == value] = 0
    plt.plot_roi(nifti, cmap = ListedColormap(colors),**kwargs)