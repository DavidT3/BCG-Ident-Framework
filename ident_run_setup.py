import matplotlib
from IPython.core.pylabtools import figsize
from astropy.cosmology import LambdaCDM
from astropy.units import Quantity, UnitConversionError
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict

from astropy.visualization import MinMaxInterval, LogStretch, SinhStretch, AsinhStretch, SqrtStretch, SquaredStretch, \
    LinearStretch, ImageNormalize
from everystamp.downloaders import LegacyDownloader, VLASSDownloader, LoTSSDownloader
import os
import json
from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.quit'] = ''
stretch_dict = {'LOG': LogStretch(), 'SINH': SinhStretch(), 'ASINH': AsinhStretch(), 'SQRT': SqrtStretch(),
                'SQRD': SquaredStretch(), 'LIN': LinearStretch()}

# Useful constants are set up here
HISTORY_ROOT = os.path.abspath("history/") 
HISTORY_FILE_PATH = os.path.join(HISTORY_ROOT, 'bcg_ident_proj_save.json')
OUTPUT_ROOT = os.path.abspath("outputs/")
OUTPUT_SAMP_FILE_PATH = os.path.join(OUTPUT_ROOT, 'bcg_output_sample.csv')
OUTPUT_CLUSTER_PATH = os.path.join(OUTPUT_ROOT, 'indiv_cluster/{n}/')

# Sets up the cosmology to be used throughout this BCG identification run
cosmo = LambdaCDM(70, 0.3, 0.7)

# The path to the input sample file
init_samp_file = "input_sample_files/sdssrm-xcs_txerr_lim_clusters.csv"
# We use the file name to create a 'project name' - helps us name things
proj_name = os.path.basename(init_samp_file).split('.')[0]


# Defines the side-length of the images we want to download/generate - THIS ISN'T HALF SIDE LENGTH
side_length = Quantity(2000, 'kpc')

# Configures the missions from which images should be downloaded or generated
include_miss = {'xmm': True,
                'desi-ls': True,
                'vlass': True,
                'lofar-lotss': False
               }
rel_miss = [mn for mn, m_use in include_miss.items() if m_use]

# Matches mission names to 'EveryStamp' downloader classes
rel_downloaders = {'xmm': None,
                   'desi-ls': LegacyDownloader,
                   'vlass': VLASSDownloader,
                   'lofar-lotss': LoTSSDownloader
                  }
rel_downloaders = {rd_name: rd for rd_name, rd in rel_downloaders.items() if include_miss[rd_name]}

# -------------------------------- USEFUL FUNCTIONS --------------------------------
def load_history() -> dict:
    """
    Simple function that loads in the history file from JSON to a Python dictionary, then returns it. It also checks
    to ensure that configuration values have not been changed in this script compared to how they are set in the history.

    :return: The history of the BCG identification project, as a dictionary.
    :rtype: dict
    """
    
    if not os.path.exists(HISTORY_FILE_PATH):
        raise FileNotFoundError("BCG identification project setup has not been run!")

    with open(HISTORY_FILE_PATH, 'r') as historo:
        read_hist = json.load(historo)

    # Bunch of tedious validity checks
    if read_hist['project_name'] != proj_name:
        raise ValueError("The current project name is different from the history file value.")

    if read_hist['chosen_missions'] != rel_miss:
        raise ValueError("Chosen missions in the history file are different than currently "
                         "configured - adding new missions to an identification project is not currently supported.")
    
    if read_hist['cosmo_repr'] != str(cosmo):
        raise ValueError("Cosmology in the history file is different than currently "
                         "configured - you cannot make configuration changes without making a new project.")

    if read_hist['side_length'] != side_length.to('kpc').value:
        raise ValueError("The side length in the history file is different than currently "
                         "configured - you cannot make configuration changes without making a new project.")
        
    return read_hist
        

def update_history(new_entry: Union[dict, List[dict]]) -> dict:

    # Minimal checks because only the code I write will use this - making all inputs iterable
    if isinstance(new_entry, dict):
        new_entry = [new_entry]

    # Loading in the history file as it is now
    pre_change_history = load_history()

    # Make a copy
    new_history = deepcopy(pre_change_history)
    
    for new_en in new_entry:
        new_history.update(new_en)

    with open(HISTORY_FILE_PATH, 'w') as write_historo:
        json.dump(new_history, write_historo)

    return new_history
# ----------------------------------------------------------------------------------

# --------------------------------- USEFUL CLASSES ---------------------------------
class InteractiveView:
    def __init__(self, im_data: dict, im_wcs: dict, primary_data_name: str, figsize = (10, 4),
                 im_scale: dict = None):

        self._all_im_data = im_data
        self._all_im_wcs = im_wcs
        self._data_names = list(im_data.keys())
        self._primary_data_name = primary_data_name

        if im_scale is not None:
            self._all_im_scale = {n: im_scale[n] if n in im_scale else None for n in self._all_im_data.keys()}
        else:
            self._all_im_scale = {n: None for n in self._data_names}

        self._norms = {}
        for data_name in self._data_names:
            if self._all_im_scale[data_name] is not None:
                cur_norm = ImageNormalize(self._all_im_data[data_name], self._all_im_scale[data_name]['interval'],
                                          stretch=self._all_im_scale[data_name]['stretch'])
            else:
                cur_norm = None
            self._norms[data_name] = cur_norm

        self._cmaps = {n: self._all_im_scale[n]['cmap'] if self._all_im_scale[n] is not None else None
                       for n in self._data_names}

        self._im_bounds = {}
        # QUICK CALCULATION OF IMAGE DATA RA-DEC LIMITS
        for n in self._data_names:
            cur_data = self._all_im_data[n]
            cur_wcs = self._all_im_wcs[n]
            y_max, x_max = cur_data.shape[:2]
            bottom_left = cur_wcs.all_pix2world(0, 0, 0)
            bottom_right = cur_wcs.all_pix2world(x_max, 0, 0)
            top_left = cur_wcs.all_pix2world(0, y_max, 0)
            top_right = cur_wcs.all_pix2world(x_max, y_max, 0)
            self._im_bounds[n] = (bottom_left, bottom_right, top_right, top_left)

        # self._regions = deepcopy(phot_prod.regions)

        # This is for storing references to artists with an ObsID key, so we know which artist belongs
        #  to which ObsID. Populated in the first part of _draw_regions. We also construct the reverse so that
        #  an artist instance can be easily used to lookup the ObsID it belongs to
        # self._obsid_artists = {o: [] for o in self._parent_phot_obj.obs_ids}
        # self._artist_obsids = {}
        # # In the same vein I setup a lookup dictionary for artist to region
        # self._artist_region = {}

        # Getting a total length (in pixels) of the two ratemaps - this was in order to try and figure out good
        #  coordinates for the axes to be added to the figure. It all got a bit hand wavey though, and I just adjusted
        #  things further down until I ended up with a nice looking figure.
        # tot_x = sum([im_d.shape[1] for im_d in self._all_im_data.values()]) + plt_buff * len(self._data_names)

        # pix_height = [im_dat.shape[0] for im_dat in self._all_im_data.values()]
        #
        # ind_max_height = np.argmax(pix_height)
        # name_max_height = self._data_names[ind_max_height]
        # val_max_height = pix_height[ind_max_height]
        #
        # y_conv = {n: val_max_height / pix_height[n_ind] for n_ind, n in enumerate(self._data_names)}
        # ax_fracs = {n: y_conv[n] / sum(list(y_conv.values())) for n in self._data_names}

        # width = height*sum(list(y_conv.values()))
        # width = height*3

        # height = width / sum(list(y_conv.values()))
        #
        # print((width, height))
        #
        # if figsize is not None:
        #     height = figsize[1]
        #     width = figsize[0]

        # Setting up the figure within which all the axes (data, buttons, etc.) are placed
        # in_fig = plt.figure(figsize=(width, height))

        # A bodge-factor I was going to allow for a spacing in between the two ratemaps
        # plt_buff = 1
        plt_buff = 0

        # self._im_axes = {}
        # bodge_x = -0.01*tot_x
        # bodge_x = 0
        # iter_x_off = 0
        # for im_d_name, im_d in self._all_im_data.items():
        #     cur_ax = in_fig.add_axes((iter_x_off / tot_x, 0, (iter_x_off + im_d.shape[1]) / tot_x, 1))
        #     self._im_axes[im_d_name] = cur_ax
        #     iter_x_off += im_d.shape[1] + plt_buff - bodge_x

        # for data_name_ind, data_name in enumerate(self._data_names):
        #     # start_x = sum(list(ax_fracs.values())[:data_name_ind])
        #     # stop_x = start_x + ax_fracs[data_name]
        #     # cur_ax = in_fig.add_axes((start_x, 0, stop_x, 1))
        #     # self._im_axes[data_name] = cur_ax
        #     test_frac = (1/len(self._all_im_data))
        #     start_x = data_name_ind * test_frac
        #     stop_x = start_x + test_frac
        #     cur_ax = in_fig.add_axes((start_x, 0, stop_x, 1))
        #     self._im_axes[data_name] = cur_ax

        in_fig, all_axes = plt.subplots(1, ncols=len(self._data_names), sharex=False, sharey=False,
                                        figsize=figsize)
        self._im_axes = {n: all_axes[n_ind] for n_ind, n in enumerate(self._data_names)}

        # Storing the figure in an attribute, as well as the image axis (i.e. the axis on which the data
        #  are displayed) in another attribute, for convenience.
        self._fig = in_fig
        self._ax_locs = {}
        for data_name, im_ax in self._im_axes.items():
            # Setting up the look of the data axis, removing ticks and tick labels because it's an image
            im_ax.tick_params(axis='both', direction='in', which='both', top=False, right=False)
            im_ax.xaxis.set_ticklabels([])
            im_ax.yaxis.set_ticklabels([])
            # im_ax.margins(0)
            self._ax_locs[data_name] = im_ax.get_position()

        self._im_axes[self._primary_data_name].annotate(r'COORD = [N/A, N/A] $^{\circ}$', [0.05, 1.02],
                                                        xycoords='axes fraction',
                                                        fontsize=11, color="black", fontweight = "bold",
                                                        annotation_clip=False)
        self._fig.tight_layout(w_pad=0.4)

        # Removes the save figure button from the toolbar
        new_tt = [t_item for t_item in self._fig.canvas.manager.toolbar.toolitems if t_item[0] != 'Download']
        self._fig.canvas.manager.toolbar.toolitems = new_tt

        # Setting up some visual stuff that is used in multiple places throughout the class
        # First the colours of buttons in an active and inactive state (the region toggles)
        self._but_act_col = "0.85"
        self._but_inact_col = "0.99"
        # Now the standard line widths used both for all regions, and for the region that is currently selected
        self._reg_line_width = 1.2
        self._sel_reg_line_width = 2.3
        # These are the increments when adjusting the regions by pressing wasd and qe. So for the size and
        #  angle of the selected region.
        self._size_step = 2
        self._rot_step = 10

        # Setting up and storing the connections to events on the matplotlib canvas. These are what
        #  allow specific methods to be triggered when things like button presses or clicking on the
        #  figure occur. They are stored in attributes, though I'm not honestly sure that's necessary
        # Not all uses of this class will make use of all of these connections, but I'm still defining them
        #  all here anyway
        self._pick_cid = self._fig.canvas.mpl_connect("pick_event", self._on_region_pick)
        self._move_cid = self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._rel_cid = self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._undo_cid = self._fig.canvas.mpl_connect("key_press_event", self._key_press)
        self._click_cid = self._fig.canvas.mpl_connect("button_press_event", self._click_event)

        # All uses of this class (both editing regions and just having a vaguely interactive view of the
        #  observation) will have these buttons that allow regions to be turned off and on, so they are
        #  defined here. All buttons are defined in separate axes.
        # These buttons act as toggles, they are all active by default and clicking one will turn off the source
        #  type its associated with. Clicking it again will turn it back on.
        # This button toggles extended (green) sources.
        # top_pos = self._ax_loc.y1-0.0771
        # ext_src_loc = plt.axes([0.045, top_pos, 0.075, 0.075])
        # self._ext_src_button = Button(ext_src_loc, "EXT", color=self._but_act_col)
        # self._ext_src_button.on_clicked(self._toggle_ext)
        #
        # # This button toggles point (red) sources.
        # pnt_src_loc = plt.axes([0.045, top_pos-(0.075 + 0.005), 0.075, 0.075])
        # self._pnt_src_button = Button(pnt_src_loc, "PNT", color=self._but_act_col)
        # self._pnt_src_button.on_clicked(self._toggle_pnt)
        #
        # # This button toggles types of region other than green or red (mostly valid for XCS XAPA sources).
        # oth_src_loc = plt.axes([0.045, top_pos-2*(0.075 + 0.005), 0.075, 0.075])
        # self._oth_src_button = Button(oth_src_loc, "OTHER", color=self._but_act_col)
        # self._oth_src_button.on_clicked(self._toggle_oth)
        #
        # # This button toggles custom source regions
        # cust_src_loc = plt.axes([0.045, top_pos-3*(0.075 + 0.005), 0.075, 0.075])
        # self._cust_src_button = Button(cust_src_loc, "CUST", color=self._but_act_col)
        # self._cust_src_button.on_clicked(self._toggle_cust)
        #
        # # These are buttons that can be present depending on the usage of the class
        # self._new_ell_button = None
        # self._new_circ_button = None
        #
        # # A dictionary describing the current type of regions that are on display
        # self._cur_act_reg_type = {"EXT": True, "PNT": True, "OTH": True, "CUST": True}

        # These set up the default colours, red for point, green for extended, and white for custom. I already
        #  know these colour codes because this is what the regions module colours translate into in matplotlib
        # Maybe I should automate this rather than hard coding
        self._colour_convert = {(1.0, 0.0, 0.0, 1.0): 'red', (0.0, 0.5019607843137255, 0.0, 1.0): 'green',
                                (1.0, 1.0, 1.0, 1.0): 'white'}
        # There can be other coloured regions though, XAPA for instance has lots of subclasses of region. This
        #  loop goes through the regions and finds their colour name / matplotlib colour code and adds it to the
        #  dictionary for reference
        # for region in [r for o, rl in self._regions.items() for r in rl]:
        #     art_reg = region.as_artist()
        #     self._colour_convert[art_reg.get_edgecolor()] = region.visual["edgecolor"]

        # This just provides a conversion between name and colour tuple, the inverse of colour_convert
        self._inv_colour_convert = {v: k for k, v in self._colour_convert.items()}

        # Unfortunately I cannot rely on regions being of an appropriate type (Ellipse/Circle) for what they
        #  are. For instance XAPA point source regions are still ellipses, just with the height and width
        #  set equal. So this dictionary is an independent reference point for the shape, with entries for the
        #  original regions made in the first part of _draw_regions, and otherwise set when a new region is added.
        # self._shape_dict = {}

        # I also wish to keep track of whether a particular region has been edited or not, for reference when
        #  outputting the final edited region list (if it is requested). I plan to do this with a similar approach
        #  to the shape_dict, have a dictionary with artists as keys, but then have a boolean as a value. Will
        #  also be initially populated in the first part of _draw_regions.
        # self._edited_dict = {}

        # This controls whether interacting with regions is allowed - turned off for the dynamic view method
        #  as that is not meant for editing regions
        self._interacting_on = False
        # The currently selected region is referenced in this attribute
        self._cur_pick = None
        # The last coordinate ON THE IMAGE that was clicked is stored here. Initial value is set to the centre
        self._last_click = (self._all_im_data[self._primary_data_name].shape[0] / 2,
                            self._all_im_data[self._primary_data_name].shape[1] / 2)
        self._last_radec = self._all_im_wcs[self._primary_data_name].all_pix2world(*self._last_click, 0)

        # This describes whether the artist stored in _cur_pick (if there is one) is right now being clicked
        #  and held - this is used for enabling clicking and dragging so the method knows when to stop.
        self._select = False
        self._history = []

        # These store the current settings for colour map, stretch, and scaling
        # self._cmap = cmap
        self._interval = MinMaxInterval()
        self._stretch = stretch_dict['LOG']
        # This is just a convenient place to store the name that XGA uses for the current stretch - it lets us
        #  access the current stretch instance from stretch_dict more easily (and accompanying buttons etc.)
        self._active_stretch_name = 'LOG'
        # This is used to store all the button instances created for controlling stretch
        self._stretch_buttons = {}

        # Here we define attribute to store the data and normalisation in. I copy the data to make sure that
        #  the original information doesn't get changed when smoothing is applied.
        # self._plot_data = self._parent_phot_obj.data.copy()
        # # It's also possible to mask and display the data, and the current mask is stored in this attribute
        # self._plot_mask = np.ones(self._plot_data.shape)
        # self._norm = self._renorm()

        # The output of the imshow command lives in here
        self._im_plot = None
        # Adds the actual image to the axis.
        self._replot_data()

        # self._im_axes[self._primary_data_name].axhline(400, color='white')
        # self._im_axes[self._primary_data_name].axvline(400, color='white')

        # This bit is where all the stretch buttons are set up, as well as the slider. All methods should
        #  be able to use re-stretching so that's why this is all in the init
        # ax_slid = plt.axes([self._ax_loc.x0, 0.885, 0.7771, 0.03], facecolor="white")
        # Hides the ticks to make it look nicer
        # ax_slid.set_xticks([])
        # ax_slid.set_yticks([])
        # Use the initial defined MinMaxInterval to get the initial range for the RangeSlider - used both
        #  as upper and lower boundaries and starting points for the sliders.
        # init_range = self._interval.get_limits(self._plot_data)
        # Define the RangeSlider instance, set the value text to invisible, and connect to the method it activates
        # self._vrange_slider = RangeSlider(ax_slid, 'DATA INTERVAL', *init_range, valinit=init_range)
        # # We move the RangeSlider label so that is sits within the bar
        # self._vrange_slider.label.set_x(0.6)
        # self._vrange_slider.label.set_y(0.45)
        # self._vrange_slider.valtext.set_visible(False)
        # self._vrange_slider.on_changed(self._change_interval)

        # Sets up an initial location for the stretch buttons to iterate over, so I can make this
        #  as automated as possible. An advantage is that I can just add new stretches to the stretch_dict,
        #  and they should be automatically added here.
        # loc = [self._ax_loc.x0 - (0.075 + 0.005), 0.92, 0.075, 0.075]
        # # Iterate through the stretches that I chose to store in the stretch_dict
        # for stretch_name, stretch in stretch_dict.items():
        #     # Increments the position of the button
        #     loc[0] += (0.075 + 0.005)
        #     # Sets up an axis for the button we're about to create
        #     stretch_loc = plt.axes(loc)
        #
        #     # Sets the colour for this button. Sort of unnecessary to do it like this because LOG should always
        #     #  be the initially active stretch, but better to generalise
        #     if stretch_name == self._active_stretch_name:
        #         col = self._but_act_col
        #     else:
        #         col = self._but_inact_col
        #     # Creates the button for the current stretch
        #     self._stretch_buttons[stretch_name] = Button(stretch_loc, stretch_name, color=col)
        #
        #     # Generates and adds the function for the current stretch button
        #     self._stretch_buttons[stretch_name].on_clicked(self._change_stretch(stretch_name))
        #
        # # This is the bit where we set up the buttons and slider for the smoothing function
        # smooth_loc = plt.axes([self._ax_loc.x1 + 0.005, top_pos, 0.095, 0.075])
        # self._smooth_button = Button(smooth_loc, "SMOOTH", color=self._but_inact_col)
        # self._smooth_button.on_clicked(self._toggle_smooth)
        #
        # ax_smooth_slid = plt.axes([self._ax_loc.x1 + 0.03, self._ax_loc.y0+0.002, 0.05, 0.685], facecolor="white")
        # # Hides the ticks to make it look nicer
        # ax_smooth_slid.set_xticks([])
        # # Define the Slider instance, add and position a label, and connect to the method it activates
        # self._smooth_slider = Slider(ax_smooth_slid, 'KERNEL RADIUS', 0.5, 5, valinit=1, valstep=0.5,
        #                              orientation='vertical')
        # # Remove the annoying line representing initial value that is automatically added
        # self._smooth_slider.hline.remove()
        # # We move the Slider label so that is sits within the bar
        # self._smooth_slider.label.set_rotation(270)
        # self._smooth_slider.label.set_x(0.5)
        # self._smooth_slider.label.set_y(0.45)
        # self._smooth_slider.on_changed(self._change_smooth)
        #
        # # We also create an attribute to store the current value of the slider in. Not really necessary as we
        # #  could always fetch the value out of the smooth slider attribute but its neater this way I think
        # self._kernel_rad = self._smooth_slider.val

        # This is a definition for a save button that is used in edit_view
        self._save_button = None

        # Adding a button to apply a mask generated from the regions, largely to help see if any emission
        #  from an object isn't being properly removed.
        # mask_loc = plt.axes([self._ax_loc.x0 + (0.075 + 0.005), self._ax_loc.y0 - 0.08, 0.075, 0.075])
        # self._mask_button = Button(mask_loc, "MASK", color=self._but_inact_col)
        # self._mask_button.on_clicked(self._toggle_mask)

        # This next part allows for the over-plotting of annuli to indicate analysis regions, this can be
        #  very useful to give context when manually editing regions. The only way I know of to do this is
        #  with artists, but unfortunately artists (and iterating through the artist property of the image axis)
        #  is the way a lot of stuff in this class works. So here I'm going to make a new class attribute
        #  that stores which artists are added to visualise analysis areas and therefore shouldn't be touched.
        self._ignore_arts = []
        # As this was largely copied from the get_view method of Image, I am just going to define this
        #  variable here for ease of testing
        ch_thickness = 0.8
        # If we want a cross-hair, then we put one on here
        # if cross_hair is not None:
        #     # For the case of a single coordinate
        #     if cross_hair.shape == (2,):
        #         # Converts from whatever input coordinate to pixels
        #         pix_coord = self._parent_phot_obj.coord_conv(cross_hair, pix).value
        #         # Drawing the horizontal and vertical lines
        #         self._im_ax.axvline(pix_coord[0], color="white", linewidth=ch_thickness)
        #         self._im_ax.axhline(pix_coord[1], color="white", linewidth=ch_thickness)
        #
        #     # For the case of two coordinate pairs
        #     elif cross_hair.shape == (2, 2):
        #         # Converts from whatever input coordinate to pixels
        #         pix_coord = self._parent_phot_obj.coord_conv(cross_hair, pix).value
        #
        #         # This draws the first crosshair
        #         self._im_ax.axvline(pix_coord[0, 0], color="white", linewidth=ch_thickness)
        #         self._im_ax.axhline(pix_coord[0, 1], color="white", linewidth=ch_thickness)
        #
        #         # And this the second
        #         self._im_ax.axvline(pix_coord[1, 0], color="white", linewidth=ch_thickness, linestyle='dashed')
        #         self._im_ax.axhline(pix_coord[1, 1], color="white", linewidth=ch_thickness, linestyle='dashed')
        #
        #         # Here I reset the pix_coord variable, so it ONLY contains the first entry. This is for the benefit
        #         #  of the annulus-drawing part of the code that comes after
        #         pix_coord = pix_coord[0, :]
        #
        #     else:
        #         # I don't want to bring someone's code grinding to a halt just because they passed crosshair wrong,
        #         #  it isn't essential, so I'll just display a warning
        #         warnings.warn("You have passed a cross_hair quantity that has more than two coordinate "
        #                       "pairs in it, or is otherwise the wrong shape.")
        #         # Just in case annuli were also passed, I set the coordinate to None so that it knows something is
        #         # wrong
        #         pix_coord = None
        #
        #     if pix_coord is not None:
        #         # Drawing annular radii on the image, if they are enabled and passed. If multiple coordinates have
        #         #  been passed then I assume that they want to centre on the first entry
        #         for ann_rad in radial_bins_pix:
        #             # Creates the artist for the current annular region
        #             artist = Circle(pix_coord, ann_rad.value, fill=False, ec='white',
        #                             linewidth=ch_thickness)
        #             # Means it can't be interacted with
        #             artist.set_picker(False)
        #             # Adds it to the list that lets the class know it needs to not treat it as a region
        #             #  found by a source detector
        #             self._ignore_arts.append(artist)
        #             # And adds the artist to axis
        #             self._im_ax.add_artist(artist)
        #
        #         # This draws the background region on as well, if present
        #         if back_bin_pix is not None:
        #             # The background annulus is guaranteed to only have two entries, inner and outer
        #             inn_artist = Circle(pix_coord, back_bin_pix[0].value, fill=False, ec='white',
        #                                 linewidth=ch_thickness, linestyle='dashed')
        #             out_artist = Circle(pix_coord, back_bin_pix[1].value, fill=False, ec='white',
        #                                 linewidth=ch_thickness, linestyle='dashed')
        #             # Make sure neither region can be interacted with
        #             inn_artist.set_picker(False)
        #             out_artist.set_picker(False)
        #             # Add to the ignore list and to the axis
        #             self._im_ax.add_artist(inn_artist)
        #             self._ignore_arts.append(inn_artist)
        #             self._im_ax.add_artist(out_artist)
        #             self._ignore_arts.append(out_artist)
        #
        # # This chunk checks to see if there were any matched regions associated with the parent
        # #  photometric object, and if so it adds them to the image and makes sure that they
        # #  cannot be interacted with
        # for obs_id, match_reg in self._parent_phot_obj.matched_regions.items():
        #     if match_reg is not None:
        #         art_reg = match_reg.as_artist()
        #         # Setting the style for these regions, to make it obvious that they are different from
        #         #  any other regions that might be displayed
        #         art_reg.set_linestyle('dotted')
        #
        #         # Makes sure that the region cannot be 'picked'
        #         art_reg.set_picker(False)
        #         # Sets the standard linewidth
        #         art_reg.set_linewidth(self._sel_reg_line_width)
        #         # And actually adds the artist to the data axis
        #         self._im_ax.add_artist(art_reg)
        #         # Also makes sure this artist is on the ignore list, as it's a constant and shouldn't be redrawn
        #         #  or be able to be modified
        #         self._ignore_arts.append(art_reg)

    def dynamic_view(self):
        """
        The simplest view method of this class, enables the turning on and off of regions.
        """
        # Draws on any regions associated with this instance
        self._draw_regions()

        # I THINK that activating this is what turns on automatic refreshing
        plt.ion()
        plt.show()

    def edit_view(self):
        """
        An extremely useful view method of this class - allows for direct interaction with and editing of
        regions, as well as the ability to add new regions. If a save path for region files was passed on
        declaration of this object, then it will be possible to save new region files in RA-Dec coordinates.
        """
        # This mode we DO want to be able to interact with regions
        self._interacting_on = True

        # Add two buttons to the figure to enable the adding of new elliptical and circular regions
        new_ell_loc = plt.axes([0.045, 0.191, 0.075, 0.075])
        self._new_ell_button = Button(new_ell_loc, "ELL")
        self._new_ell_button.on_clicked(self._new_ell_src)

        new_circ_loc = plt.axes([0.045, 0.111, 0.075, 0.075])
        self._new_circ_button = Button(new_circ_loc, "CIRC")
        self._new_circ_button.on_clicked(self._new_circ_src)

        # This sets up a button that saves an updated region list to a file path that was passed in on the
        #  declaration of this instance of the class. If no path was passed, then the button doesn't
        #  even appear.
        if self._reg_save_path is not None:
            save_loc = plt.axes([self._ax_loc.x0, self._ax_loc.y0 - 0.08, 0.075, 0.075])
            self._save_button = Button(save_loc, "SAVE", color=self._but_inact_col)
            self._save_button.on_clicked(self._save_region_files)

        # Draws on any regions associated with this instance
        self._draw_regions()

        plt.ion()
        plt.show(block=True)

    def _replot_data(self):
        """
        This method updates the currently plotted data using the relevant class attributes. Such attributes
        are updated and edited by other parts of the class. The plot mask is always applied to data, but when
        not turned on by the relevant button it will be all ones so will make no difference.
        """
        cur_norm = self._norms[self._primary_data_name]
        self._im_axes[self._primary_data_name].imshow(self._all_im_data[self._primary_data_name], origin="lower",
                                                      norm=cur_norm, cmap=self._cmaps[self._primary_data_name])
        # norm=self._norm

        for data_name, data in self._all_im_data.items():
            if data_name == self._primary_data_name:
                continue
            cur_norm = self._norms[data_name]

            # This does the actual plotting bit, saving the output in an attribute, so it can be
            #  removed when re-plotting
            # norm=self._norm
            self._im_plot = self._im_axes[data_name].imshow(data, origin="lower", norm=cur_norm,
                                                            cmap=self._cmaps[data_name])

        for data_name, cur_ax in self._im_axes.items():
            cur_wcs = self._all_im_wcs[data_name]

            for bnd_name, bounds in self._im_bounds.items():
                if data_name == bnd_name:
                    continue

                cur_pix_bnds = cur_wcs.all_world2pix(bounds, 0)

                cur_cov = Polygon(cur_pix_bnds, facecolor='None', edgecolor='red', alpha=0.7)
                cur_ax.add_patch(cur_cov)

    def _renorm(self) -> ImageNormalize:
        """
        Re-calculates the normalisation of the plot data with current interval and stretch settings. Takes into
        account the mask if applied. The plot mask is always applied to data, but when not turned on by the
        relevant button it will be all ones so will make no difference.

        :return: The normalisation object.
        :rtype: ImageNormalize
        """

        norm = ImageNormalize(data=self._plot_data*self._plot_mask, interval=self._interval,
                              stretch=self._stretch)

        return norm

    def _draw_crosshair(self):

        for im_ax in self._im_axes.values():
            for art in list(im_ax.lines):
                art.remove()

            for child in im_ax.get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    child.remove()

        prim_ax = self._im_axes[self._primary_data_name]

        # TODO FINALISE THIS TEMPORARY BIT
        ra_dec_ch = np.array(self._all_im_wcs[self._primary_data_name].all_pix2world(*self._last_click, 0))
        self._last_radec = ra_dec_ch

        pos_str = "[" + str(ra_dec_ch[0].round(4)) + ', ' + str(ra_dec_ch[1].round(4)) + "]" + r" $^{\circ}$"
        prim_ax.annotate("COORD = " + pos_str, [0.05, 1.02], xycoords='axes fraction', fontsize=11, color="black",
                         fontweight="bold", annotation_clip=False)

        prim_ax.axvline(self._last_click[0], color="white", linewidth=0.8)
        prim_ax.axhline(self._last_click[1], color="white", linewidth=0.8)

        for data_name, im_ax in self._im_axes.items():
            if data_name == self._primary_data_name:
                continue

            rel_wcs = self._all_im_wcs[data_name]
            rel_xy = np.array(rel_wcs.all_world2pix(*ra_dec_ch, 0))
            im_ax.axvline(rel_xy[0], color="white", linewidth=0.8)
            im_ax.axhline(rel_xy[1], color="white", linewidth=0.8)

    def _draw_regions(self):
        """
        This method is called by an _InteractiveView instance when regions need to be drawn on top of the
        data view axis (i.e. the image/ratemap). Either for the first time or as an update due to a button
        click, region changing, or new region being added.
        """
        # These artists are the ones that represent regions, the ones in self._ignore_arts are there
        #  just for visualisation (for instance showing an analysis/background region) and can't be
        #  turned on or off, can't be edited, and shouldn't be saved.
        # rel_artists = [arty for arty in self._im_ax.artists if arty not in self._ignore_arts]
        rel_artists = [arty for arty in self._im_ax.patches if arty not in self._ignore_arts]

        # This will trigger in initial cases where there ARE regions associated with the photometric product
        #  that has spawned this InteractiveView, but they haven't been added as artists yet. ALSO, this will
        #  always run prior to any artists being added that are just there to indicate analysis regions, see
        #  toward the end of the __init__ for what I mean.

        if len(rel_artists) == 0 and len([r for o, rl in self._regions.items() for r in rl]) != 0:
            for o in self._regions:
                for region in self._regions[o]:
                    # Uses the region module's convenience function to turn the region into a matplotlib artist
                    art_reg = region.as_artist()
                    # Makes sure that the region can be 'picked', which enables selecting regions to modify
                    art_reg.set_picker(True)
                    # Sets the standard linewidth
                    art_reg.set_linewidth(self._reg_line_width)
                    # And actually adds the artist to the data axis
                    self._im_ax.add_artist(art_reg)
                    # Adds an entry to the shape dictionary. If a region from the parent Image is elliptical but
                    #  has the same height and width then I define it as a circle.
                    if type(art_reg) == Circle or (type(art_reg) == Ellipse and art_reg.height == art_reg.width):
                        self._shape_dict[art_reg] = 'circle'
                    elif type(art_reg) == Ellipse:
                        self._shape_dict[art_reg] = 'ellipse'
                    else:
                        raise NotImplementedError("This method does not currently support regions other than "
                                                  "circles or ellipses, but please get in touch to discuss "
                                                  "this further.")
                    # Add entries in the dictionary that keeps track of whether a region has been edited or
                    #  not. All entries start out being False of course.
                    self._edited_dict[art_reg] = False
                    # Here we save the knowledge of which artists belong to which ObsID, and vice versa
                    self._obsid_artists[o].append(art_reg)
                    self._artist_obsids[art_reg] = o
                    # This allows us to lookup the original regions from their artist
                    self._artist_region[art_reg] = region

            # Need to update this in this case
            # rel_artists = [arty for arty in self._im_ax.artists if arty not in self._ignore_arts]
            rel_artists = [arty for arty in self._im_ax.patches if arty not in self._ignore_arts]

        # This chunk controls which regions will be drawn when this method is called. The _cur_act_reg_type
        #  dictionary has keys representing the four toggle buttons, and their values are True or False. This
        #  first option is triggered if all entries are True and thus draws all regions
        if all(self._cur_act_reg_type.values()):
            allowed_colours = list(self._colour_convert.keys())

        # This checks individual entries in the dictionary, and adds allowed colours to the colour checking
        #  list which the method uses to identify the regions its allowed to draw for a particular call of this
        #  method.
        else:
            allowed_colours = []
            if self._cur_act_reg_type['EXT']:
                allowed_colours.append(self._inv_colour_convert['green'])
            if self._cur_act_reg_type['PNT']:
                allowed_colours.append(self._inv_colour_convert['red'])
            if self._cur_act_reg_type['CUST']:
                allowed_colours.append(self._inv_colour_convert['white'])
            if self._cur_act_reg_type['OTH']:
                allowed_colours += [self._inv_colour_convert[c] for c in self._inv_colour_convert
                                    if c not in ['green', 'red', 'white']]

        # This iterates through all the artists currently added to the data axis, setting their linewidth
        #  to zero if their colour isn't in the approved list
        for artist in rel_artists:
            if artist.get_edgecolor() in allowed_colours:
                # If we're here then the region type of this artist is enabled by a button, and thus it should
                #  be visible. We also use set_picker to make sure that this artist is allowed to be clicked on.
                artist.set_linewidth(self._reg_line_width)
                artist.set_picker(True)

                # Slightly ugly nested if statement, but this just checks to see whether the current artist
                #  is one that the user has selected. If yes then the line width should be different.
                if self._cur_pick is not None and self._cur_pick == artist:
                    artist.set_linewidth(self._sel_reg_line_width)

            else:
                # This part is triggered if the artist colour isn't 'allowed' - the button for that region type
                #  hasn't been toggled on. And thus the width is set to 0 and the region becomes invisible
                artist.set_linewidth(0)
                # We turn off 'picker' to make sure that invisible regions can't be selected accidentally
                artist.set_picker(False)
                # We also make sure that if this artist (which is not currently being displayed) was the one
                #  selected by the user, it is de-selected, so they don't accidentally make changes to an invisible
                #  region.
                if self._cur_pick is not None and self._cur_pick == artist:
                    self._cur_pick = None

    def _change_stretch(self, stretch_name: str):
        """
        Triggered when any of the stretch change buttons are pressed - acts as a generator for the response
        functions that are actually triggered when the separate buttons are pressed. Written this way to
        allow me to just write one of these functions rather than one function for each stretch.

        :param str stretch_name: The name of the stretch associated with a specific button.
        :return: A function matching the input stretch_name that will change the stretch applied to the data.
        """
        def gen_func(event):
            """
            A generated function to change the data stretch.

            :param event: The event passed by clicking the button associated with this function
            """
            # This changes the colours of the buttons so the active button has a different colour
            self._stretch_buttons[stretch_name].color = self._but_act_col
            # And this sets the previously active stretch button colour back to inactive
            self._stretch_buttons[self._active_stretch_name].color = self._but_inact_col
            # Now I change the currently active stretch stored in this class
            self._active_stretch_name = stretch_name

            # This alters the currently selected stretch stored by this class. Fetches the appropriate stretch
            #  object by using the stretch name passed when this function was generated.
            self._stretch = stretch_dict[stretch_name]
            # Performs the renormalisation that takes into account the newly selected stretch
            self._norm = self._renorm()
            # Performs the actual re-plotting that takes into account the newly calculated normalisation
            self._replot_data()

        return gen_func

    def _change_interval(self, boundaries: Tuple):
        """
        This method is called when a change is made to the RangeSlider that controls the interval range
        of the data that is displayed.

        :param Tuple boundaries: The lower and upper boundary currently selected by the RangeSlider
            controlling the interval.
        """
        # Creates a new interval, manually defined this time, with boundaries taken from the RangeSlider
        self._interval = ManualInterval(*boundaries)
        # Recalculate the normalisation with this new interval
        self._norm = self._renorm()
        # And finally replot the data.
        self._replot_data()

    def _apply_smooth(self):
        """
        This very simple function simply sets the internal data to a smooth version, making using of the
        currently stored information on the kernel radius. The smoothing is with a 2D Gaussian kernel, but
        the kernel is symmetric.
        """
        # Sets up the kernel instance - making use of Astropy because I've used it before
        the_kernel = Gaussian2DKernel(self._kernel_rad, self._kernel_rad)
        # Using an FFT convolution for now, I think this should be okay as this is purely for visual
        #  use and so I don't care much about edge effects
        self._plot_data = convolve_fft(self._plot_data, the_kernel)

    def _toggle_smooth(self, event):
        """
        This method is triggered by toggling the smooth button, and will turn smoothing on or off.

        :param event: The button event that triggered this toggle.
        """
        # If the current colour is the active button colour then smoothing is turned on already. Don't
        #  know why I didn't think of doing it this way before
        if self._smooth_button.color == self._but_act_col:
            # Put the button colour back to inactive
            self._smooth_button.color = self._but_inact_col
            # Sets the plot data back to the original unchanged version
            self._plot_data = self._parent_phot_obj.data.copy()
        else:
            # Set the button colour to active
            self._smooth_button.color = self._but_act_col
            # This runs the symmetric 2D Gaussian smoothing, then stores the result in the data
            #  attribute of the class
            self._apply_smooth()

        # Runs re-normalisation on the data and then re-plots it, necessary for either option of the toggle.
        self._renorm()
        self._replot_data()

    def _change_smooth(self, new_rad: float):
        """
        This method is triggered by a change of the slider, and sets a new smoothing kernel radius
        from the slider value. This will trigger a change if smoothing is currently turned on.

        :param float new_rad: The new radius for the smoothing kernel.
        """
        # Sets the kernel radius attribute to the new value
        self._kernel_rad = new_rad
        # But if the smoothing button is the active colour (i.e. smoothing is on), then we update the smooth
        if self._smooth_button.color == self._but_act_col:
            # Need to reset the data even though we're still smoothing, otherwise the smoothing will be
            #  applied on top of other smoothing
            self._plot_data = self._parent_phot_obj.data.copy()
            # Same deal as the else part of _toggle_smooth
            self._apply_smooth()
            self._renorm()
            self._replot_data()

    def _toggle_mask(self, event):
        """
        A method triggered by a button press that toggles whether the currently displayed image is
        masked or not.

        :param event: The event passed by the button that triggers this toggle method.
        """
        # In this case we know that masking is already applied because the button is the active colour and
        #  we set about to return everything to non-masked
        if self._mask_button.color == self._but_act_col:
            # Set the button colour to inactive
            self._mask_button.color = self._but_inact_col
            # Reset the plot mask to just ones, meaning nothing is masked
            self._plot_mask = np.ones(self._parent_phot_obj.shape)
        else:
            # Set the button colour to active
            self._mask_button.color = self._but_act_col
            # Generate a mask from the current regions
            self._plot_mask = self._gen_cur_mask()

        # Run renorm and replot, which will both now apply the current mask, whether it's been set to all ones
        #  or one generated from the current regions
        self._renorm()
        self._replot_data()

    def _gen_cur_mask(self):
        """
        Uses the current region list to generate a mask for the parent image that can be applied to the data.

        :return: The current mask.
        :rtype: np.ndarray
        """
        masks = []
        # Because the user might have added regions, we have to generate an updated region dictionary. However,
        #  we don't want to save the updated region list in the existing _regions attribute as that
        #  might break things
        cur_regs = self._update_reg_list()
        # Iterating through the flattened region dictionary
        for r in [r for o, rl in cur_regs.items() for r in rl]:
            # If the rotation angle is zero then the conversion to mask by the regions module will be upset,
            #  so I perturb the angle by 0.1 degrees
            if isinstance(r, EllipsePixelRegion) and r.angle.value == 0:
                r.angle += Quantity(0.1, 'deg')
            masks.append(r.to_mask().to_image(self._parent_phot_obj.shape))

        interlopers = sum([m for m in masks if m is not None])
        mask = np.ones(self._parent_phot_obj.shape)
        mask[interlopers != 0] = 0

        return mask

    def _toggle_ext(self, event):
        """
        Method triggered by the extended source toggle button, either causes extended sources to be displayed
        or not, depending on the existing state.

        :param event: The matplotlib event passed through from the button press that triggers this method.
        """
        # Need to save the new state of this type of region being displayed in the dictionary thats used
        #  to keep track of such things. The invert function just switches whatever entry was already there
        #  (True or False) to the opposite (False or True).
        self._cur_act_reg_type['EXT'] = np.invert(self._cur_act_reg_type['EXT'])

        # Then the colour of the button is switched to indicate whether its toggled on or not
        if self._cur_act_reg_type['EXT']:
            self._ext_src_button.color = self._but_act_col
        else:
            self._ext_src_button.color = self._but_inact_col

        # Then the currently displayed regions are updated with this method
        self._draw_regions()

    def _toggle_pnt(self, event):
        """
        Method triggered by the point source toggle button, either causes point sources to be displayed
        or not, depending on the existing state.

        :param event: The matplotlib event passed through from the button press that triggers this method.
        """
        # See the _toggle_ext method for comments explaining
        self._cur_act_reg_type['PNT'] = np.invert(self._cur_act_reg_type['PNT'])
        if self._cur_act_reg_type['PNT']:
            self._pnt_src_button.color = self._but_act_col
        else:
            self._pnt_src_button.color = self._but_inact_col

        self._draw_regions()

    def _toggle_oth(self, event):
        """
        Method triggered by the other source toggle button, either causes other (i.e. not extended,
        point, or custom) sources to be displayed or not, depending on the existing state.

        :param event: The matplotlib event passed through from the button press that triggers this method.
        """
        # See the _toggle_ext method for comments explaining
        self._cur_act_reg_type['OTH'] = np.invert(self._cur_act_reg_type['OTH'])
        if self._cur_act_reg_type['OTH']:
            self._oth_src_button.color = self._but_act_col
        else:
            self._oth_src_button.color = self._but_inact_col

        self._draw_regions()

    def _toggle_cust(self, event):
        """
        Method triggered by the custom source toggle button, either causes custom sources to be displayed
        or not, depending on the existing state.

        :param event: The matplotlib event passed through from the button press that triggers this method.
        """
        # See the _toggle_ext method for comments explaining
        self._cur_act_reg_type['CUST'] = np.invert(self._cur_act_reg_type['CUST'])
        if self._cur_act_reg_type['CUST']:
            self._cust_src_button.color = self._but_act_col
        else:
            self._cust_src_button.color = self._but_inact_col

        self._draw_regions()

    def _new_ell_src(self, event):
        """
        Makes a new elliptical region on the data axis.

        :param event: The matplotlib event passed through from the button press that triggers this method.
        """
        # This matplotlib patch is what we add as an 'artist' to the data (i.e. image) axis and is the
        #  visual representation of our new region. This creates the matplotlib instance for an extended
        #  source, which is an Ellipse.
        new_patch = Ellipse(self._last_click, 36, 28)
        # Now the face and edge colours are set up. Face colour is completely see through as I want regions
        #  to just be denoted by their edges. The edge colour is set to white, fetching the colour definition
        #  set up in the class init.
        new_patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
        new_patch.set_edgecolor(self._inv_colour_convert['white'])
        # This enables 'picking' of the artist. When enabled picking will trigger an event when the
        #  artist is clicked on
        new_patch.set_picker(True)
        # Setting up the linewidth of the new region
        new_patch.set_linewidth(self._reg_line_width)
        # And adds the artist into the axis. As this is a new artist we don't call _draw_regions for this one.
        self._im_ax.add_artist(new_patch)
        # Updates the shape dictionary
        self._shape_dict[new_patch] = 'ellipse'
        # Adds an entry to the dictionary that keeps track of whether regions have been modified or not. In
        #  this case the region in question is brand new so the entry will always be True.
        self._edited_dict[new_patch] = True

    def _new_circ_src(self, event):
        """
        Makes a new circular region on the data axis.

        :param event: The matplotlib event passed through from the button press that triggers this method.
        """
        # This matplotlib patch is what we add as an 'artist' to the data (i.e. image) axis and is the
        #  visual representation of our new region. This creates the instance, a circle in this case.
        new_patch = Circle(self._last_click, 8)
        # Now the face and edge colours are set up. Face colour is completely see through as I want regions
        #  to just be denoted by their edges. The edge colour is set to white, fetching the colour definition
        #  set up in the class init.
        new_patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
        new_patch.set_edgecolor(self._inv_colour_convert['white'])
        # This enables 'picking' of the artist. When enabled picking will trigger an event when the
        #  artist is clicked on
        new_patch.set_picker(True)
        # Setting up the linewidth of the new region
        new_patch.set_linewidth(self._reg_line_width)
        # And adds the artist into the axis. As this is a new artist we don't call _draw_regions for this one.
        self._im_ax.add_artist(new_patch)
        # Updates the shape dictionary
        self._shape_dict[new_patch] = 'circle'
        # Adds an entry to the dictionary that keeps track of whether regions have been modified or not. In
        #  this case the region in question is brand new so the entry will always be True.
        self._edited_dict[new_patch] = True

    def _click_event(self, event):
        """
        This method is triggered by clicking somewhere on the data axis.

        :param event: The click event that triggered this method.
        """
        # Checks whether the click was 'in axis' - so whether it was actually on the image being displayed
        #  If it wasn't then we don't care about it
        if (event.inaxes == self._im_axes[self._primary_data_name] and
                self._fig.canvas.toolbar.get_state()['_current_action'] == ''):
            # This saves the position that the user clicked as the 'last click', as the user may now which
            #  to insert a new region there
            self._last_click = (event.xdata, event.ydata)

            self._draw_crosshair()

    def _on_region_pick(self, event):
        """
        This is triggered by selecting a region

        :param event: The event triggered on 'picking' an artist. Contains information about which artist
            triggered the event, location, etc.
        """
        # If interacting is turned off then we don't want this to do anything, likewise if a region that
        #  is just there for visualisation is clicked ons
        if not self._interacting_on or event.artist in self._ignore_arts:
            return

        # The _cur_pick attribute references which artist is currently selected, which we can grab from the
        #  artist picker event that triggered this method
        self._cur_pick = event.artist
        # Makes sure the instance knows a region is selected right now, set to False again when the click ends
        self._select = True
        # Stores the current position of the current pick
        # self._history.append([self._cur_pick, self._cur_pick.center])

        # Redraws the regions so that thicker lines are applied to the newly selected region
        self._draw_regions()

    def _on_release(self, event):
        """
        Method triggered when button released.

        :param event: Event triggered by releasing a button click.
        """
        # This method's one purpose is to set this to False, meaning that the currently picked artist
        #  (as referenced in self._cur_pick) isn't currently being clicked and held on
        self._select = False

    def _on_motion(self, event):
        """
        This is triggered when someone clicks and holds an artist, and then drags it around.

        :param event: Event triggered by motion of the mouse.
        """
        # Makes sure that an artist is actually clicked and held on, to make sure something should be
        #  being moved around right now
        if self._select is False:
            return

        # Set the new position of the currently picked artist to the new position of the event
        self._cur_pick.center = (event.xdata, event.ydata)

        # Changes the entry in the edited dictionary to True, as the region in question has been moved
        self._edited_dict[self._cur_pick] = True

    def _key_press(self, event):
        """
        A method triggered by the press of a key (or combination of keys) on the keyboard. For most keys
        this method does absolutely nothing, but it does enable the resizing and rotation of regions.

        :param event: The keyboard press event that triggers this method.
        """
        # if event.key == "ctrl+z":
        #     if len(self._history) != 0:
        #         self._history[-1][0].center = self._history[-1][1]
        #         self._history[-1][0].figure.canvas.draw()
        #         self._history.pop(-1)

        if event.key == "w" and self._cur_pick is not None:
            if type(self._cur_pick) == Circle:
                self._cur_pick.radius += self._size_step
            # It is possible for actual artist type to be an Ellipse but for the region to be circular when
            #  it was taken from the parent Image of this instance, and in that case we still want it to behave
            #  like a circle for resizing.
            elif self._shape_dict[self._cur_pick] == 'circle':
                self._cur_pick.height += self._size_step
                self._cur_pick.width += self._size_step
            else:
                self._cur_pick.height += self._size_step
            self._cur_pick.figure.canvas.draw()
            # The region has had its size changed, thus we make sure the class knows the region has been edited
            self._edited_dict[self._cur_pick] = True

        # For comments for the rest of these, see the event key 'w' one, they're the same but either shrinking
        #  or growing different axes
        if event.key == "s" and self._cur_pick is not None:
            if type(self._cur_pick) == Circle:
                self._cur_pick.radius -= self._size_step
            elif self._shape_dict[self._cur_pick] == 'circle':
                self._cur_pick.height -= self._size_step
                self._cur_pick.width -= self._size_step
            else:
                self._cur_pick.height -= self._size_step
            self._cur_pick.figure.canvas.draw()
            # The region has had its size changed, thus we make sure the class knows the region has been edited
            self._edited_dict[self._cur_pick] = True

        if event.key == "d" and self._cur_pick is not None:
            if type(self._cur_pick) == Circle:
                self._cur_pick.radius += self._size_step
            elif self._shape_dict[self._cur_pick] == 'circle':
                self._cur_pick.height += self._size_step
                self._cur_pick.width += self._size_step
            else:
                self._cur_pick.width += self._size_step
            self._cur_pick.figure.canvas.draw()
            # The region has had its size changed, thus we make sure the class knows the region has been edited
            self._edited_dict[self._cur_pick] = True

        if event.key == "a" and self._cur_pick is not None:
            if type(self._cur_pick) == Circle:
                self._cur_pick.radius -= self._size_step
            elif self._shape_dict[self._cur_pick] == 'circle':
                self._cur_pick.height -= self._size_step
                self._cur_pick.width -= self._size_step
            else:
                self._cur_pick.width -= self._size_step
            self._cur_pick.figure.canvas.draw()
            # The region has had its size changed, thus we make sure the class knows the region has been edited
            self._edited_dict[self._cur_pick] = True

        if event.key == "q" and self._cur_pick is not None:
            self._cur_pick.angle += self._rot_step
            self._cur_pick.figure.canvas.draw()
            # The region has had its size changed, thus we make sure the class knows the region has been edited
            self._edited_dict[self._cur_pick] = True

        if event.key == "e" and self._cur_pick is not None:
            self._cur_pick.angle -= self._rot_step
            self._cur_pick.figure.canvas.draw()
            # The region has had its size changed, thus we make sure the class knows the region has been edited
            self._edited_dict[self._cur_pick] = True

    def _update_reg_list(self) -> Dict:
        """
        This method goes through the current artists, checks whether any represent new or updated regions, and
        generates a new list of region objects from them.

        :return: The updated region dictionary.
        :rtype: Dict
        """
        # Here we use the edited dictionary to note that there have been changes to regions
        if any(self._edited_dict.values()):
            # Setting up the dictionary to store the altered regions in. We include keys for each of the ObsIDs
            #  associated with the parent product, and then another list with the key 'new'; for regions
            #  that have been added during the editing.
            new_reg_dict = {o: [] for o in self._parent_phot_obj.obs_ids}
            new_reg_dict['new'] = []

            # These artists are the ones that represent regions, the ones in self._ignore_arts are there
            #  just for visualisation (for instance showing an analysis/background region) and can't be
            #  turned on or off, can't be edited, and shouldn't be saved.
            # rel_artists = [arty for arty in self._im_ax.artists if arty not in self._ignore_arts]
            rel_artists = [arty for arty in self._im_ax.patches if arty not in self._ignore_arts]
            for artist in rel_artists:
                # Fetches the boolean variable that describes if the region was edited
                altered = self._edited_dict[artist]
                # The altered variable is True if an existing region has changed or if a new artist exists
                if altered and type(artist) == Ellipse:
                    # As instances of this class are always declared internally by an Image class, and I know
                    #  the image class always turns SkyRegions into PixelRegions, we know that its valid to
                    #  output PixelRegions here
                    cen = PixCoord(x=artist.center[0], y=artist.center[1])
                    # Creating the equivalent region object from the artist
                    new_reg = EllipsePixelRegion(cen, artist.width, artist.height, Quantity(artist.angle, 'deg'))
                    # Fetches and sets the colour of the region, converting from matplotlib colour
                    new_reg.visual['edgecolor'] = self._colour_convert[artist.get_edgecolor()]
                    new_reg.visual['facecolor'] = self._colour_convert[artist.get_edgecolor()]
                elif altered and type(artist) == Circle:
                    cen = PixCoord(x=artist.center[0], y=artist.center[1])
                    # Creating the equivalent region object from the artist
                    new_reg = CirclePixelRegion(cen, artist.radius)
                    # Fetches and sets the colour of the region, converting from matplotlib colour
                    new_reg.visual['edgecolor'] = self._colour_convert[artist.get_edgecolor()]
                    new_reg.visual['facecolor'] = self._colour_convert[artist.get_edgecolor()]
                else:
                    # Looking up the region because if we get to this point we know its an original region that
                    #  hasn't been altered
                    # Note that in this case its not actually a new reg, its just called that
                    new_reg = self._artist_region[artist]

                # Checks to see whether it's an artist that has been modified or a new one
                if artist in self._artist_obsids:
                    new_reg_dict[self._artist_obsids[artist]].append(new_reg)
                else:
                    new_reg_dict['new'].append(new_reg)

        # In this case none of the entries in the dictionary that stores whether regions have been
        #  edited (or added) is True, so the new region list is exactly the same as the old one
        else:
            new_reg_dict = self._regions

        return new_reg_dict

    def _save_region_files(self, event=None):
        """
        This just creates the updated region dictionary from any modifications, converts the separate ObsID
        entries to individual region files, and then saves them to disk. All region files are output in RA-Dec
        coordinates, making use of the parent photometric objects WCS information.

        :param event: If triggered by a button, this is the event passed.
        """
        if self._reg_save_path is not None:
            # If the event is not the default None then this function has been triggered by the save button
            if event is not None:
                # In the case of this button being successfully clicked I want it to turn green. Really I wanted
                #  it to just flash green, but that doesn't seem to be working so turning green will be fine
                self._save_button.color = 'green'

            # Runs the method that updates the list of regions with any alterations that the user has made
            final_regions = self._update_reg_list()
            for o in final_regions:
                # Read out the regions for the current ObsID
                rel_regs = final_regions[o]
                # Convert them to degrees
                rel_regs = [r.to_sky(self._parent_phot_obj.radec_wcs) for r in rel_regs]
                # Construct a save_path
                rel_save_path = self._reg_save_path.replace('.reg', '_{o}.reg'.format(o=o))
                # write_ds9(rel_regs, rel_save_path, 'image', radunit='')
                # This function is a part of the regions module, and will write out a region file.
                #  Specifically RA-Dec coordinate system in units of degrees.
                Regions(rel_regs).write(rel_save_path, format='ds9')

        else:
            raise ValueError('No save path was passed, so region files cannot be output.')

# ----------------------------------------------------------------------------------

