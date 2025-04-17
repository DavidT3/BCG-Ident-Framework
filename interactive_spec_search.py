import gzip

import matplotlib
import requests
from IPython.core.pylabtools import figsize
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity, UnitConversionError
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel
from bs4 import BeautifulSoup
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pyvo.dal import TAPService
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict
import io
from sparcl.client import SparclClient

from astropy.visualization import MinMaxInterval, LogStretch, SinhStretch, AsinhStretch, SqrtStretch, SquaredStretch, \
    LinearStretch, ImageNormalize
import os
import json
from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, RegularPolygon

from xga.imagetools.misc import pix_deg_scale

from ident_run_setup import load_history, update_history

stretch_dict = {'LOG': LogStretch(), 'SINH': SinhStretch(), 'ASINH': AsinhStretch(), 'SQRT': SqrtStretch(),
                'SQRD': SquaredStretch(), 'LIN': LinearStretch()}

ALL_SPEC_SERV = {'rcsedv2': "https://dc.voxastro.org/tap",
                 'noirlab-desidr1': "https://datalab.noirlab.edu/tap"}

RCSEDv2_FIT_TYPES = {'6df': ['emiles_emis1', 'miles_mgfe_emis1', 'pegase_expsfh_emis1', 'pegase_ssp_emis1'],
                     'sdss': ['miles', 'miles_mgfe', 'pegase', 'xshooter'],
                     'eboss': ['miles', 'miles_mgfe', 'pegase', 'xshooter'],
                     'hectospec': ['emiles_emis1', 'miles_mgfe_emis1', 'pegase_expsfh_emis1', 'pegase_ssp_emis1'],
                     'gama': ['emiles', 'miles_mgfe', 'xshooter'],
                     'fast': ['emiles_emis1', 'miles_mgfe_emis1', 'pegase_expsfh_emis1', 'pegase_ssp_emis1'],
                     '2df': ['emiles_emis1', 'miles_mgfe_emis1', 'pegase_expsfh_emis1', 'pegase_ssp_emis1']}

DEFAULT_RCSEDv2_FIT_TYPE = {'6df': 'emiles_emis1',
                            'sdss': 'miles',
                            'eboss': 'miles',
                            'hectospec': 'emiles_emis1',
                            'gama': 'emiles',
                            'fast': 'emiles_emis1',
                            '2df': 'emiles_emis1'}

RCSEDv2_BASE = 'https://data.voxastro.org/rcsed2/'
RCSEDv2_FIT_DOWN_URLS = {'6df': RCSEDv2_BASE + '6dfgs/{ft}/{f_id}/nbursts_6df_{f_id}{s_id}.fits.gz',
                         'sdss': RCSEDv2_BASE + 'sdss/{ft}/{dk}/{p}/nbursts_sdss_{p}_{m}_{f}_{ft}_{dk}.fits.gz',
                         'eboss': RCSEDv2_BASE + 'sdss/{ft}/{dk}/{p}/nbursts_sdss_{p}_{m}_{f}_{ft}_{dk}.fits.gz',
                         'hectospec': RCSEDv2_BASE + 'hectospec/{ft}/{hd}/{hds}/nbursts_hectospec_{hs}.gz',
                         'gama': RCSEDv2_BASE + 'gama/{ft}/{dk}/{gid}/nbursts_gama_{fgid}_{ft}_{dk}.fits.gz',
                         'fast': RCSEDv2_BASE + 'fast/{ft}/{y}/{fd}/nbursts_fast_{fid}.fits.gz',
                         '2df': RCSEDv2_BASE + '2dfgrs/{ft}/{f}/{c}/nbursts_2df_{s_id}_{e}.fits.gz'}

RCSEDv2_SURVEY_SPEC_IDS = {'6df': {'sixdf_specid': 'obs_spec_id'},
                           'sdss': {'sdss_mjd': 'obs_mjd', 'sdss_plate': 'obs_plate_id',
                                    'sdss_fiberid': 'obs_fiber_id'},
                           'eboss': {'sdss_mjd': 'obs_mjd', 'sdss_plate': 'obs_plate_id',
                                     'sdss_fiberid': 'obs_fiber_id'},
                           'hectospec': {'hectospec_date': 'obs_date', 'hectospec_dataset': 'obs_dataset_id',
                                         'hectospec_spec': 'obs_dataset_spec_id'},
                           'gama': {'gama_specid': 'obs_spec_id'},
                           'fast': {'fast_date': 'obs_date', 'fast_dataset': 'obs_dataset_id',
                                    'fast_spec' :'fast_dataset_spec_id'},
                           '2df': {'twodf_seqnum': 'obs_sequence_id', 'twodf_ifield': 'obs_field_id',
                                   'twodf_conf_id': 'obs_iconf', 'twodf_extnum': 'obs_extension_id'},
                           '2dflens': {'twodflens_target': 'obs_spec_id'},
                           'cfa': {'cfa_rfn': 'obs_spec_id'},
                           'uzc': {'uzc_zname': 'obs_spec_id'},
                           'lamost': {'lamost_obsid': 'obs_spec_id', 'lamost_planid': 'planned_target',
                                      'lamost_obsdate': 'obs_date', 'lamost_lmjd': 'obs_mjd',
                                      'lamost_spid': 'spectrograph_id', 'lamost_fiberid': 'obs_fiber_id'},
                           'lega_c': {'lega_c_spectid': 'obs_spec_id'},
                           'deep2': {'deep2_objno': 'obs_spec_id'},
                           'deep3': {'deep3_objno': 'obs_spec_id'},
                           'wigglez': {'wigglez_specfile': 'obs_spec_id'}
                           }

# RADII SPECIFICALLY
# TODO COULDN'T FIND APERTURE FOR CFA/UZC - FAST APERTURE DEPENDS ON SLIT, lega_c ALSO DEPENDS ON SLIT BUT I THINK
#  THEY USED 1ARCSEC WIDTH, DEEP2/3 USED CLEVER ADAPTIVE MASK THINGS I THINK (BASICALLY I DON'T KNOW HOW BIG THEY ARE)
SURVEY_AP_SIZE = {'6df': Quantity(6.7/2, 'arcsec'),
                  'sdss': Quantity(3/2, 'arcsec'),
                  'eboss': Quantity(2/2, 'arcsec'),
                  'lamost': Quantity(3.3/2, 'arcsec'),
                  'cfa': Quantity(10, 'arcsec'),
                  'uzc': Quantity(10, 'arcsec'),
                  'hectospec': Quantity(1.5/2, 'arcsec'),
                  'gama': Quantity(2/2, 'arcsec'),
                  'fast': Quantity(10, 'arcsec'),
                  '2df': Quantity(2.1/2, 'arcsec'),
                  '2dflens': Quantity(2.1/2, 'arcsec'),
                  'desi': Quantity(1.5/2, 'arcsec'),
                  'lega_c': Quantity(1/2, 'arcsec'),
                  'deep2': Quantity(3/2, 'arcsec'),
                  'deep3': Quantity(3/2, 'arcsec'),
                  'wigglez': Quantity(1.6/2, 'arcsec')}

SURVEY_COLOURS = {'6df': 'limegreen',
                  'sdss': 'mediumorchid',
                  'eboss': 'gold',
                  'lamost': 'deeppink',
                  'cfa': 'olive',
                  'hectospec': 'tab:cyan',
                  'gama': 'darkorange',
                  'fast': 'forestgreen',
                  '2df': 'palegreen',
                  'uzc': 'teal',
                  'desi': 'crimson'}


class SpecSearch:
    def __init__(self, im_data: dict, im_wcs: dict, primary_data_name: str, cluster_name: str, bcg_cands: SkyCoord,
                 figsize=(10, 4), im_scale: dict = None, spec_sources: list = None, im_spec_ratio: list = None,
                 default_smooth_spec: int = None):

        self._all_im_data = im_data
        self._all_im_wcs = im_wcs
        self._data_names = list(im_data.keys())
        self._primary_data_name = primary_data_name
        self._cluster_name = cluster_name
        self._bcg_cand_coords = bcg_cands
        # These will help sort the spectra for different BCG candidates
        self._bcg_cand_identified_spec = {}
        # The index of the BCG candidate that is currently being matched to spectral observations
        self._cur_cand_search_ind = 0
        # Place to hold the assembled and saved spec info for each BCG candidate
        self._save_spec_info = {}

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

        if im_spec_ratio is None:
            im_spec_ratio = [1, 2]

        # May change
        in_fig, all_axes = plt.subplots(1, ncols=2, sharex=False, sharey=False, figsize=figsize,
                                        width_ratios=im_spec_ratio)
        # , gridspec_kw={"width_ratios": [0.4, 0.6]}

        self._im_axes = {primary_data_name: all_axes[0]}

        self._spec_ax = all_axes[1]

        self._both_axes = {primary_data_name: all_axes[0], 'spec': self._spec_ax}

        # Storing the figure in an attribute, as well as the image axis (i.e. the axis on which the data
        #  are displayed) in another attribute, for convenience.
        self._fig = in_fig
        self._ax_locs = {}
        for data_name, im_ax in self._im_axes.items():
            # Setting up the look of the data axis, removing ticks and tick labels because it's an image
            im_ax.tick_params(axis='both', direction='in', which='both', top=False, right=False)
            im_ax.xaxis.set_ticklabels([])
            im_ax.yaxis.set_ticklabels([])
            im_ax.margins(0)
            self._ax_locs[data_name] = im_ax.get_position()

        self._im_axes[self._primary_data_name].annotate(r'COORD = [N/A, N/A] $^{\circ}$', [0.05, 1.02],
                                                        xycoords='axes fraction',
                                                        fontsize=11, color="black", fontweight="bold",
                                                        annotation_clip=False)

        # Now the spec axis
        self._spec_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        self._spec_ax.minorticks_on()
        self._spec_ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]', fontsize=14)
        self._spec_ax.set_ylabel('Flux', fontsize=14)


        self._fig.suptitle(cluster_name)
        self._fig.tight_layout(w_pad=0.4)

        # ------------------- CUSTOMISING TOOLBAR BUTTONS -------------------
        # Removes the save figure button from the toolbar
        new_tt = [t_item for t_item in self._fig.canvas.manager.toolbar.toolitems if t_item[0] != 'Download']

        # ADDING A NEW SAVE BUTTON - uses the same icon, but actually saves the cross-hair position as a
        #  BCG candidate position, puts down a white circle as a reminder, and clears the cross-hair
        def save_spec_cand():
            if len(self._cur_sel_spec) != 0:

                read_hist = load_history()
                rel_entry = read_hist['bcg_spec_identification'][self._cluster_name]
                cur_bcg_name = 'BCG' + str(self._cur_cand_search_ind+1)

                # This is a top level key, showing whether we've gone through all the BCGs yet
                rel_entry['ident_complete'] = self._cur_cand_search_ind == (len(self._bcg_cand_coords)-1)

                spec_info = {}
                spec_cnt = 0
                for spec_source, sel_specs in self._cur_sel_spec.items():
                    for sel_spec in sel_specs:
                        cur_en = {}
                        rel_full_tab = self._field_spec_search_tables[spec_source]
                        rel_row = rel_full_tab[rel_full_tab['spec_id'] == sel_spec]

                        cur_en['spec_id'] = sel_spec
                        if spec_source == 'rcsedv2':
                            cur_en['rcsedv2_spec_id'] = sel_spec
                            cur_en['survey'] = rel_row['survey'].data.tolist()[0]

                            rel_ident_cols = RCSEDv2_SURVEY_SPEC_IDS[cur_en['survey']]
                            cur_en['survey_spec_id'] = {new_col: rel_row[og_col].data.tolist()[0]
                                                        for og_col, new_col in rel_ident_cols.items()}

                            cur_en['ra'] = rel_row['ra_j2000'].data.tolist()[0]
                            cur_en['dec'] = rel_row['dec_j2000'].data.tolist()[0]
                            cur_en['approx_aperture_arcsec'] = float(SURVEY_AP_SIZE[cur_en['survey']].value)

                            cur_en['z'] = rel_row['z'].data.tolist()[0]
                            cur_en['z_err'] = rel_row['z_err'].data.tolist()[0]
                            cur_en['z_quality'] = rel_row['z_q'].data.tolist()[0]

                        elif spec_source == 'noirlab-desidr1':
                            cur_en['desidr1_target_id'] = sel_spec
                            cur_en['survey'] = "desi_dr1"

                            cur_en['ra'] = rel_row['mean_fiber_ra'].data.tolist()[0]
                            cur_en['dec'] = rel_row['mean_fiber_dec'].data.tolist()[0]

                            cur_en['approx_aperture_arcsec'] = float(SURVEY_AP_SIZE['desi'].value)

                            cur_en['z'] = rel_row['z'].data.tolist()[0]
                            cur_en['z_err'] = rel_row['zerr'].data.tolist()[0]
                            # cur_en['z_quality'] = rel_row['z_q'].data.tolist()[0]

                        spec_name = 'spec'+str(spec_cnt)
                        spec_info[spec_name] = cur_en
                        spec_cnt += 1

                self._save_spec_info[cur_bcg_name] = spec_info

                rel_entry[cur_bcg_name] = {'no_spec': False, 'identified_spectra': spec_info}
                read_hist['bcg_spec_identification'][self._cluster_name] = rel_entry
                update_history(read_hist)

                self._next_bcg_cand()

        # This is a bit of an unsafe bodge, which I got from a GitHub issue reply, but you can add the function
        #  object as an attribute after it has been declared
        self._fig.canvas.manager.toolbar.save_spec_cand = save_spec_cand
        # Add the new button to the modified set of tool items
        new_tt.append(("BCG", "Save Identified Spectra", "save", "save_spec_cand"))

        # ADDING A REFRESH BUTTON - this is in case the user regrets their choice of spec(s), it will clear previously
        #  selected specs and remove that information from the history
        def reset_bcg_spec():
            if len(self._save_spec_info) != 0:

                # Have to remove the history entry
                read_hist = load_history()
                rel_entry = {'ident_complete': False}
                read_hist['bcg_spec_identification'][self._cluster_name] = rel_entry
                update_history(read_hist)

                self._cur_cand_search_ind = 0
                self._draw_cur_bcg_cand()
                self._fig.suptitle(cluster_name, color='black')
                self._save_spec_info = {}

        # Use the bodge again, adding the reset function
        self._fig.canvas.manager.toolbar.reset_bcg_spec = reset_bcg_spec
        new_tt.append(("Reset Spectra", "Reset Selected Spectra", "refresh", "reset_bcg_spec"))

        # ADDING A NO SPEC IDENTIFIED BUTTON
        def no_bcg_spec():
            read_hist = load_history()

            rel_entry = read_hist['bcg_spec_identification'][self._cluster_name]
            cur_bcg_name = 'BCG' + str(self._cur_cand_search_ind + 1)

            # This is a top level key, showing whether we've gone through all the BCGs yet
            rel_entry['ident_complete'] = self._cur_cand_search_ind == (len(self._bcg_cand_coords) - 1)

            self._save_spec_info[cur_bcg_name] = {}

            rel_entry[cur_bcg_name] = {'no_spec': True}
            read_hist['bcg_spec_identification'][self._cluster_name] = rel_entry
            update_history(read_hist)
            self._next_bcg_cand()

        # Use the bodge again, adding the no BCG function
        self._fig.canvas.manager.toolbar.no_bcg_spec = no_bcg_spec
        new_tt.append(("No Spectra", "No BCG Spectra", "exclamation-circle", "no_bcg_spec"))

        # Finally, we add the new set of toolitems back into the toolbar instance
        self._fig.canvas.manager.toolbar.toolitems = new_tt
        # -------------------------------------------------------------------

        # Setting up some visual stuff that is used in multiple places throughout the class
        # First the colours of buttons in an active and inactive state (the region toggles)
        self._but_act_col = "0.85"
        self._but_inact_col = "0.99"
        # Now the standard line widths used both for all regions, and for the region that is currently selected
        self._reg_line_width = 1.2
        self._sel_reg_line_width = 5
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
        # self._move_cid = self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._rel_cid = self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        # self._undo_cid = self._fig.canvas.mpl_connect("key_press_event", self._key_press)
        self._click_cid = self._fig.canvas.mpl_connect("button_press_event", self._click_event)

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
        self._cur_pick = []
        # The last coordinate ON THE IMAGE that was clicked is stored here. Initial value is set to the centre
        self._last_click = (None, None)
        self._last_radec = (None, None)

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

        # The output of the imshow command lives in here
        self._im_plot = None
        # Adds the actual image to the axis.
        self._replot_data()

        # Not all artists we add are meant to be interacted with - the BCG candidate ones for instance. Storing them
        #  in this means they won't be able to be clicked on
        self._ignore_arts = []
        # Draw the BCG candidates
        self._draw_cur_bcg_cand()

        # ------------------- SETTING UP SPEC ACCESS -------------------

        # Setting up the TAP services required
        if spec_sources is None:
            spec_sources = ALL_SPEC_SERV
        else:
            spec_sources = {ss: ALL_SPEC_SERV[ss] for ss in spec_sources}

        self._tap_services = {ss: TAPService(ss_url) for ss, ss_url in spec_sources.items()}

        # Setting up NOIRLab Sparcl Client
        self._noirlab_client = SparclClient()

        # -------------------------------------------------------------------

        # --------------------- SETTING UP SPEC STORAGE ---------------------
        self._field_spec_search_tables = {ss: {} for ss in spec_sources}
        self._field_spec_ra_dec = {ss: {} for ss in spec_sources}

        self._all_spec_data = {ss: {} for ss in spec_sources}
        self._spec_plot_data = {ss: {} for ss in spec_sources}

        self._spec_arts = {}

        # -------------------------------------------------------------------

        # --------------------- SETTING UP SPEC PLOTTING --------------------
        self._smth_std_dev = default_smooth_spec
        # -------------------------------------------------------------------

        # --------------------- SETTING UP SPEC CHOOSING --------------------
        self._event_hist = []
        self._cur_sel_spec = {}
        # -------------------------------------------------------------------

        self._search_rcsed()
        self._search_desidr1()
        self._update_spec_locs()

    def _next_bcg_cand(self):
        if (len(self._bcg_cand_coords) - 1) == self._cur_cand_search_ind:
            self._fig.suptitle(self._fig.get_suptitle() + " - SPEC IDENTIFICATION COMPLETE", fontsize=16,
                               weight="bold", color='darkgoldenrod')
        else:
            self._cur_cand_search_ind += 1
            self._draw_cur_bcg_cand()

    def _draw_cur_bcg_cand(self):
        ax = self._im_axes[self._primary_data_name]

        # The only thing in the artists to be ignored is the previous selected BCG artist
        if len(self._ignore_arts) != 0:
            self._ignore_arts[0].remove()
            self._ignore_arts = []

        cur_wcs = self._all_im_wcs[self._primary_data_name]

        bcg_cand_pos = self._bcg_cand_coords[self._cur_cand_search_ind]

        bcg_pos_pix = cur_wcs.all_world2pix(*bcg_cand_pos, 0)

        bcg_artist = RegularPolygon(bcg_pos_pix, numVertices=5, radius=20, facecolor='None', edgecolor='white',
                                    linewidth=3)
        ax.add_artist(bcg_artist)
        self._ignore_arts.append(bcg_artist)

    def _search_rcsed(self):
        bottom_left = self._im_bounds[self._primary_data_name][0]
        top_right = self._im_bounds[self._primary_data_name][2]

        min_ra = min([bottom_left[0], top_right[0]])
        max_ra = max([bottom_left[0], top_right[0]])

        min_dec = min([bottom_left[1], top_right[1]])
        max_dec = max([bottom_left[1], top_right[1]])

        query = """
            SELECT
            *
            FROM rcsed_v2.spectrum
            WHERE 
            ra_j2000 BETWEEN {min_ra} AND {max_ra}
            AND
            dec_j2000 BETWEEN {min_de} AND {max_de}
            """.format(min_ra=min_ra, max_ra=max_ra, min_de=min_dec, max_de=max_dec)

        search_res = self._tap_services['rcsedv2'].search(query, maxrec=2000000)

        cur_spec_pos = {}
        for ind in range(len(search_res)):
            cur_pos = Quantity([search_res.getrecord(ind)['ra_j2000'], search_res.getrecord(ind)['dec_j2000']], 'deg')
            cur_spec_pos[int(search_res.getrecord(ind)['r2id_spec'])] = cur_pos

        self._field_spec_ra_dec['rcsedv2'] = cur_spec_pos
        search_res = search_res.to_table()
        search_res.rename_column('r2id_spec', 'spec_id')
        self._field_spec_search_tables['rcsedv2'] = search_res

    def _search_desidr1(self):

        bottom_left = self._im_bounds[self._primary_data_name][0]
        top_right = self._im_bounds[self._primary_data_name][2]

        min_ra = min([bottom_left[0], top_right[0]])
        max_ra = max([bottom_left[0], top_right[0]])

        min_dec = min([bottom_left[1], top_right[1]])
        max_dec = max([bottom_left[1], top_right[1]])

        query = """
            SELECT
            *
            FROM desi_dr1.zpix
            WHERE 
            mean_fiber_ra BETWEEN {min_ra} AND {max_ra}
            AND
            mean_fiber_dec BETWEEN {min_de} AND {max_de}
            """.format(min_ra=min_ra, max_ra=max_ra, min_de=min_dec, max_de=max_dec)

        search_res = self._tap_services['noirlab-desidr1'].search(query, maxrec=2000000)

        cur_spec_pos = {}
        for ind in range(len(search_res)):
            cur_pos = Quantity([search_res.getrecord(ind)['mean_fiber_ra'],
                                search_res.getrecord(ind)['mean_fiber_dec']], 'deg')
            cur_spec_pos[int(search_res.getrecord(ind)['targetid'])] = cur_pos

        self._field_spec_ra_dec['noirlab-desidr1'] = cur_spec_pos
        search_res = search_res.to_table()
        search_res.rename_column('targetid', 'spec_id')
        search_res['survey'] = 'desi'
        self._field_spec_search_tables['noirlab-desidr1'] = search_res

    def _fetch_rcsed_spec_models(self, rcsed_spec_id: Union[List, str] = None,
                                 rcsed_fit_type: Union[List, str] = None):

        if rcsed_spec_id is None:
            rcsed_spec_id = self._field_spec_search_tables['rcsedv2']['spec_id'].data.astype(int).tolist()
        elif isinstance(rcsed_spec_id, (str, int)):
            rcsed_spec_id = [int(rcsed_spec_id)]

        pass_ft_none = rcsed_fit_type is None
        if isinstance(rcsed_fit_type, str):
            rcsed_fit_type = [rcsed_fit_type]

        for r_spec_id in rcsed_spec_id:
            self._all_spec_data['rcsedv2'].setdefault(r_spec_id, {})
            self._spec_plot_data['rcsedv2'].setdefault(r_spec_id, {})

            row_ind = np.where(self._field_spec_search_tables['rcsedv2']['spec_id'].data == r_spec_id)
            rel_row = self._field_spec_search_tables['rcsedv2'][row_ind[0][0]]
            if rel_row['survey'] in RCSEDv2_FIT_DOWN_URLS:

                if pass_ft_none:
                    # rcsed_fit_type = RCSEDv2_FIT_TYPES[rel_row['survey']]
                    rcsed_fit_type = [DEFAULT_RCSEDv2_FIT_TYPE[rel_row['survey']]]

                if rel_row['survey'] == '6df':
                    id_6df = str(rel_row['sixdf_specid']).zfill(6)
                    fh_id = id_6df[:3]
                    sh_id = id_6df[3:]
                    for cur_ft in rcsed_fit_type:
                        if cur_ft in self._all_spec_data['rcsedv2'][r_spec_id]:
                            continue

                        rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft=cur_ft, f_id=fh_id, s_id=sh_id)

                        try:
                            with requests.get(rel_url, stream=True) as responso:
                                # This opens the data as using the astropy fits interface (using io.BytesIO() to
                                #  stream it into memory first so that fits.open can access it as an already
                                #  opened file handler).
                                with fits.open(io.BytesIO(gzip.decompress(responso.content))) as cur_sp:
                                    cur_tab = Table(cur_sp['SPECTRUM'].data)
                                    if len(self._spec_plot_data['rcsedv2'][r_spec_id]) == 0:
                                        sp_plot_data = {'wavelength': cur_tab['WAVE'].data.flatten(),
                                                        'flux': cur_tab['FLUX'].data.flatten(),
                                                        'flux_err': cur_tab['ERROR'].data.flatten(),
                                                        'survey': rel_row['survey']}

                                        self._spec_plot_data['rcsedv2'][r_spec_id] = sp_plot_data

                                    del cur_tab['WAVE']
                                    del cur_tab['FLUX']
                                    del cur_tab['ERROR']
                                    self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
                        except gzip.BadGzipFile:
                            pass

                elif rel_row['survey'] == 'sdss' or rel_row['survey'] == 'eboss':
                    cur_mjd = str(rel_row['sdss_mjd'])
                    cur_plate = str(rel_row['sdss_plate']).zfill(5)
                    cur_fiber_id = str(rel_row['sdss_fiberid']).zfill(5)

                    for cur_ft in rcsed_fit_type:
                        if cur_ft in self._all_spec_data['rcsedv2'][r_spec_id]:
                            continue
                        # 'sdss/{ft}/{dk}/{p}/nbursts_sdss_{p}_{m}_{f}_miles_{dk}.fits.gz'

                        # This opens a session that will persist - then a lot of the next session is for checking that the expected
                        #  directories are present.
                        session = requests.Session()

                        ft_url = RCSEDv2_BASE + "sdss/{ft}/".format(ft=cur_ft)
                        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
                        #  that the directories that I need to download unprocessed Suzaku data are present
                        dont_know = [en['href'][:-1] for en in
                                     BeautifulSoup(session.get(ft_url).text, "html.parser").find_all("a")
                                     if en['href'][0] == 'n']

                        found_spec = False
                        for dk in dont_know:
                            rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft=cur_ft, p=cur_plate,
                                                                                      m=cur_mjd, f=cur_fiber_id,
                                                                                      dk=dk)
                            if session.head(rel_url).ok:
                                found_spec = True
                                break

                        if not found_spec:
                            rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft='nofit', p=cur_plate,
                                                                                      m=cur_mjd, f=cur_fiber_id,
                                                                                      dk='')

                            rel_url = rel_url.replace('_.', '.')
                            if not session.head(rel_url).ok:
                                break
                            cur_ft = 'nofit'

                        try:
                            with requests.get(rel_url, stream=True) as responso:
                                # This opens the data as using the astropy fits interface (using io.BytesIO() to
                                #  stream it into memory first so that fits.open can access it as an already
                                #  opened file handler).
                                with fits.open(io.BytesIO(gzip.decompress(responso.content))) as cur_sp:
                                    cur_tab = Table(cur_sp['SPECTRUM'].data)
                                    if len(self._spec_plot_data['rcsedv2'][r_spec_id]) == 0:
                                        sp_plot_data = {'wavelength': cur_tab['WAVE'].data.flatten(),
                                                        'flux': cur_tab['FLUX'].data.flatten(),
                                                        'flux_err': cur_tab['ERROR'].data.flatten(),
                                                        'survey': rel_row['survey']}

                                        self._spec_plot_data['rcsedv2'][r_spec_id] = sp_plot_data

                                    del cur_tab['WAVE']
                                    del cur_tab['FLUX']
                                    del cur_tab['ERROR']
                                    self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
                        except gzip.BadGzipFile:
                            pass

                elif rel_row['survey'] == 'hectospec':
                    hectospec_date = rel_row['hectospec_date']
                    hectospec_dataset = rel_row['hectospec_dataset']
                    hectospec_spec = rel_row['hectospec_spec']

                    for cur_ft in rcsed_fit_type:
                        if cur_ft in self._all_spec_data['rcsedv2'][r_spec_id]:
                            continue

                        rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft=cur_ft, hd=hectospec_date,
                                                                                  hds=hectospec_dataset,
                                                                                  hs=hectospec_spec)
                        try:
                            with requests.get(rel_url, stream=True) as responso:
                                # This opens the data as using the astropy fits interface (using io.BytesIO() to
                                #  stream it into memory first so that fits.open can access it as an already
                                #  opened file handler).
                                with fits.open(io.BytesIO(gzip.decompress(responso.content))) as cur_sp:
                                    cur_tab = Table(cur_sp['SPECTRUM'].data)
                                    if len(self._spec_plot_data['rcsedv2'][r_spec_id]) == 0:
                                        sp_plot_data = {'wavelength': cur_tab['WAVE'].data.flatten(),
                                                        'flux': cur_tab['FLUX'].data.flatten(),
                                                        'flux_err': cur_tab['ERROR'].data.flatten(),
                                                        'survey': rel_row['survey']}

                                        self._spec_plot_data['rcsedv2'][r_spec_id] = sp_plot_data

                                    del cur_tab['WAVE']
                                    del cur_tab['FLUX']
                                    del cur_tab['ERROR']
                                    self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
                        except gzip.BadGzipFile:
                            pass

                elif rel_row['survey'] == 'gama':
                    full_gama_id = rel_row['gama_specid']
                    trunc_gama_id = "_".join(full_gama_id.split('_')[:-1])

                    for cur_ft in rcsed_fit_type:
                        if cur_ft in self._all_spec_data['rcsedv2'][r_spec_id]:
                            continue

                        # This opens a session that will persist - then a lot of the next session is for checking
                        #  that the expected directories are present.
                        session = requests.Session()

                        ft_url = RCSEDv2_BASE + "gama/{ft}/".format(ft=cur_ft)
                        # This uses the beautiful soup module to parse the HTML of the top level archive
                        # directory
                        dont_know = [en['href'][:-1] for en in
                                     BeautifulSoup(session.get(ft_url).text, "html.parser").find_all("a")
                                     if en['href'][0] == 'n']

                        found_spec = False
                        for dk in dont_know:
                            rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft=cur_ft, gid=trunc_gama_id,
                                                                                      fgid=full_gama_id, dk=dk)
                            if session.head(rel_url).ok:
                                found_spec = True
                                break

                        if not found_spec:
                            rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft='nofit', gid=trunc_gama_id,
                                                                                      fgid=full_gama_id,
                                                                                      dk='')
                            rel_url = rel_url.replace('_.', '.')
                            if not session.head(rel_url).ok or 'nofit' in self._all_spec_data['rcsedv2'][r_spec_id]:
                                break
                            cur_ft = 'nofit'

                        try:
                            with requests.get(rel_url, stream=True) as responso:
                                # This opens the data as using the astropy fits interface (using io.BytesIO() to
                                #  stream it into memory first so that fits.open can access it as an already
                                #  opened file handler).
                                with fits.open(io.BytesIO(gzip.decompress(responso.content))) as cur_sp:
                                    cur_tab = Table(cur_sp['SPECTRUM'].data)
                                    if len(self._spec_plot_data['rcsedv2'][r_spec_id]) == 0:
                                        sp_plot_data = {'wavelength': cur_tab['WAVE'].data.flatten(),
                                                        'flux': cur_tab['FLUX'].data.flatten(),
                                                        'flux_err': cur_tab['ERROR'].data.flatten(),
                                                        'survey': rel_row['survey']}

                                        self._spec_plot_data['rcsedv2'][r_spec_id] = sp_plot_data

                                    del cur_tab['WAVE']
                                    del cur_tab['FLUX']
                                    del cur_tab['ERROR']
                                    self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
                        except gzip.BadGzipFile:
                            pass

                elif rel_row['survey'] == 'fast':
                    fast_year = rel_row['fast_date']
                    fast_dataset = rel_row['fast_dataset']
                    fast_spec = rel_row['fast_spec'].split('.ms')[0]

                    for cur_ft in rcsed_fit_type:
                        if cur_ft in self._all_spec_data['rcsedv2'][r_spec_id]:
                            continue

                        rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft=cur_ft, y=fast_year,
                                                                                  fd=fast_dataset, fid=fast_spec)

                        try:
                            with requests.get(rel_url, stream=True) as responso:
                                # This opens the data as using the astropy fits interface (using io.BytesIO() to
                                #  stream it into memory first so that fits.open can access it as an already
                                #  opened file handler).
                                with fits.open(io.BytesIO(gzip.decompress(responso.content))) as cur_sp:
                                    cur_tab = Table(cur_sp['SPECTRUM'].data)
                                    if len(self._spec_plot_data['rcsedv2'][r_spec_id]) == 0:
                                        sp_plot_data = {'wavelength': cur_tab['WAVE'].data.flatten(),
                                                        'flux': cur_tab['FLUX'].data.flatten(),
                                                        'flux_err': cur_tab['ERROR'].data.flatten(),
                                                        'survey': rel_row['survey']}

                                        self._spec_plot_data['rcsedv2'][r_spec_id] = sp_plot_data

                                    del cur_tab['WAVE']
                                    del cur_tab['FLUX']
                                    del cur_tab['ERROR']
                                    self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
                        except gzip.BadGzipFile:
                            pass

                elif rel_row['survey'] == '2df':
                    twodf_seq = rel_row['twodf_seqnum']
                    twodf_field = rel_row['twodf_ifield']
                    twodf_conf = rel_row['twodf_iconf']
                    twodf_ext = rel_row['twodf_extnum']

                    for cur_ft in rcsed_fit_type:
                        if cur_ft in self._all_spec_data['rcsedv2'][r_spec_id]:
                            continue

                        rel_url = RCSEDv2_FIT_DOWN_URLS[rel_row['survey']].format(ft=cur_ft, f=twodf_field,
                                                                                  c=twodf_conf, s_id=twodf_seq,
                                                                                  e=twodf_ext)
                        try:
                            with requests.get(rel_url, stream=True) as responso:
                                # This opens the data as using the astropy fits interface (using io.BytesIO() to
                                #  stream it into memory first so that fits.open can access it as an already
                                #  opened file handler).
                                with fits.open(io.BytesIO(gzip.decompress(responso.content))) as cur_sp:

                                    cur_tab = Table(cur_sp[1].data)

                                    if len(self._spec_plot_data['rcsedv2'][r_spec_id]) == 0:
                                        sp_plot_data = {'wavelength': cur_tab['WAVE'].data.flatten(),
                                                        'flux': cur_tab['FLUX'].data.flatten(),
                                                        'flux_err': cur_tab['ERROR'].data.flatten(),
                                                        'survey': rel_row['survey']}

                                        self._spec_plot_data['rcsedv2'][r_spec_id] = sp_plot_data

                                    del cur_tab['WAVE']
                                    del cur_tab['FLUX']
                                    del cur_tab['ERROR']
                                    self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
                        except gzip.BadGzipFile:
                            pass

    def _fetch_desi(self, desi_spec_id):
        if desi_spec_id is None:
            desi_spec_id = self._field_spec_search_tables['noirlab-desi']['spec_id'].data.astype(int).tolist()
        elif isinstance(desi_spec_id, (str, int)):
            desi_spec_id = [int(desi_spec_id)]

        to_fetch = ['specid', 'redshift', 'flux', 'wavelength', 'spectype', 'specprimary', 'survey',
                    'program', 'targetid', 'redshift_warning']

        fetched_spec = self._noirlab_client.retrieve_by_specid(desi_spec_id, include=to_fetch,
                                                               dataset_list=['DESI-DR1'], limit=10000)
        
        for spec_data in fetched_spec.data[1:]:
            
            cur_d_spec_id = spec_data['specid']

            sp_plot_data = {'wavelength': spec_data['wavelength'].flatten(),
                            'flux': spec_data['flux'].flatten(),
                            # 'flux_err': spec_data['flux_err'].flatten(),
                            'survey': 'noirlab-desidr1'}

            self._spec_plot_data['noirlab-desidr1'][cur_d_spec_id] = sp_plot_data

            # # del cur_tab['WAVE']
            # # del cur_tab['FLUX']
            # # del cur_tab['ERROR']
            # # self._all_spec_data['rcsedv2'][r_spec_id][cur_ft] = cur_tab
            #
            # self._all_spec_data['noirlab-desi'][cur_d_spec_id]['nofit'] = spec_data

    def _update_spec_locs(self):
        ax = self._im_axes[self._primary_data_name]
        cur_wcs = self._all_im_wcs[self._primary_data_name]

        repr_art = []
        repr_art_surv = []
        for spec_source, spec_poses in self._field_spec_ra_dec.items():
            for sp_id, sp_pos in spec_poses.items():

                sp_pos_pix = cur_wcs.all_world2pix(*sp_pos, 0)

                rel_search = self._field_spec_search_tables[spec_source]

                rel_row = rel_search[rel_search['spec_id'] == sp_id]
                cur_surv = rel_row['survey'].data[0]

                # pix_rad = (SURVEY_AP_SIZE[cur_surv]/pix_deg_scale(sp_pos, cur_wcs)).to('pix').value
                pix_rad = 10


                # Possible I've not added some survey colours, so making sure it doesn't just fall over
                if cur_surv in SURVEY_COLOURS:
                    cur_col = SURVEY_COLOURS[cur_surv]
                else:
                    cur_col = 'white'

                sp_art = Circle(sp_pos_pix, pix_rad, facecolor='None', edgecolor=cur_col,
                                linewidth=self._reg_line_width)
                sp_art.set_picker(True)
                ax.add_artist(sp_art)

                if cur_surv not in repr_art_surv:
                    # sp_art.set_label(cur_surv)

                    leg_art = Line2D([], [], color=cur_col, marker='o', markerfacecolor="None", linewidth=0)
                    repr_art.append(leg_art)
                    repr_art_surv.append(cur_surv)

                self._spec_arts[sp_art] = {'spec_source': spec_source, 'spec_id': sp_id}

        leg_cols = 4
        # leg_rows = np.ceil(len(repr_art_per_surv)/leg_cols).astype(int)

        ax.legend(repr_art, repr_art_surv, loc="upper left", bbox_to_anchor=(0.01, 0), ncol=leg_cols, borderaxespad=0)

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

        norm = ImageNormalize(data=self._plot_data * self._plot_mask, interval=self._interval,
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

            prim_ax = self._im_axes[self._primary_data_name]

            for art in list(prim_ax.lines):
                art.remove()

            for child in prim_ax.get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    child.remove()

            ra_dec_ch = np.array(self._all_im_wcs[self._primary_data_name].all_pix2world(*self._last_click, 0))
            self._last_radec = ra_dec_ch

            pos_str = "[" + str(ra_dec_ch[0].round(4)) + ', ' + str(ra_dec_ch[1].round(4)) + "]" + r" $^{\circ}$"
            prim_ax.annotate("COORD = " + pos_str, [0.05, 1.02], xycoords='axes fraction', fontsize=11, color="black",
                             fontweight="bold", annotation_clip=False)

            # self._draw_crosshair()

    def _on_region_pick(self, event):
        """
        This is triggered by selecting a region

        :param event: The event triggered on 'picking' an artist. Contains information about which artist
            triggered the event, location, etc.
        """
        # If interacting is turned off then we don't want this to do anything, likewise if a region that
        #  is just there for visualisation is clicked ons

        if event.artist in self._ignore_arts:
            return

        if len(self._event_hist) != 0 and ((event.mouseevent.x != self._event_hist[-1].mouseevent.x) or
                                           (event.mouseevent.y != self._event_hist[-1].mouseevent.y)):
            self._event_hist = []

            self._spec_ax.clear()
            self._spec_ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
            self._spec_ax.minorticks_on()
            self._spec_ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]', fontsize=14)
            self._spec_ax.set_ylabel('Flux', fontsize=14)

            self._cur_sel_spec = {}

            for cur_art in self._cur_pick:
                cur_art.set_lw(self._reg_line_width)
            self._cur_pick = []

        # The _cur_pick attribute references which artist is currently selected, which we can grab from the
        #  artist picker event that triggered this method
        self._cur_pick.append(event.artist)
        # Makes sure the instance knows a region is selected right now, set to False again when the click ends
        self._select = True

        event.artist.set_lw(self._sel_reg_line_width)

        sp_src_id = self._spec_arts[event.artist]
        rel_id = sp_src_id['spec_id']
        if sp_src_id['spec_source'] == 'rcsedv2':
            rel_row = self._field_spec_search_tables['rcsedv2'][
                self._field_spec_search_tables['rcsedv2']['spec_id'] == rel_id]
            rel_z = rel_row['z'].data[0].round(6)
            # self._spec_ax.set_title("{ss} - {s_id}".format(ss=sp_src_id['spec_source'], s_id=rel_id))
            self._fetch_rcsed_spec_models(rel_id)
            if len(self._spec_plot_data['rcsedv2'][rel_id]) != 0:
                self._draw_spectrum(sp_src_id['spec_source'], rel_id, z=rel_z)

        elif sp_src_id['spec_source'] == 'noirlab-desidr1':
            rel_row = self._field_spec_search_tables['noirlab-desidr1'][
                self._field_spec_search_tables['noirlab-desidr1']['spec_id'] == rel_id]
            rel_z = rel_row['z'].data[0].round(6)

            self._fetch_desi(rel_id)
            self._draw_spectrum(sp_src_id['spec_source'], rel_id, z=rel_z)
        else:
            pass

        self._cur_sel_spec.setdefault(sp_src_id['spec_source'], [])
        self._cur_sel_spec[sp_src_id['spec_source']].append(rel_id)
        self._event_hist.append(event)

    def _draw_spectrum(self, spec_source, spec_id, smooth_stddev: int = None, z: float = None):

        if smooth_stddev is None:
            smooth_stddev = self._smth_std_dev

        sp_ax = self._spec_ax

        rel_data = self._spec_plot_data[spec_source][spec_id]

        lab = "{s} {s_id}".format(s=rel_data['survey'], s_id=spec_id)
        if z is not None:
            lab += " z={z}".format(z=z)

        og_dat_line = sp_ax.plot(rel_data['wavelength'], rel_data['flux'], alpha=0.5, label=lab)
        chos_col = og_dat_line[0].get_color()

        if smooth_stddev is not None:
            smoothed = convolve(rel_data['flux'], Gaussian1DKernel(smooth_stddev))

            sp_ax.plot(rel_data['wavelength'], smoothed, color=chos_col, alpha=1)

            dat_lims = (np.nanmin(smoothed)*0.7, np.nanmax(smoothed)*1.3)

        else:
            dat_lims = (np.nanmin(rel_data)*0.7, np.nanmax(rel_data)*1.3)

        cur_lims = sp_ax.get_ylim()
        fin_lims = (min(dat_lims[0], cur_lims[0]), max(dat_lims[1], cur_lims[1]))

        sp_ax.set_ylim(fin_lims)

        if len(self._cur_pick) < 2:
            sp_ax.set_title('{ss} {s} - {s_id}'.format(ss=spec_source, s=rel_data['survey'], s_id=spec_id))

        sp_ax.legend()


        # self._fig.canvas.draw()
        # sp_ax.redraw_in_frame()

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
