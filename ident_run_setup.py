from astropy.cosmology import LambdaCDM
from astropy.units import Quantity, UnitConversionError
import pandas as pd
import numpy as np
from typing import Union, List
from everystamp.downloaders import LegacyDownloader, VLASSDownloader, LoTSSDownloader
import os
import json
from copy import deepcopy

# Useful constants are set up here
HISTORY_ROOT = os.path.abspath("history/") 
HISTORY_FILE_PATH = os.path.join(HISTORY_ROOT, 'bcg_ident_proj_save.json')

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
        raise ValueError("Chosen missions in the history file are different than currently configured - adding new missions to an "\
            "identification project is not currently supported.")
    
    if read_hist['cosmo_repr'] != str(cosmo):
        raise ValueError("Cosmology in the history file is different than currently configured - you cannot make configuration "\
            "changes without making a new project.")

    if read_hist['side_length'] != side_length.to('kpc').value:
        raise ValueError("The side length in the history file is different than currently configured - you cannot make configuration "\
            "changes without making a new project.")
        
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


