#%%
import unittest

import xarray as xr; import pandas as pd; 
import numpy as np; 
import os
import sys
import matplotlib.pyplot as plt

'''
Module to test features of pwp code.
'''
#%%
# set paths
src_folder = "src\\"
data_folder = "PTCA_data\\NAM_argo_profiles\\NAM\\"

# get directories
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

# import modules
src_directory = os.path.abspath(os.path.join(parent_directory, src_folder))
if src_directory not in sys.path:
    sys.path.append(os.path.join(parent_directory, src_directory))
import pwp
import met

#%%
# import data
# data_directory = os.path.abspath(os.path.join(parent_directory, data_folder))
# if data_directory not in sys.path:
#     sys.path.append(os.path.join(parent_directory, data_directory))

# Load an ARGO profile to use while running tests
profile_src = "data\\float-6902855-cyc-87-Oct-2019.nc"
test_profile = xr.open_dataset( profile_src )
plt_profile = pwp.translate_argo( test_profile )


# load meteorology

meteorology_src = "data\\single_point_Ike_2008.nc"
test_meteorology = xr.open_dataset( meteorology_src )
forcing = met.prepare_forcing( test_meteorology )

#%%
# plots
fig, ax1 = plt.subplots()

ax1.plot( plt_profile.dens, plt_profile.z, 'k-' )
ax1.set_xlabel ( "density" )
ax1.tick_params( axis='x', labelcolor='k' )

ax2 = ax1.twiny()
ax2.plot( plt_profile.temp, plt_profile.z, 'r-' )
ax2.set_xlabel( "temperature" )
ax2.tick_params( axis='x', labelcolor = 'r' )

ax2.set_ylabel( "depth" )
plt.gca().invert_yaxis()
plt.title( "test profile" )
plt.show()

#%%
# more plots
fig, ax1 = plt.subplots()

ax1.plot( forcing.time, forcing.taux )

ax1.plot( forcing.time, forcing.tauy, 'r' )


# %%
# run a pwp step
# Create instance of world to call methods
test_world = pwp.World( lat = 20 )
profile = pwp.World.interp_profile( test_world, plt_profile )
profile = pwp.pwp_step( test_world, profile, forcing )

# take a single pwp step



# %%
