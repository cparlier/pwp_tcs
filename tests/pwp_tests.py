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
forcing_full = met.prepare_forcing( test_meteorology )

# temporarily force into a single value
forcing = forcing_full.isel(time=200)



# %%
# set things up and create instance of world to call methods
test_world = pwp.World( lat = 20 )
profile = pwp.World.interp_profile( test_world, plt_profile )

#%%
# fix inputs temporarily
forcing['q_in'] = 0; forcing['taux'] = -1; forcing['tauy'] = -1; test_world.dt = 1000
#%%
# first running comparison plot
step_num = 1
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
fig, ax = plt.subplots()
ax.plot( profile.temp, profile.z, colors[step_num % len(colors)], label=f"Step {step_num}")
ax.set_xlabel( "temperature" )
ax.set_ylabel( "depth" )
ax.invert_yaxis()
plt.title( "temperature profiles as pwp steps run" )
ax.legend()
plt.show()


#%%
# take a pwp step and update plot
step_num += 1
profile = pwp.pwp_step( test_world, profile, forcing )

# update plot
ax.plot( profile.temp, profile.z, colors[step_num % len(colors)], label=f"Step {step_num}")
ax.legend()
fig



# %%
# temp code
# forcing['q_in'] = 1000; forcing['taux'] = -1; forcing['tauy'] = -1; test_world.dt = 1000




#%%
# more plots
# forcing plot
# fig, ax1 = plt.subplots()
# ax1.plot( forcing.time, forcing.taux )
# ax1.plot( forcing.time, forcing.tauy, 'r' )

# density plot
fig, ax1 = plt.subplots()
ax1.plot(profile.dens, profile.z)
ax1.set_xlabel('density')
ax1.set_ylabel('depth')
ax1.invert_yaxis()


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