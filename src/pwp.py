import xarray as xr; import pandas as pd; 
from datetime import datetime, timedelta; 
import numpy as np; import seawater, gsw; 
from dataclasses import dataclass
import argo_filter


'''
Big tasks pending as of March 27, 2024:
    - Does World need all the responsibilities it currently has?
    - Steps 4.4, 4.5, 4.6 in algorithm below
    - Details for forcing: coupled vs. uncoupled (separate class? functions?)
    - Design tests for individual steps in algorithm
'''

''' 
This module is largely refactored from Earlew's pwp_python_00 package. Here, PWP functions have been broken into multiple subfunctions,
and relevant classes were to improve organization and modularity with other applications, such as coupling with atmospheric models.

@earlew describes the algorithm of his pwp.run() as 

        1) Set model parameters (see set_params function in PWP_helper.py). 
        2) Read in forcing and initial profile data.
        3) Prepare forcing and profile data for model run (see prep_data in PWP_helper.py).
            3.1) Interpolate forcing data to prescribed time increments.
            3.2) Interpolate profile data to prescribed depth increments.
            3.3) Initialize model output variables.
        4) Iterate the PWP model specified time interval:
            4.1) apply heat and salt fluxes
            4.2) rotate, adjust to wind, rotate
            4.3) apply bulk Richardson number mixing
            4.4) apply gradient Richardson number mixing
            4.5) apply drag associated with internal wave dissipation
            4.5) apply diapycnal diffusion       
        5) Save results to output file

'''

class simulation:
    '''
    Joint management of simulation input, rules, and orders.
    '''

    def __init__( self, world, prof0, forcing ):
        self.world = world
        self.prof0 = self.prep_profile( prof0 )
        self.forcing = forcing

    def prep_profile( self, profile, argo = True ):
        # Identify source of profile and apply necessary tasks
        # to get it ready to act as initial condition
        if argo:
            profile = translate_argo( profile );

        profile = self.world.interp_profile( profile )
        return profile
 


def translate_argo( xr_prof ):
    # Take in an argo profile, as saved by argo_cleaner, and prepare it for pwp
    # comes in with coordinate PRES, fields TEMP, CT, RHO, PSAL, more
    # needs to leave with coordinate z, fields 
    
    # get rid of repeated data
    xr_prof = argo_filter.simplify_profile( xr_prof ) 
    # change variable names
    xr_prof = xr_prof.rename( {'PRES':'z' , 'RHO':'dens', 'TEMP':'temp', 'PSAL':'sal'} )
    # extrapolate variables to surface using nearest datum
    new_z = np.concatenate( [ [0], xr_prof['z'].values ] );
    xr_prof = xr_prof.interp( z = new_z , kwargs = { 'fill_value':'extrapolate' } , 
                              method = 'nearest' )
    return xr_prof

@dataclass
class World: 
    ''' 
    This class stores PWP simulation parameters and some major methods involved in timestepping.
    Methods described here become available to any other functions once parameters are set. 
    All parameters are set to default values unless user decides otherwise. 
    This class exists to ensure consistency between steps and experiments.
    '''

    # parameters necessary for grids and vectors
    lat : float; 
    dz : float = 0.5; # in meters
    dt : float = 300; # in seconds
    zmax : float = 150; # in meters
    # parameters involved in mixing
    mld_thres : float = 0.05; # density threshold for MLD
    Ri_g : float = 0.25; # threshold gradient richardson number
    Ri_b : float = 0.65; # threshold bulk richardson number
    rkz : float = 1e-6; # backgroudn vertical diffusivity
    g : float = 9.81; #gravitational constant in m/s^2
    # thermodynamics and optics
    beta_1 : float = 0.6; # longwave extinction coefficient (meters)
    beta_2 : float = 20; # shortwave extinction coefficient
    cpw : float = 4183; # heat capacity of seawater
    drag_coef : float = 0.1; # internal wave drag, to be multiplied by f
    # set flags to decouple aspects of air_sea interaction
    wind_ON : bool = True; # momentum fluxes
    heat_ON : bool = True; # heat fluxes
    drag_ON : bool = True; # rayleigh friction on flow
    emp_ON : bool = True; # evap minus precip freshwater fluxes


    def f( self ):
        return 4 * np.pi * np.sin( self.lat / 180 * np.pi ) / 24 / 3600

    def make_absorption( self, profile ):
        rs1 = 0.6; # fraction of heat contained in shortwave
        ztop = profile['z'] - self.dz / 2; # depth of top of cells
        zbot = profile['z'] + self.dz / 2; # depth of bottom of cells
        # treat shortwave and longwave separately
        absorb_sw = rs1 * ( np.exp( - ztop / self.beta_1 ) - np.exp( - zbot / self.beta_1 ) ) 
        absorb_lw = ( 1 - rs1 ) * ( np.exp( -ztop / self.beta_2 ) - np.exp( - zbot / self.beta_2 ) );
        # save total absorbption
        tot_absorb = xr.DataArray( data = absorb_sw + absorb_lw , 
                            coords = { 'z' : profile['z'] } )
        return tot_absorb


    def interp_profile( self , profile, preset_u = None ):
        # Profile is xr.dataset with temp, sal, dens, u, v
        # determine depth of cell centers
        zvals = np.arange( self.dz / 2 , self.zmax + 0.01 - self.dz/2 , self.dz ); 
        profile = profile.interp( z = zvals );
        # add absorption profile
        profile['absorb'] = self.make_absorption( profile )
        
        # Predefined initial condition for ocean u,v 
        if preset_u is None:
            all_zeros = np.zeros( profile['absorb'].shape )
            my_coords = profile['absorb'].coords
            profile['u'] = xr.DataArray( data = all_zeros, 
                             coords = my_coords )
            profile['v'] = xr.DataArray( data = all_zeros, 
                             coords = my_coords )
        return profile

    def flag_forcing( self, forcing ):
        ''' 
        Apply forcing flags. This might fit better in a separate class that handles forcing exclusive and 
        that must be instantiated in relation to an instance of World. TBD as project evolves.
        '''
        if ~ wind_ON :
            forcing['taux'] = 0; 
            forcing['tauy'] = 0; 
        if ~ heat_ON :
            forcing['q_in'] = 0; 
            forcing['q_out'] = 0;
        if ~ emp_ON : 
            forcing['emp'] = 0; 
        return forcing

    def rotate( self, profile ):
        # Apply rotation that results from Coriolis over dt / 2
        f = self.f(); 
        profile['u'] += f * profile['v'] * self.dt / 2
        profile['v'] += - f * profile['u'] * self.dt / 2 
        return profile 

    def find_MLD( self, profile ):
        # find ml depth of profile and its index in z
        rho_diff = profile['dens'] - profile['dens'].isel( z = 0 ); 
        # find first index for which diff is above threshold
        mld_idx = np.flatnonzero( rho_diff > self.mld_thres )[0]
        assert mld_idx.size != 0, 'Error: ML depth is undefined'
        # get numerical value of MLD
        mld = profile['z'].isel( z = mld_idx )
        return mld, mld_idx

    def wind_on_ML( self, profile , forcing ):
        # apply momentum forcing to mixed layer
        mld, mld_idx = self.find_MLD( profile );
        mass = mld * profile['dens'][0]; 
        dU = forcing['taux'] / mass * self.dt 
        dV = forcing['tauy'] / mass * self.dt
        # ------ now add velocities to old profile
        profile['u'][ : mld_idx ] = profile['u'][ : mld_idx ] + dU
        profile['v'][ : mld_idx ] = profile['v'][ : mld_idx ] + dV;        
        return profile

    def update_surface( self, profile , forcing ):
        # Apply heat and salinity fluxes to uppermost level of a profile
        dT = ( forcing['q_in'] * profile['absorb'].isel( z = 0 )  + forcing['q_out'] ) \
                 * self.dt / ( self.dz * profile['dens'].isel( z = 0 ) * self.cpw )
        dS = forcing['emp'] * profile['sal'].isel( z = 0 ) * self.dt / self.dz
        # set new values
        print( dS.values )
        profile['temp'][0] += dT
        profile['sal'][0] += dS
        return profile

    def subsurface_sw( self, profile, forcing ):
        # apply downwelling sw radiation to subsurface layers
        dT = ( forcing['q_in'] * profile['absorb'][1:] ) * self.dt 
        dT = dT / ( self.dz * self.cpw * profile['dens'][1:] )
        # update temperature value
        profile['temp'][1:] += dT
        return profile 

    def rayleigh_friction( self, profile ):
        if self.drag_ON:
            drag_loss = self.drag_coef * self.f() * self.dt
            profile['u'] = profile['u'] * ( 1 - drag_loss )
            profile['v'] = profile['v'] * ( 1 - drag_loss )
        else: 
            pass
        return profile
   
    def bulk_mix( self, profile ):
        # mix based on bulk Ri, between thermocline levels and surface

        mld, mld_idx = self.find_MLD( profile ); 
        # get surface properties (might have to change for ML average)
        rho_0 = profile['dens'].isel( z = 0 )
        vel_0 = ( profile['u'] + 1j * profile['v'] ).isel( z = 0 );
        
        # now cycle through thermocline and apply mixing where necessary 
        for jj in range( mld_idx , len( profile['z'] ) ):
            dif_rho = profile['dens'][jj] - rho_0
            dif_vel = ( profile['u'] + 1j * profile['v'] ).isel( z = jj ) 
            dif_vel = np.abs( dif_vel - vel_0 ) ** 2 
            
            if dif_vel == 0:
                continue
            else:
                # Compute bulk Ri
                Ri_v = self.g * dif_rho / dif_vel / ( self.dz * jj )             
            
                # now use these values and call mixing routine if necessary
                if Ri_v > self.Ri_b:
                    continue
                else: 
                    profile = mix5( profile , 0, jj ); 
                    # mix jj to surface, as earlew
       
        return profile 

    def grad_mix ( self, profile ):
        # initialize Richardson number vector
        Ri = np.zeros( len( profile['z'] ) - 1 )

        # calculate initial Richardson numbers
        for jj in range ( 0 , len( profile['z'] ) - 1 ):
            dif_rho = profile['dens'][jj + 1] - profile['dens'][jj]
            dif_vel = ( profile['u'][jj + 1] - profile['u'][jj] )**2 + \
                ( profile['v'][jj + 1] - profile['v'][jj] )**2
            if dif_vel > 1e-6:
                Ri[jj] = self.g * self.dz * dif_rho / dif_vel
            else:
                Ri[jj] = np.inf
        
        # loop until all criticality is gone
        while True:

            # find smallest Ri and check criticality, exit if no criticality
            Ri_min = np.min(Ri)
            Ri_min_idx = np.argmin(Ri)
            if Ri_min > self.Ri_g:
                return profile
            
            # mix cells
            # don't really understand this first step, but it's the accepted process
            Ri_con = 0.02 + (self.Ri_g - Ri_min) / 2
            Ri_new = self.Ri_g + Ri_con / 5
            f = 1 - Ri_min / Ri_new
            
            # mix temp
            delta_temp = ( profile['temp'][Ri_min_idx + 1] - profile['temp'][Ri_min_idx] ) * f / 2
            profile['temp'][Ri_min_idx + 1] = profile['temp'][Ri_min_idx + 1] - delta_temp
            profile['temp'][Ri_min_idx] = profile['temp'][Ri_min_idx] + delta_temp
            
            # mix salinity
            delta_sal = ( profile['sal'][Ri_min_idx + 1] - profile['sal'][Ri_min_idx] ) * f / 2
            profile['sal'][Ri_min_idx + 1] = profile['sal'][Ri_min_idx + 1] - delta_sal
            profile['sal'][Ri_min_idx] = profile['sal'][Ri_min_idx] + delta_sal
            
            # mix velocities
            delta_u = ( profile['u'][Ri_min_idx + 1] - profile['u'][Ri_min_idx] ) * f / 2
            profile['u'][Ri_min_idx + 1] = profile['u'][Ri_min_idx + 1] - delta_u
            profile['u'][Ri_min_idx] = profile['u'][Ri_min_idx] + delta_u
            
            delta_v = ( profile['v'][Ri_min_idx + 1] - profile['v'][Ri_min_idx] ) * f / 2
            profile['v'][Ri_min_idx + 1] = profile['v'][Ri_min_idx + 1] - delta_v
            profile['v'][Ri_min_idx] = profile['v'][Ri_min_idx] + delta_v

            # calculate new density
            profile['dens'][Ri_min_idx : Ri_min_idx + 1] = gsw.rho_t_exact(
                    profile['sal'][Ri_min_idx : Ri_min_idx + 1], 
                    profile['temp'][Ri_min_idx : Ri_min_idx + 1],
                    profile['z'][Ri_min_idx : Ri_min_idx + 1] )
            
            # calculate new Ri
            jj_start = Ri_min_idx - 2
            if jj_start < 1: jj_start = 0
            jj_end = Ri_min_idx + 2
            if jj_end > len( profile['z'] ) - 1: jj_end = len( profile['z'] ) - 1

            for jj in range( jj_start , jj_end ):
                dif_rho = profile['dens'][jj + 1] - profile['dens'][jj]
                dif_vel = ( profile['u'][jj + 1] - profile['u'][jj] )**2 + \
                    ( profile['v'][jj + 1] - profile['v'][jj] )**2
                if dif_vel > 1e-10:
                    Ri[jj] = self.g * self.dz * dif_rho / dif_vel
                else:
                    Ri[jj] = np.inf



def pwp_step( world, profile, forcing ):
    # Apply PWP algorithm i
    profile = profile.copy()
    # Heat and salinity fluxes at the surface and then below
    profile = world.update_surface( profile, forcing );
    profile = world.subsurface_sw( profile, forcing );          
    # --- might want to add a point checking for freezing here
    profile['dens'] = gsw.rho_t_exact( profile['sal'], 
                          profile['temp'], profile['z'] ); # update density
    # --- relieve static instability
    profile = remove_static_instability( profile )
    # apply momentum flux and coriolis rotation
    profile = world.rotate( profile );
    profile = world.wind_on_ML( profile, forcing ); 
    profile = world.rayleigh_friction( profile );
    profile = world.rotate( profile ); # coriolis for dt / 2
    # time to apply mixing parameterizations
    profile = world.bulk_mix( profile );
    # gradient mixing
    profile = world.grad_mix( profile );

    # diffusion
    return profile 

# -------------- below are pwp functions independent of simulation parameters

def remove_static_instability( profile ):
    # Find and relieve any static instability in density array
    # doesn't involve any simulation parameters
    stat_unstable = True;
    print( 'removing static instability' )
    while stat_unstable:
        # compute density gradient and find unstable points
        rho_grad = - profile['dens'].differentiate( 'z' ); 
        
        if np.any( rho_grad > 0 ):
            # means we found an unstable point
            stat_unstable = True 
            # get index of first unstable point
            inst0_idx = np.flatnonzero( rho_grad > 0 )[0]
            # prepare to mix 2 cells above and 2 cells below
            inst_top = max( 0, inst0_idx - 2 ) # avoid going beyond array 
            inst_bot = min( len( profile['z'].values ) - 1 , inst0_idx + 2 )
            # apply mixing
            profile = mix5( profile , inst_top, inst_bot );
        
        else: 
            # no unstable points left
            stat_unstable = False
    
    return profile


def mix5( profile, ztop, zbot ):
    # removed from world because it doesn't use world properties
    # mix all fluid properties between ztop and zbot  
    # need to verify definitions, because earlew sets ztop as surface
    vars2change = ['u','v','temp','sal']
    for key in vars2change : 
        mval = profile[key][ ztop : zbot ].mean('z')
        profile[key][ ztop : zbot ] = mval; 
        
    # update density with the new temp, salt
    profile['dens'][ ztop : zbot ] = gsw.rho_t_exact( \
                 profile['sal'][ztop:zbot] , 
                 profile['temp'][ztop:zbot] ,
                 profile['z'][ztop:zbot] )
    return profile 

# ------------------------------------------------------------
# ------------------- ROUTINES THAT DEAL WITH FORCING --------
# ------------------- WERE MOVED TO MET.PY -------------------
    






