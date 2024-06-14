import numpy as np; import xarray as xr;


def prepare_forcing( met_xr ):
    # Give variables standard names
    met_xr = translate_met( met_xr );
    # Now compute wind stress, heat flux, and salinity/fresh
    met_xr['taux'], met_xr['tauy'] = get_tau( met_xr )
    met_xr['q_in'] = met_xr['q_sw']
    met_xr['q_out'] = met_xr['q_lw'] + met_xr['q_lat'] \
                         + met_xr['q_sen']
    met_xr['emp'] = np.abs( met_xr['evap'] ) - np.abs( met_xr['prec'] )
    return met_xr

    #         raise Exception('Variable ' + var + ' not found in xarray.' )
    #     elif len( which_name ) > 1 :
    #         raise Exception('Found multiple matches for variable ' + var + ' in xarray.')
    #     else:
    #         # Restructure var_name dict such that current name is key
    #         var_names[ which_name[0] ] = var;
    #         del var_names[var]

    # # Now change names of variables in xr_obj
    # met_xr = met_xr.rename( var_names )
    # return met_xr


def translate_met( met_xr ):
    # Take in met data and translate variable names so pwp 
    # can understand. 

    # Create dictionary whose keys are correct names, and 
    # assigned are lists of alternate names
    var_names = { 'q_sw' : ['msnswrf'], 'q_lw' : ['msnlwrf'],
                  'q_lat' : ['mslhf'] , 'q_sen' : ['msshf'],
                  'prec' : ['mtpr'] , 'evap' : ['mer'] ,
                  'u10' : ['u10'], 'v10' : ['v10'] }
    original_keys = list( var_names.keys() )

    # Cycle through variables
    for var in original_keys:
        possible_names = var_names[var]
        # Check which of possible names is stored in met_xr
        which_name = [ var_in_obj for var_in_obj \
                       in list( met_xr.variables ) if \
                       var_in_obj in possible_names ]
        if len( which_name ) == 0 :
            raise Exception('Variable ' + var + ' not found in xarray.' )
        elif len( which_name ) > 1 :
            raise Exception('Found multiple matches for variable ' + var + ' in xarray.')
        else:
            # switch name of variables in dict
            var_names[ which_name[0] ] = var;
            del var_names[var]

    # Now change names of variables in xr_obj
    met_xr = met_xr.rename( var_names )
    return met_xr

def get_tau( met_xr ):
    # Compute wind stress using the speed-dependent drag of Large and Pond (1981)
    # with a correction for high speeds following Powell et al (2003)
    compvec = met_xr['u10'] + 1j*met_xr['v10']
    speed = np.abs( compvec )
    # Create masks for each speed category
    slow = speed <= 11;
    moderate =  ( ~ slow ) * ( speed < 30 )
    fast = speed > 30;
    # Set values for Cd
    Cd = np.zeros( compvec.shape );
    Cd[slow] = 1.2e-3;
    Cd[moderate] = ( 0.49 + 0.065 * speed[moderate] )*1e-3
    Cd[fast] = 1.9e-3
    Cd = xr.DataArray( data = Cd , coords = compvec.coords )
    # Use Cd to compute wind stress
    taux = 1.22 * Cd * speed * met_xr['u10']
    tauy = 1.22 * Cd * speed * met_xr['v10']
    return taux, tauy



