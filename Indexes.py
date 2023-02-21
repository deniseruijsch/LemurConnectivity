# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Apr 11 16:39:06 2022

# @author: ruijsch
# """

from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ErrorCode, ParticleFile, ScipyParticle
from parcels import UnitConverter, Field, Variable
import numpy as np
import math
from datetime import timedelta, datetime
from datetime import timedelta as delta
from operator import attrgetter
from glob import glob
import copy
import xarray as xr
import matplotlib.pyplot as plt

import pandas as pd
from datetime import date, timedelta

#%%
#Load the data from Lorenz and define the indices
 
data_path_ocean = '/storage/shared/oceanparcels/input_data/MOi/'
# Load only a few time steps of the model output, to speed up this test simulation
ufiles = sorted(glob(data_path_ocean+'psy4v3r1/psy4v3r1-daily_U_*.nc'))[-1345:-970]
vfiles = [f.replace('_U_', '_V_') for f in ufiles]
wfiles = [f.replace('_U_', '_W_') for f in ufiles][:4]
mesh_mask = data_path_ocean + 'domain_ORCA0083-N006/coordinates.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},

             }
variables = {'U': 'vozocrtx', 'V': 'vomecrty'}
dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}

# Madagascar indices
indices = {'lon': np.arange(3500,4321).tolist(),
            'lat': np.arange(900, 1600).tolist()}

#%%

data_path_wind = '/storage/shared/oceanparcels/output_data/data_Mikael/ERA5/wind/'
# Load only a few time steps of the model output, to speed up this test simulation
wind_files = sorted(glob(data_path_wind+'ERA5_global_wind_monthly_*.nc'))[288:]

# the mesh mask comes from the ocean data path
mesh_mask = data_path_ocean + 'domain_ORCA0083-N006/coordinates.nc'


#%%

data_path_bathymetry = '/storage/shared/oceanparcels/input_data/MOi/'

maskfiles = data_path_bathymetry+'domain_ORCA0083-N006/bathymetry_ORCA12_V3.3.nc'

filenames_mask = {'mask': {'lon': maskfiles, 'lat': maskfiles, 'data': maskfiles}}

variables_mask = {'mask': 'mask'}
dimensions_mask = {'mask': {'lon': 'nav_lon', 'lat': 'nav_lat'}}

# Madagascar indices
indices_mask = {'lon': np.arange(3500,4321).tolist(),
            'lat': np.arange(900, 1600).tolist()}


#%%



coastfiles = 'distance_to_coast_22_04_22.nc'

filenames_coast = {'distance': {'lon': coastfiles, 'lat': coastfiles, 'data': coastfiles}}

variables_coast = {'distance': 'distance'}
dimensions_coast = {'distance': {'lon': 'lon', 'lat': 'lat'}}

# Madagascar indices -> it already has the right indices
# indices_coast = {'lon': np.arange(3500,4321).tolist(),
#             'lat': np.arange(900, 1600).tolist()}


boxfiles = 'box.nc'

filenames_coast = {'distance': {'lon': coastfiles, 'lat': coastfiles, 'data': coastfiles}}

variables_coast = {'distance': 'distance'}
dimensions_coast = {'distance': {'lon': 'lon', 'lat': 'lat'}}


#%%
#Make the fieldsets for the currents, wind, land/sea mask and the coast

# note the indices argument and the 'allow time extrapolation=True' to speed up simulation
fset_currents = FieldSet.from_nemo(filenames, variables, dimensions, indices=indices) #, allow_time_extrapolation=True)


#%%

withwind = 0.03  # wind factor of 3%
if withwind:
    windfiles = {'U': wind_files,
                 'V': wind_files}
    winddimensions = {'lon': 'lon', 
                      'lat': 'lat', 
                      'time': 'time'}
    windvariables = {'U': 'u10', 'V': 'v10'}
    # Madagascar indices
    indices_wind = {'lon': np.arange(750, 1000).tolist(),
               'lat': np.arange(200, 400).tolist()}
    fset_wind = FieldSet.from_nemo(windfiles, windvariables, winddimensions,indices=indices_wind)#, allow_time_extrapolation=True)
    fset_wind.U.set_scaling_factor(withwind)
    fset_wind.V.set_scaling_factor(withwind)
    

#%%


fset_mask = FieldSet.from_nemo(filenames_mask, variables_mask, dimensions_mask, indices=indices_mask)
fset_coast = FieldSet.from_nemo(filenames_coast, variables_coast, dimensions_coast)

#%%

if withwind:
    fieldset = FieldSet(U=fset_currents.U + fset_wind.U, V=fset_currents.V + fset_wind.V)
else:
    fieldset = FieldSet(U=fset_currents.U, V=fset_currents.V)
    
fieldset.add_field(fset_mask.mask)
fieldset.add_field(fset_coast.distance)



fieldset.mask.data[:,:,600:] = 0
fieldset.mask.data[:,:,:230] = 0
fieldset.mask.data[:,:200,:] = 0
fieldset.mask.data[:,570:,:] = 0

lonss = np.load('lons_001_present.npy')
latss = np.load('lats_001_present.npy')
depths = np.ones(np.shape(lonss)[0])*0.5

output_files = sorted(glob('2022_11_21_present_day_2013_wind/2022_11_20_present_day_2013_whole_grid_001_windage_3_day*.nc'))

parcels = xr.open_dataset(output_files[0])

start_lat = np.array(parcels.lat[:,0])
start_lon = np.array(parcels.lon.isel(obs=0))

def end_lon_lat(file):
    
    parcels = xr.open_dataset(file)
    
    end_lon = np.array(parcels.lon.isel(obs=24))
    end_lat = np.array(parcels.lat[:,-1])

    return end_lon, end_lat

end_lon = np.zeros((len(output_files),len(parcels.lat[:,-1])))
end_lat = np.zeros((len(output_files),len(parcels.lat[:,-1])))

for i in range(len(output_files)):
    print(i)
    end_lon_a, end_lat_a = end_lon_lat(output_files[i])
    end_lon[i,:] = end_lon_a
    end_lat[i,:] = end_lat_a
    
start_lon_lat = np.zeros((len(start_lon),2))

for i in range(len(start_lon)):
    start_lon_lat[i,0] = start_lon[i]
    start_lon_lat[i,1] = start_lat[i]
    
#start_lon_lat

end_lon_lat = np.zeros((len(output_files),np.shape(end_lon)[1],2))

for i in range(len(output_files)):
    print(i)
    for j in range(len(start_lon)):
        end_lon_lat[i,j,0] = end_lon[i,j]
        end_lon_lat[i,j,1] = end_lat[i,j]
        
np.save('2022_11_24_start_lon_lat_present_2013_wind.npy',(start_lon_lat))
np.save('2022_11_24_end_lon_lat_present_2013_wind.npy',(end_lon_lat))

start_lon_lat = np.load('2022_11_24_start_lon_lat_present_2013_wind.npy')
end_lon_lat = np.load('2022_11_24_end_lon_lat_present_2013_wind.npy')

X,Y = np.meshgrid(fieldset.mask.lon, fieldset.mask.lat)
p = fieldset.mask.data == 1

box_lon = X[p[0,:,:]]
box_lat = Y[p[0,:,:]]

coastal = np.load('2022_10_24_coastal_nodes_present.npy')


X__ = X[::4,::4]
Y__ = Y[::4,::4]
p__ = p[:,::4,::4]

box_lon__ = X__[p__[0,:,:]]
box_lat__ = Y__[p__[0,:,:]]

coastal_ = np.load('2022_10_24_coastal_nodes_present_low.npy')

# box_lon_lat__ = np.zeros((len(box_lon__),2))

# for i in range(len(box_lon__)):
#     box_lon_lat__[i,0] = box_lon__[i]
#     box_lon_lat__[i,1] = box_lat__[i]
    
    
# locations = box_lon_lat__

# np.save('2022_10_24_locations_present.npy',(locations))

locations = np.load('2022_10_24_locations_present.npy')

index_start = np.zeros(len(start_lon_lat))
index_end = np.zeros((len(output_files),len(start_lon_lat)))

for j in range(len(start_lon_lat)):
#for j in range(0,20):
    latvalue = start_lon_lat[j,:][1]
    lat_min = (np.abs(locations[:,1] - latvalue))
    idxlat_start = np.where(lat_min == lat_min.min())

    lonvalue = start_lon_lat[j,:][0]
    lon_min = (np.abs(locations[idxlat_start][:,0] - lonvalue))
    idxlon_start = np.where(lon_min == lon_min.min()) 
    idxlon_start = idxlon_start + idxlat_start[0][0]

    index_ = np.intersect1d(np.asarray(idxlon_start), np.asarray(idxlat_start))
    index__ = index_[0]
    #print(index_1)

    if len(index_) != 0:
        index_start[j] = index__
    else: 
        #index[i,j] = 10e10
        latvalue = idxlat_start[0][0]
        idx_min = (np.abs(idxlon - latvalue))
        a = np.argwhere(idx_min == idx_min.min())[0,1]
        index_start[j] = idxlon[0][a]
        
        
for i in range(len(output_files)):
    print(i)
    for j in range(np.shape(end_lon_lat)[1]):
        latvalue = end_lon_lat[i,j][1]
        lat_min = (np.abs(locations[:,1] - latvalue))
        idxlat_end = np.where(lat_min == lat_min.min())

        lonvalue = end_lon_lat[i,j][0]
        lon_min = (np.abs(locations[idxlat_end][:,0] - lonvalue))
        idxlon_end = np.where(lon_min == lon_min.min()) 
        idxlon_end = idxlon_end + idxlat_end[0][0]
        #print(idxlon_end,idxlat_end)

        index_ = np.intersect1d(np.asarray(idxlon_end), np.asarray(idxlat_end))
        #print(index_1)

        if len(index_) != 0:
            index_end[i,j] = index_[0]
        else: 
            #index[i,j] = 10e10
            latvalue = idxlat[0][0]
            idx_min = (np.abs(idxlon - latvalue))
            a = np.argwhere(idx_min == idx_min.min())[0,1]
            index_end[i,j] = idxlon[0][a]
            
np.save('2022_11_24_index_start_present_2013_001_wind.npy',(index_start))
np.save('2022_11_24_index_end_present_2013_001_wind.npy',(index_end))


index_start = np.load('2022_11_24_index_start_present_2013_001_wind.npy')
index_end = np.load('2022_11_24_index_end_present_2013_001_wind.npy')


#output_files = sorted(glob('2022_10_17_present_day_run/2022_10_17_present_day_whole_grid_001_windage_0_day*.nc'))

# G = np.zeros((len(output_files),len(locations),len(locations)))
# start = 0
# for year in range(len(output_files)):
#     print(year)
#     for i,j in zip(index_end[year+start],index_start):
#         G[year,int(i),int(j)] = G[year,int(i),int(j)] + 1
        
# afr_unique = np.load("2022_10_25_afr_unique_present.npy")
# mad_unique = np.load("2022_10_25_mad_unique_present.npy")

# for year in range(len(output_files)):
#     for mad in mad_unique:
#         G[year,:,int(mad)] = G[0,:,int(mad)] *0 #from node 1221 to all nodes
        
# np.save('2022_11_07_G_present_windage3.npy',(G)
        

