import netCDF4
import os
import numpy as np
import datetime as dt
import re
#%load_ext autoreload
#%autoreload 2


def to_nc(self, filename, replace=True, group='', meta_dict_file=None, 
          meta_dict=None, group_attrs=None, Verbose=False,
          default_fill_value=-9999):
    """
    write a data object to an netCDF4 file
    """
    
    nctype = {'float64':'f8',
              'float32':'f4',
              'int8':'i1',
              'int16':'i2',
              'int32':'i32'}

    # check whether overwriting existing files
    # append to existing files as default
    mode = 'w' if replace else 'a'

    if mode == 'w' and os.path.isfile(filename):
        os.remove(filename)

    fh_out=netCDF4.Dataset(filename, mode, format="NETCDF4")
    fh_out.setncattr('Conventions','CF-1.6')
    
    groups={None:fh_out}
    if group is not None and group != '':
        if not group in fh_out.groups:
            #fh_out=fh_out.groups[group]
            groups[None]=fh_out.createGroup(group)
    
    field_dict={}
    if meta_dict is None:
        field_dict = {field:field for field in self.fields}
    else:
        field_dict = {}
        for field, this_md in meta_dict.items():
            if this_md['source_field'] is not None and this_md['source_field'] in self.fields:
                field_dict[field] = this_md['source_field']
            elif field in self.fields:
                field_dict[field] = field

    # loop over metadata, create groups
    for out_field, md in meta_dict.items():
        if 'group' in md and md['group'] is not None and md['group'] != '/':
            if md['group'] not in groups:
                groups=[md['group']]=groups[None].createGroup(md['group'])

    # loop over metadata, create dimensions
    for out_field, md in meta_dict.items():
        
        if 'Dimensions' in md and md['Dimensions'] in ['yes','Yes','True']:
            if Verbose:
                print('creating dimension '+out_field)
            if 'source_field' in md and md['source_field'] is not None:
                size=getattr(self, md['source_field']).size
            else:
                size=getattr(self, out_field).size
            if 'group' in md and md['group'] in groups:
                groups[md['group']].createDimension(out_field, size)
        
    for out_field, field in field_dict.items():

        this_data=getattr(self, field)
        this_meta=meta_dict[out_field]
        maxshape=this_data.shape
        if Verbose:
            print( f'applying metadata for field {out_field}:\n\t{this_meta}')
        precision=None
        try: 
            precision=int(this_meta['precision'])
        except (TypeError, KeyError):
            precision=None
            
        fill_value=default_fill_value
        if 'fill_value' in this_meta and this_meta['fill_value'] is not None:
            fill_value=this_meta['fill_value']
               
        data = np.nan_to_num(this_data, nan=fill_value)
        dim=this_meta['Dimensions']
        this_group = this_meta['group']
        if this_group=='/' or this_group=='':
            this_group=None
        if dim in ['yes','Yes','True']:
            dim=out_field_name
        dsetvar = groups[this_group].createVariable(out_field,
                                    nctype[str(this_data.dtype)],
                                    dimensions=dim, 
                                    zlib=True,
                                    least_significant_digit=precision,
                                    fill_value=fill_value)
        dsetvar[:] = data
        if meta_dict is not None and out_field in meta_dict:
            for key, val in meta_dict[out_field].items():
                if key.lower() not in ['group','source_field','precision','dimensions','coordinate']:
                    dsetvar.setncattr(key,str(val).encode('utf-8'))
    fh_out.close()




## test code
import ATM_waveform  as aw

in_file='ILATMW1B_20170717_144930.atm6AT5.h5_q2_out.h5'
out_file='ATM_gs_ILATMW1B_20170717_144930.atm6AT5.nc'

if os.path.isfile(out_file):
    os.remove(out_file)

D=aw.fit.data().from_h5('ILATMW1B_20170717_144930.atm6AT5.h5_q2_out.h5')
D.assign(shot_count=D.shot.copy())
import csv
meta_dict={}
with open('waveform_archive_meta.csv','r') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        for key, val in row.items():
            if val=='':
                row[key]=None
        meta_dict[row['name']]=row

time_re=re.compile('_(\d\d\d\d)(\d\d)(\d\d)')
date=list(time_re.search(D.filename).groups())
meta_dict['time']['units']=f'seconds since {".".join(date)} 00:00:00'

to_nc(D, out_file, replace=True, meta_dict_file=None, 
          meta_dict=meta_dict, group_attrs=None, group='junk')
