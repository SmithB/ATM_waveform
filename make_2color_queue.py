#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:32:10 2019

@author: ben
"""

import glob
import re
import sys

thedir=sys.argv[1]

IR_files=glob.glob(thedir+'/IR/I*.h5')
green_files=glob.glob(thedir+'/green/I*.h5')

file_re=re.compile('(\d+_\d+)')

IR_dict={}
for file in IR_files:
    IR_dict[file_re.search(file).group(1)]=file
green_dict={}
for file in green_files:
    green_dict[file_re.search(file).group(1)]=file

with open('IR_green_queue.txt','w') as fh:
    for key in green_dict.keys():
        if key in IR_dict:
            fh.write("python3 fit_ATM_scat_2color.py %s %s %s_out.h5 -f SRF_IR_full.h5 SRF_green_full.h5 -T TX_IR.h5 TX_green.h5\n" % (IR_dict[key], green_dict[key], key))
