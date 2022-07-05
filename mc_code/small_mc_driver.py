#! /usr/bin/env python
import argparse
import numpy as np


parser=argparse.ArgumentParser(description='run the small_mc c code')
parser.add_argument('--photons', '-p',  type=int, help="number of photons per run")
parser.add_argument('--asymmetry_factor', '-g', default=0, type=float, help='asymmetry factor')
parser.add_argument('--num_runs', '-N', type=int, help='number of runs')
parser.add_argument('--output_base','-o', type=str, default='mc_out', help='output name base')
parser.add_argument('--queue_file','-q', type=str, default='mc_queue.txt')
parser.add_argument('--layer_thickness','-l', type=float, default=1000, help='layer thickness, giving maximum photon depth')
args=parser.parse_args()


with open(args.queue_file,'w') as fh:
    for ii in range(args.num_runs):
        fh.write(f'small_mc_dist_depth -p {args.photons} -g {args.asymmetry_factor} -l {args.layer_thickness} -o {args.output_base}_{ii+1}.dat\n')


