# ATM_waveform
Scripts to manipulate ATM (Automatic Topographic Mapper) waveform data

This repository contains scripts that can read the ATM waveform files available from the National Snow and Ice Datacenter (NSIDC):
* Waveforms from the wide-swath scanner (Green) : https://nsidc.org/data/ilatmw1b/versions/1
* Waveforms from the narrow-swath scanner (Green): https://nsidc.org/data/ilnsaw1b/versions/1
* Waveforms from the narrow-swath scanner (Infrared): https://nsidc.org/data/ilnirw1b/versions/1

It contains functions to :
* Read waveforms and metadata into a Python _waveform_ class
* Read calibration data to calculate instrument response functions
* Calculate statistics on the waveforms
* Match waveforms against waveform shapes predicted from scattering theory for granular media

# Installation

The respository can be installed using the standard Python setuptools script by running (in the repository directory)
> pip install .

Or, for an editable installation that installs links so the source files:

> pip install -e .

Once this is done, the package can be imported into python as:

> import ATM_waveform as aw

# Workflows

The main workflow envisaged for this code is to match ATM waveform data with predicted models of returns from scattering snow surfaces.  

To perform one of these fits, you will need:
* A file of ATM waveform data (see above for links)
* An impulse-response function estimate matching the scanner and the measurement campaign for the data file (see the _data_ subdirectory)
* A scattering response function file (typically the data/srf_green_full.h5) containing the expected scattering response from a flat snow surface with a density of 400 $kg m^{-2}$.

To generate a fit for a green-wavelength ATM file (in this example, GL_19_W/2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5), run:

> fit_ATM_scat_2color.py 1 2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5 2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5_q2_out.h5 -c G  -f GL_19_W/SRF_green_full.h5 -T GL_19_W/IRF_green.h5 -r 2
Here:
* 1 is the number of channels to fit
* 2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5 is the input file
* 2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5_q2_out.h5 is the output file
* -c G specifies that the name of the channel being fit is 'G' (green)
* -f SRF_green_full.h5 specifies the scattering response file (calculated for green light)
* -r 2 specifies that only every second waveform will be processed (to save processing time)

On a modern server, a single one-minute ATM file can take 1-2 hrs to process.

The output (2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5_q2_out.h5) can be read using the ATM_waveform.fit.data() class:
in Python:

> D=aw.fit.data().from_h5('2019.04.05/ILATMW1B_20190405_175200.atm6AT6.h5_q2_out.h5')

reads the data into a structure, containing latitude, longitude, sigma, and K0 fields, where K0 is the estimated grain size, and sigma is the estimated surface roughness.

Examples of data processing, including how to visualize waveform data and how to use the fit_atm_scat_2color.py script to output model waveforms are in the notebooks subdirectory.
