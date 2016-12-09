'''
Class for doing stuff with output from AccuRT.
'''

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter1d as gaussf
from file_reading import *

class ReadART(object):
    '''Reads the output text files from AccuRT, and includes methods for
    calculating albedo and transmittance, and for simple plotting.

    Assumes that the number of detector depths and wavelengths
    does *not* change when using repeated runs.

    Positional argument:

    expname -- Name of the main config file.

    Keyword arguments:

    basefolder -- Folder where the main configfile is located.
        Default './'.

    cosine -- Boolean. If True (which is default), read in the
        total cosine irradiance.

    diffuse -- Boolean. If True, read in the diffuse cosine irradiance.
        Default False.

    direct -- Boolean. If True, read in the direct cosine irradiance.
        Default False.

    runvarfile -- Filename or list-like structure holding indices for repeated
        runs. Default None.

    scalar -- Boolean. If True, read in total scalar irradiance
        Default False.

    sine -- Boolean. If True, read in total sine weighted irradiance
        Default False.

    radiance -- Boolean. If True, read radiance.
        Default False.

    iops -- Boolean. If True, read in iops-file into a dict.
        Default False.


     Example:
     >>> a = pyAccu('atmOcean')
     >>> transm = a.transmittance(layers=(2,4),integrated=True)
     >>> a.plot()
     >>> a.plot(profile=True)
     '''

    def __init__(self, expname, basefolder='./', cosine=True,
                 diffuse=False, direct=False,
                 runvarfile=None, scalar=False, iops=False,
                 radiance=False, sine=False, material_profile=False):
        '''See PyAccu for description of arguments.'''

        self.has_cosine = False
        self.has_diffuse = False
        self.has_direct = False
        self.has_scalar = False
        self.has_sine = False
        self.has_iops = False

        outputfolder = expname + 'Output'

        fn_fmt = '{0}_irradiance_{1}_{2}ward.txt'

        if cosine:
            self.has_cosine = True
            cos_u_file = fn_fmt.format('cosine', 'total', 'up')
            cos_d_file = fn_fmt.format('cosine', 'total', 'down')

            cos_u_path = os.path.join(basefolder, outputfolder, cos_u_file)
            cos_d_path = os.path.join(basefolder, outputfolder, cos_d_file)

            self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths, self.updata = \
                read_irradiance(cos_u_path)
            *_, self.downdata = read_irradiance(cos_d_path)

        if diffuse:
            self.has_diffuse = True
            diff_u_file = fn_fmt.format('cosine', 'diffuse', 'up')
            diff_d_file = fn_fmt.format('cosine', 'diffuse', 'down')

            diff_u_path = os.path.join(basefolder, outputfolder, diff_u_file)
            diff_d_path = os.path.join(basefolder, outputfolder, diff_d_file)

            *_, self.diffuse_down = read_irradiance(diff_d_path)
            *_, self.diffuse_up = read_irradiance(diff_u_path)

            if not hasattr(self, 'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                    self.depths, self.wavelengths = _

        if direct:
            self.has_direct = True
            dir_u_file = fn_fmt.format('cosine', 'direct', 'up')
            dir_d_file = fn_fmt.format('cosine', 'direct', 'down')

            dir_u_path = os.path.join(basefolder, outputfolder, dir_u_file)
            dir_d_path = os.path.join(basefolder, outputfolder, dir_d_file)
            *_, self.direct_down = read_irradiance(dir_d_path)
            *_, self.direct_up = read_irradiance(dir_u_path)

            if not hasattr(self, 'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                  self.depths, self.wavelengths = _

        if scalar:
            self.has_scalar = True
            sclr_u_file = fn_fmt.format('scalar', 'total', 'up')
            sclr_d_file = fn_fmt.format('scalar', 'total', 'down')

            sclr_u_path = os.path.join(basefolder, outputfolder, sclr_u_file)
            sclr_d_path = os.path.join(basefolder, outputfolder, sclr_d_file)

            *_, self.scalar_down = read_irradiance(sclr_d_path)
            *_, self.scalar_up = read_irradiance(sclr_u_path)

            if not hasattr(self, 'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                  self.depths, self.wavelengths = _

        if sine:
            self.has_sine = True
            sine_u_file = fn_fmt.format('sine', 'total', 'up')
            sine_d_file = fn_fmt.format('sine', 'total', 'down')

            sine_u_path = os.path.join(basefolder, outputfolder, sine_u_file)
            sine_d_path = os.path.join(basefolder, outputfolder, sine_d_file)

            *_, self.sine_down = read_irradiance(sine_d_path)
            *_, self.sine_up = read_irradiance(sine_u_path)

            if not hasattr(self, 'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                    self.depths, self.wavelengths = _

        if radiance:
            outputfolder = expname + 'Output'
            filename = os.path.join(basefolder, outputfolder, 'radiance.txt')
            self.radiance, self.polarangles, self.azimuthangles, *_ = \
                read_radiance(filename)

            if not hasattr(self, 'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                   self.depths, self.wavelengths = _

        if iops:
            self.has_iops = True
            iops_path = os.path.join(basefolder, outputfolder, 'iops.txt')
            self.iops = read_iops(iops_path)

        if material_profile:
            mp_path = os.path.join(basefolder, outputfolder,
                                   'material_profile.txt')
            self.material_profile = read_material_profile(mp_path)

        if isinstance(runvarfile, str):
            try:
                self.runvar = np.loadtxt(os.path.join(basefolder, runvarfile))
            except FileNotFoundError:
                print('{0} not a valid filename'.format(runvarfile))
                self.runvar = runvarfile
        else:
            try:
                self.runvar = np.array(runvarfile)
            except TypeError:
                self.runvar = 'No multi-run info provided'

        with open(os.path.join(basefolder, outputfolder,
                               'version.txt'), 'r') as ver:
            self.modelversion = ver.readline()[:-1]


    def writefile(self, filename, output=None):
        '''output is 'matlab' or 'netcdf'.'''

        if filename.endswith('.mat'):
            output = 'matlab'
            filename = os.path.splitext(filename)[0]
        elif filename.endswith('.nc'):
            output = 'netcdf'
            filename = os.path.splitext(filename)[0]
        if output == 'matlab':
            data = dict(up=self.updata,
                        down=self.downdata,
                        nRuns=self.nruns,
                        nwavelengths=self.nwavelengths,
                        nDepths=self.ndepths,
                        nStreams=self.nstreams,
                        wavelengths=self.wavelengths,
                        depths=self.depths,
                        runvar=self.runvar,
                        modelversion=self.modelversion)
            if self.has_scalar:
                data['scalar_up'] = self.scalar_up
                data['scalar_down'] = self.scalar_down
            if self.has_direct:
                data['direct_up'] = self.direct_up
                data['direct_down'] = self.direct_down
            if self.has_iops:
                data['iops'] = self.iops

            sio.savemat('{0}.mat'.format(filename), data)

        elif output == 'netcdf':

            outfile = sio.netcdf_file(filename + '.nc', 'w')
            outfile.history = 'Output from AccuRT model, ' + self.modelversion

            outfile.createDimension('depth', self.ndepths)
            outfile.createDimension('wavelength', self.nwavelengths)
            outfile.createDimension('multirun', self.nruns)

            depths = outfile.createVariable('depths', 'float32', ('depth',))
            wavelengths = outfile.createVariable('wavelength', 'int32',
                                                 ('wavelength',))
            multirun = outfile.createVariable('runvar', 'float', ('multirun',))

            depths[:] = self.depths
            depths.unit = 'm'
            depths.reference = 'Top of Atmosphere'
            wavelengths[:] = self.wavelengths
            if isinstance(self.runvar, str):
                multirun[:] = np.arange(self.updata.shape[2])
            else:
                multirun[:] = self.runvar

            upward_irradiance = \
                outfile.createVariable('upward_irradiance', 'float',
                                       ('depth', 'wavelength', 'multirun'))

            upward_irradiance[:, :, :] = self.updata
            downward_irradiance = \
                outfile.createVariable('downward_irradiance', 'float',
                                       ('depth', 'wavelength', 'multirun'))

            downward_irradiance[:, :, :] = self.downdata

            outfile.close()

    def plot(self, profile=False, run=1, direction='down', ax=None):
        '''Plots data from one of the runs, either as a vertical profile or
        as spectra. Either upwelling or downwelling irradiance.'''
        if ax is None:
            _, ax = plt.subplots()

        if direction == 'up':
            data = self.updata[:, :, run-1]
        elif direction == 'down':
            data = self.downdata[:, :, run-1]

        if profile:
            ax.plot(data, self.depths)
            ax.set_ylabel('Depth below TOA [m]')
            ax.set_xlabel('Irradiance [W/m2]')
            ax.invert_yaxis()
            ax.legend([str(l) for l in self.wavelengths],
                      loc='best',
                      title='Wavelength [nm]')
        else:
            ax.plot(self.wavelengths, data.T)
            ax.set_xlabel('Wavelength [nm]')
            ax.set_ylabel('Irradiance [W/m2]')
            ax.legend([str(l) for l in self.depths],
                      loc='best',
                      title='Depth below TOA [m]')

        return ax

    def albedo(self, layer, integrated=False):
        '''Calculate albedo, return array.'''

        if layer == 'all':
            if integrated:
                albedo = np.trapz(self.updata, x=self.wavelengths, axis=1) / \
                         np.trapz(self.downdata, x=self.wavelengths, axis=1)
            else:
                albedo = self.updata / self.downdata

        else:
            incident = self.downdata[layer, :, :]
            reflected = self.updata[layer, :, :]

            if integrated:
                albedo = np.trapz(reflected, x=self.wavelengths, axis=0) / \
                         np.trapz(incident, x=self.wavelengths, axis=0)
            else:
                albedo = reflected / incident

        return albedo

    def transmittance(self, layers, integrated=False, wlrange=None):
        '''Calculate transmittance between levels given by 2-tuple layers.'''

        incident = self.downdata[layers[0], :, :]
        outgoing = self.downdata[layers[1], :, :]

        if integrated:
            if wlrange is None:
                transm = np.trapz(outgoing, x=self.wavelengths, axis=0) / \
                    np.trapz(incident, x=self.wavelengths, axis=0)
            else:
                lower_wl = np.abs(self.wavelengths-wlrange[0]).argmin()
                upper_wl = np.abs(self.wavelengths-wlrange[1]).argmin()
                if (self.wavelengths[lower_wl] > wlrange[0])\
                        and (lower_wl > 0):
                    lower_wl -= 1
                if self.wavelengths[upper_wl] < wlrange[1]:
                    upper_wl += 2
                else:
                    upper_wl += 1

                transm = \
                    np.trapz(outgoing[lower_wl:upper_wl, :],
                             x=self.wavelengths[lower_wl:upper_wl], axis=0) / \
                    np.trapz(incident[lower_wl:upper_wl, :],
                             x=self.wavelengths[lower_wl:upper_wl], axis=0)
        else:
            transm = outgoing / incident

        return transm

    def gauss_smooth(self, n=5, inplace=False):
        '''Smooth data with a Gaussian filter.
        '''

        if inplace:

            if self.has_cosine:
                self.downdata = gaussf(self.downdata, sigma=n, axis=1)
                self.updata = gaussf(self.updata, sigma=n, axis=1)

            if self.has_direct:
                self.direct_down = gaussf(self.direct_down, sigma=n, axis=1)
                self.direct_up = gaussf(self.direct_up, sigma=n, axis=1)

            if self.has_diffuse:
                self.scalar_down = gaussf(self.scalar_down, sigma=n, axis=1)
                self.scalar_up = gaussf(self.scalar_up, sigma=n, axis=1)

            if self.has_scalar:
                self.scalar_down = gaussf(self.scalar_down, sigma=n, axis=1)
                self.scalar_up = gaussf(self.scalar_up, sigma=n, axis=1)

            if self.has_sine:
                self.scalar_down = gaussf(self.scalar_down, sigma=n, axis=1)
                self.scalar_up = gaussf(self.scalar_up, sigma=n, axis=1)

        else:
            modeldata = copy.deepcopy(self)
            if modeldata.has_cosine:
                modeldata.downdata = gaussf(modeldata.downdata,
                                            sigma=n, axis=1)
                modeldata.updata = gaussf(modeldata.updata, sigma=n, axis=1)

            if modeldata.has_direct:
                modeldata.direct_down = gaussf(modeldata.direct_down,
                                               sigma=n, axis=1)
                modeldata.direct_up = gaussf(modeldata.direct_up,
                                             sigma=n, axis=1)

            if modeldata.has_diffuse:
                modeldata.scalar_down = gaussf(modeldata.scalar_down,
                                               sigma=n, axis=1)
                modeldata.scalar_up = gaussf(modeldata.scalar_up,
                                             sigma=n, axis=1)

            if modeldata.has_scalar:
                modeldata.scalar_down = gaussf(modeldata.scalar_down,
                                               sigma=n, axis=1)
                modeldata.scalar_up = gaussf(modeldata.scalar_up,
                                             sigma=n, axis=1)

            if modeldata.has_sine:
                modeldata.scalar_down = gaussf(modeldata.scalar_down,
                                               sigma=n, axis=1)
                modeldata.scalar_up = gaussf(modeldata.scalar_up,
                                             sigma=n, axis=1)
            return modeldata

    def calc_heatingrate(self):
        '''Calculate readiative heating rate using Gershun's law.'''
        if not (self.has_scalar and self.has_iops):
            raise AttributeError(
                ('Scalar irradiance and IOPs not available,'
                 ' you need scalar=True, iops=True'))
        absorbed_energy = np.empty_like(self.scalar_down)
        for k in range(absorbed_energy.shape[2]):
            if absorbed_energy.shape[2] == 1:
                layerdepths = self.iops['layer_depths']
                abscoeff = self.iops['absorption_coefficients']
            else:
                layerdepths = self.iops['layer_depths'][k]
                abscoeff = self.iops['absorption_coefficients'][k]
            layerind = [np.where(layerdepths >= dd)[0][0]
                        for dd in self.depths]
            for i, j in enumerate(layerind):
                absorbed_energy[i, :, k] = abscoeff[j] * \
                    (self.scalar_down[i, :, k] + self.scalar_up[i, :, k])
        absorbed_energy[absorbed_energy < 0] = 0

        return absorbed_energy

    def diffuse_attenuation(self, integrated=False):
        '''Calculate diffuse attenuation coefficient.'''
        delta_z = np.diff(self.depths)
        if integrated:
            dint = np.trapz(self.downdata, axis=1, x=self.wavelengths)
            diffuse_att_coeff = 1/delta_z * np.log(dint[:-1, :]/dint[1:, :])
        else:
            diffuse_att_coeff = 1/delta_z * \
                np.log(self.downdata[:-1, :, :]/self.downdata[1:, :, :])
        return diffuse_att_coeff
