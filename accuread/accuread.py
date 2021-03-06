'''
Proviedes ReadART class for doing stuff with output from AccuRT.
'''

import os
import copy
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter1d as gaussf
from .file_reading import read_irradiance, read_radiance, \
    read_material_profile, read_iops


class ReadART(object):
    '''Reads the output text files from AccuRT, and includes methods for
    calculating albedo and transmittance, and for simple plotting.

    Assumes that the number of detector depths and wavelengths
    does *not* change when using repeated runs.

    Args
    ----

    expname : string
        Name of the main config file.

    basefolder : string, optional
        Folder where the main configfile is located, default is './'.

    cosine : bool { True, False }
        If True (which is default), read in the total cosine irradiance.

    diffuse : bool { False, True }
        If True, read in the diffuse cosine irradiance.

    direct : bool { False, True }
        If True, read in the direct cosine irradiance.

    runvarfile : string or iterable
        A string should be a valid filename for a file that can be read
        with np.loadtxt, containing the variable for repeated runs.
        If list or array-like, turned into array.

    scalar : bool { False, True }
        If True, read in total scalar irradiance.

    sine : bool { False, True }
        If True, read in total sine weighted irradiance.

    radiance : bool { False, True }
        If True, read radiance.

    iops : bool { False, True }
        If True, read iops-file into a dict.

    material_profile : bool { False, True }
        If True, read material_profile-file.

    Returns
    -------
        ReadART instance

     Example:
     >>> import accuread as ar
     >>> a = ar.ReadART('atmOcean')
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
        self.has_radiance = False
        self.has_materialprofile = False
        self.expname = expname

        outputfolder = self.expname + 'Output'

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
            self.has_radiance = True
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
            if not hasattr(self, 'nruns'):
                self.nruns = self.iops['nruns']
                self.wavelengths = self.iops['wavelengths']

        if material_profile:
            self.has_materialprofile = True
            mp_path = os.path.join(basefolder, outputfolder,
                                   'material_profile.txt')
            self.material_profile = read_material_profile(mp_path)

        if isinstance(runvarfile, str):
            try:
                self.runvar = np.loadtxt(os.path.join(basefolder, runvarfile))
            except FileNotFoundError:
                print('{0} not a valid filename'.format(runvarfile))
                self.runvar = runvarfile
        elif runvarfile is None:
            self.runvar = np.arange(self.nruns)
        else:
            try:
                self.runvar = np.array(runvarfile)
            except TypeError:
                self.runvar = 'No multi-run info provided'

        with open(os.path.join(basefolder, outputfolder,
                               'version.txt'), 'r') as ver:
            self.modelversion = ver.readline()[:-1]

    def __repr__(self):
        txt = ['AccuRT output from {0}'.format(self.expname)]
        if hasattr(self, 'nruns'):
            txt.append('Number of runs: {0}'.format(self.nruns))
        if hasattr(self, 'nwavelengths'):
            txt.append('Wavelengths: {0} from {1}nm to {2}nm'.format(
                self.nwavelengths,
                self.wavelengths.min(),
                self.wavelengths.max()
                ))
        if hasattr(self, 'ndepths'):
            txt.append('Detectors: {0}'.format(self.ndepths))
        txt.append('Output:')
        if self.has_cosine:
            txt.append(' - Total cosine irradiance')
        if self.has_direct:
            txt.append(' - Direct cosine irradiance')
        if self.has_diffuse:
            txt.append(' - Diffuse cosine irradiance')
        if self.has_scalar:
            txt.append(' - Total scalar irradiance')
        if self.has_sine:
            txt.append(' - Total sine irradiance')
        if self.has_radiance:
            txt.append(' - Radiance')
        if self.has_materialprofile:
            txt.append(' - Material profile')
        if self.has_iops:
            txt.append(' - IOPs')
        return '\n'.join(txt)

    def writefile(self, filename, output=None):
        '''output is 'matlab' or 'netcdf'.'''

        if filename.endswith('.mat'):
            output = 'matlab'
            filename = os.path.splitext(filename)[0]
        elif filename.endswith('.nc'):
            output = 'netcdf'
            filename = os.path.splitext(filename)[0]
        if output == 'matlab':
            data = OrderedDict()
            data['nRuns'] = self.nruns
            data['nwavelengths'] = self.nwavelengths
            data['nDepths'] = self.ndepths
            data['nStreams'] = self.nstreams
            data['wavelengths'] = self.wavelengths
            data['depths'] = self.depths[:, np.newaxis]
            data['runvar'] = self.runvar
            data['modelversion'] = self.modelversion
            if self.has_cosine:
                data['up'] = self.updata
                data['down'] = self.downdata
            if self.has_scalar:
                data['scalar_up'] = self.scalar_up
                data['scalar_down'] = self.scalar_down
            if self.has_direct:
                data['direct_up'] = self.direct_up
                data['direct_down'] = self.direct_down
            if self.has_diffuse:
                data['diffuse_up'] = self.diffuse_up
                data['diffuse_down'] = self.diffuse_down
            if self.has_sine:
                data['sine_up'] = self.sine_up
                data['sine_down'] = self.sine_down
            if self.has_iops:
                data['iops'] = self.iops
            if self.has_radiance:
                data['polarangles'] = self.polarangles
                data['azimuthangles'] = self.azimuthangles
                data['radiance'] = self.radiance
                data['README_radiance'] = '5D matrix of radiances. '\
                  'The dimensions of the matrix are (depth) x (wavelength) '\
                  'x (polar angle) x (azimuth angle) x (run number).'
            if (self.has_cosine or self.has_sine or self.has_diffuse or
                    self.has_direct or self.has_scalar):
                data['README_irradiance'] = 'Irradiance matrices are 3D. '\
                  'The dimensions of the matrices are (depth) x (wavelength) '\
                  'x (run number).'

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
            depths.unit = 'metre'
            depths.reference = 'Top of Atmosphere'
            depths.description = 'Distance from top of atmosphere'
            wavelengths[:] = self.wavelengths
            wavelengths.unit = 'nanometre'

            if isinstance(self.runvar, str):
                multirun[:] = np.arange(self.updata.shape[2])
            else:
                multirun[:] = self.runvar

            if self.has_cosine:
                cos_up = \
                    outfile.createVariable('cosine_upward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                cos_up[:, :, :] = self.updata
                cos_down = \
                    outfile.createVariable('cosine_downward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                cos_down[:, :, :] = self.downdata
                cos_up.unit = 'Watt per square metre'
                cos_down.unit = 'Watt per square metre'
                cos_up.description = 'Upward total planar irradiance'
                cos_down.description = 'Downward total planar irradiance'
            if self.has_direct:
                dir_up = \
                    outfile.createVariable('direct_upward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                dir_up[:, :, :] = self.direct_up
                dir_down = \
                    outfile.createVariable('direct_downward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                dir_down[:, :, :] = self.direct_down
                dir_up.unit = 'Watt per square metre'
                dir_down.unit = 'Watt per square metre'
                dir_up.description = 'Upward planar irradiance, direct beam'
                dir_down.description = 'Downward planar irradiance, '\
                                       'direct beam'
            if self.has_diffuse:
                diff_up = \
                    outfile.createVariable('diffuse_upward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                diff_up[:, :, :] = self.diffuse_up
                diff_down = \
                    outfile.createVariable('diffuse_downward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                diff_down[:, :, :] = self.diffuse_down
                diff_up.unit = 'Watt per square metre'
                diff_down.unit = 'Watt per square metre'
                diff_up.description = 'Upward planar irradiance, diffuse field'
                diff_down.description = 'Downward planar irradiance, '\
                                        'diffuse field'
            if self.has_sine:
                sine_up = \
                    outfile.createVariable('sine_upward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                sine_up[:, :, :] = self.sine_up
                sine_down = \
                    outfile.createVariable('sine_downward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                sine_down[:, :, :] = self.sine_down
                sine_up.unit = 'Watt per square metre'
                sine_down.unit = 'Watt per square metre'
                sine_up.description = 'Upward total sine-weighted irradiance'
                sine_down.description = 'Downward total sine-weighted '\
                                        'irradiance'
            if self.has_scalar:
                sclr_up = \
                    outfile.createVariable('scalar_upward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                sclr_up[:, :, :] = self.scalar_up
                sclr_down = \
                    outfile.createVariable('scalar_downward', 'float',
                                           ('depth', 'wavelength', 'multirun'))
                sclr_down[:, :, :] = self.scalar_down
                sclr_up.unit = 'Watt per square metre'
                sclr_down.unit = 'Watt per square metre'
                sclr_up.description = 'Upward total scalar irradiance'
                sclr_down.description = 'Downward total scalar irradiance'

            outfile.close()

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

    def gauss_smooth(self, sigma=5, inplace=False):
        '''Smooth data with a Gaussian filter.
        '''

        if inplace:
            if self.has_cosine:
                self.downdata = gaussf(self.downdata, sigma=sigma, axis=1)
                self.updata = gaussf(self.updata, sigma=sigma, axis=1)

            if self.has_direct:
                self.direct_down = gaussf(self.direct_down,
                                          sigma=sigma, axis=1)
                self.direct_up = gaussf(self.direct_up, sigma=sigma, axis=1)

            if self.has_diffuse:
                self.diffuse_down = gaussf(self.diffuse_down,
                                           sigma=sigma, axis=1)
                self.diffuse_up = gaussf(self.diffuse_up, sigma=sigma, axis=1)

            if self.has_scalar:
                self.scalar_down = gaussf(self.scalar_down,
                                          sigma=sigma, axis=1)
                self.scalar_up = gaussf(self.scalar_up, sigma=sigma, axis=1)

            if self.has_sine:
                self.sine_down = gaussf(self.sine_down, sigma=sigma, axis=1)
                self.sine_up = gaussf(self.sine_up, sigma=sigma, axis=1)

        else:
            modeldata = copy.deepcopy(self)
            if modeldata.has_cosine:
                modeldata.downdata = gaussf(modeldata.downdata,
                                            sigma=sigma, axis=1)
                modeldata.updata = gaussf(modeldata.updata,
                                          sigma=sigma, axis=1)

            if modeldata.has_direct:
                modeldata.direct_down = gaussf(modeldata.direct_down,
                                               sigma=sigma, axis=1)
                modeldata.direct_up = gaussf(modeldata.direct_up,
                                             sigma=sigma, axis=1)

            if modeldata.has_diffuse:
                modeldata.diffuse_down = gaussf(modeldata.diffuse_down,
                                                sigma=sigma, axis=1)
                modeldata.diffuse_up = gaussf(modeldata.diffuse_up,
                                              sigma=sigma, axis=1)

            if modeldata.has_scalar:
                modeldata.scalar_down = gaussf(modeldata.scalar_down,
                                               sigma=sigma, axis=1)
                modeldata.scalar_up = gaussf(modeldata.scalar_up,
                                             sigma=sigma, axis=1)

            if modeldata.has_sine:
                modeldata.sine_down = gaussf(modeldata.sine_down,
                                             sigma=sigma, axis=1)
                modeldata.sine_up = gaussf(modeldata.sine_up,
                                           sigma=sigma, axis=1)
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

    def plot(self, profile=False, run=0, direction='down',
             ax=None, kind='cosine'):
        '''Plots irradiance from one of the runs, either as a vertical profile
        or as spectra. Either upwelling or downwelling irradiance.'''
        if ax is None:
            _, ax = plt.subplots()

        if direction == 'up':
            if kind == 'cosine':
                data = self.updata[:, :, run]
            elif kind == 'sine':
                data = self.sine_up[:, :, run]
            elif kind == 'scalar':
                data = self.scalar_up[:, :, run]
            elif kind == 'direct':
                data = self.direct_up[:, :, run]
            elif kind == 'diffuse':
                data = self.diffuse_up[:, :, run]
        elif direction == 'down':
            if kind == 'cosine':
                data = self.downdata[:, :, run]
            elif kind == 'sine':
                data = self.sine_down[:, :, run]
            elif kind == 'scalar':
                data = self.scalar_down[:, :, run]
            elif kind == 'direct':
                data = self.direct_down[:, :, run]
            elif kind == 'diffuse':
                data = self.diffuse_down[:, :, run]

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

    def plot_iops(self, run=0, wl=None, wl_index=None, z=None, z_index=None):
        '''
        z corresponds to layer depth.
        '''
        if not self.has_iops:
            raise AttributeError(('IOPs not available, you need iops=True'))
        if wl is not None:
            wl_index = np.argmin(np.abs(self.wavelengths-wl))
        if (wl is None) and (wl_index is None):
            wl_index = 0
        if z is not None:
            z_index = np.searchsorted(self.iops['layer_depths'][run], z)
        if (z is None) and (z_index is None):
            z_index = 0
        _wl = self.wavelengths
        abs_coeff = self.iops['absorption_coefficients'][run, z_index, :]
        scat_coeff_scaled = \
            self.iops['scattering_coefficients'][run, z_index, :]
        _sf = self.iops['scattering_scaling_factors'][run, z_index, :]
        if isinstance(self.iops['phase_moments'], list):
            g_scaled = self.iops['phase_moments'][run][z_index, :, 1]
        else:
            g_scaled = self.iops['phase_moments'][run, z_index, :, 1]

        scat_coeff_unscaled = scat_coeff_scaled/_sf
        g_unscaled = (g_scaled-1)*_sf + 1

        ssa_scaled = scat_coeff_scaled / (abs_coeff + scat_coeff_scaled)
        ssa_unscaled = scat_coeff_unscaled / (abs_coeff + scat_coeff_unscaled)

        fig, ((a, b), (g, ssa)) = plt.subplots(
            ncols=2, nrows=2, sharex=True, figsize=(10, 7)
            )

        a.plot(_wl, abs_coeff)
        a.set_ylabel(r'Absorption coefficient [$\mathrm{m}^{-1}$]')

        b.plot(_wl, scat_coeff_scaled, label='Scaled (left ordinate)')
        b.set_ylabel(r'Scaled $b$ [$\mathrm{m}^{-1}$]')

        bu = b.twinx()
        bu.plot(_wl, scat_coeff_unscaled,
                linestyle='--', label='Unscaled (right ordinate)')
        bu.set_ylabel(r'Unscaled $b$ [$\mathrm{m}^{-1}$]')
        bu.get_yaxis().get_major_formatter().set_useOffset(False)

        _h1, _l1 = b.get_legend_handles_labels()
        _h2, _l2 = bu.get_legend_handles_labels()
        bu.legend(_h1+_h2, _l1+_l2, frameon=False,
                  title='Scattering coefficient')

        g.plot(_wl, g_scaled, label='Scaled $g$')
        g.plot(_wl, g_unscaled, label='Unscaled $g$')
        g.plot(_wl, _sf, label='Scattering scaling factor')
        g.set_xlabel('Wavelength [nm]')
        g.legend(frameon=False)

        ssa.plot(_wl, ssa_scaled, label='Scaled (left ordinate)')
        ssa.set_ylabel('Scaled SSA')
        ssa.set_xlabel('Wavelength [nm]')

        ssau = ssa.twinx()
        ssau.plot(_wl, ssa_unscaled,
                  linestyle='--', label='Unscaled (right ordinate)')
        ssau.set_ylabel('Unscaled SSA')
        ssau.get_yaxis().get_major_formatter().set_useOffset(False)

        _h1, _l1 = ssa.get_legend_handles_labels()
        _h2, _l2 = ssau.get_legend_handles_labels()
        ssau.legend(_h1+_h2, _l1+_l2, frameon=False,
                    title='Single scattering albedo')

        fig.suptitle(
            r'Run {0}, layer {1} (bottom at {2}), $\lambda = {3}$ nm'.format(
                run, z_index, self.iops['layer_depths'][run, z_index],
                self.wavelengths[wl_index]))

        fig.tight_layout()
        for axis in [a, b, bu]:
            p = axis.get_position()
            axis.set_position([p.x0, p.y0-0.02, p.width, p.height*0.95])
        for axis in [g, ssa, ssau]:
            p = axis.get_position()
            axis.set_position([p.x0, p.y0, p.width, p.height*0.95])

        return ((a, b), (g, ssa))

    def plot_rad_contour(self, run=0, wl=None, wl_index=None,
                         z=None, z_index=None):
        '''
        z corresponds to detector depth.
        '''
        if not self.has_radiance:
            raise AttributeError(
                ('Radiance not available, you need radiance=True'))
        if wl is not None:
            wl_index = np.argmin(np.abs(self.wavelengths-wl))
        if (wl is None) and (wl_index is None):
            wl_index = 0
        if z is not None:
            z_index = np.argmin(np.abs(self.depths-z))
        if (z is None) and (z_index is None):
            z_index = 0

        pol = self.polarangles
        if self.azimuthangles.min() >= 0 and self.azimuthangles.max() <= 180:
            az = np.hstack((-self.azimuthangles[::-1], self.azimuthangles))
            rad = np.hstack((
                self.radiance[z_index, wl_index, :, ::-1, run],
                self.radiance[z_index, wl_index, :, :, run]
                ))
        else:
            az = self.azimuthangles
            rad = self.radiance[z_index. wl_index, :, :, run]

        n_ind = np.where(pol >= 90)[0]
        z_ind = np.where(pol <= 90)[0]
        n_rad = rad[n_ind, :]
        z_rad = rad[z_ind, :]
        fig, (nadir, zenith) = plt.subplots(
            ncols=2,
            figsize=(10, 5),
            subplot_kw=dict(projection='polar')
            )

        _r1 = nadir.contourf(np.radians(az), np.abs(pol[n_ind]-180), n_rad)
        _r2 = zenith.contourf(np.radians(az), pol[z_ind], z_rad)
        nadir.set_title('Downward radiance')
        zenith.set_title('Upward radiance')

        fig.suptitle(r'Run {0}, $\lambda = {1}$ nm, $z={2:.3f}$ m'.format(
            run, self.wavelengths[wl_index], self.depths[z_index]))

        _c1 = fig.colorbar(_r1, ax=nadir, orientation='horizontal')
        _c2 = fig.colorbar(_r2, ax=zenith, orientation='horizontal')
        for c in [_c1, _c2]:
            c.set_label(
                r'$\mathrm{W}\,\mathrm{m}^{-2}\,'
                r'\mathrm{nm}^{-1}\,\mathrm{sr}^{-1}$'
            )

        fig.tight_layout()
        for axis in [nadir, zenith]:
            p = axis.get_position()
            axis.set_position([p.x0, p.y0-0.05, p.width, p.height])
            axis.set_yticks(np.arange(0, 91, 30))

        plt.show()
        return (nadir, zenith)
