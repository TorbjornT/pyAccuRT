'''
Class for doing stuff with output from AccuRT.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

class PyAccu(object):
    '''Notes to self:
     - Input: folder
     - read diffuse/direct/both(?)
     - self: data-blob
     - methods:
       - plot profiles
       - calculate transmission etc. between given layers
         return values.
         - integrated or lambda dependent
         - plot if desired
       - write netCDF/HDF5
       - return numpy-array
       - read e.g. solar_zenith_angle.txt
         - have data as pandas dataframe? use sza as index?
         - (kanskje like greit aa droppe det?)
     - assume top level folder (where main config file is)
       - default name of outputfolder, be able to specify other name

     Example:
     >>> a = pyAccu('atmOcean')
     >>> a.plot_transmission(layers=(2,4))
     >>> transm = a.transmission(layers=(2,4),integrate=True)
     >>> a.plot()
     >>> a.plot(profile=True)
     >>> b = pyAccu(expname='stuff',direct=True)

     '''

    def __init__(self,expname,basefolder='./',mode='diffuse',
                 runvarfile=None):
        '''
        expname: name of main config file.
        basefolder: where main config file is.
        mode: 'diffuse' (default) or 'direct'
        '''

        outputfolder = basefolder + exname + 'Output/'

        if mode == 'diffuse':
            upfile = 'cosine_irradiance_upward.txt'
            downfile = 'cosine_irradiance_downward.txt'
        elif mode == 'direct':
            upfile = 'cosine_irradiance_direct_upward.txt'
            downfile = 'cosine_irradiance_direct_downward.txt'

        up = basefolder + outputfolder + upfile
        down = basefolder + outputfolder + downfile
        
        self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
            self.depths, self.wavelengths, self.updata = \
            self.readirradiance(up)
        *_, self.downdata  = \
            self.readirradiance(down)


        if isinstance(runvarfile,str):
            try:
                self.runvar = np.loadtxt(basefolder + runvarfile)
            except FileNotFoundError:
                print('{0} not a valid filename'.format(runvarfile))
                self.runvar = runvarfile
        else:
            try:
                iterator = iter(runvarfile)
            except TypeError:
                self.runvar = 'No multi-run info provided'
            else:
                self.runvar = np.array(runvarfile)

        with open(basefolder + outputfolder + 'version.txt','r') as ver:
            self.modelversion = ver.readline()[:-1]


        


    def readirradiance(self,filename):
        '''Read output textfiles from AccuRT model.
        Return dict with data.'''


        with open(filename,'r') as f:

            # read number of runsm streams, depths, wavelengths
            # and lists of detector depths, wavelengths
            nruns = int(f.readline()) 
            nstreams = int(f.readline())
            ndepths, nwavelengths = [int(j) for j in f.readline().split()]
            depths = [float(j) for j in f.readline().split()]
            wavelengths = [float(j) for j in f.readline().split()]

            # initiate array for irradiances
            irradiances = np.empty((ndepths,nwavelengths,nruns))

            # read values for first run
            for j in range(ndepths):
                irradiances[j,:,0] = \
                    [float(n) for n in f.readline().split()]

            # read values for rest of runs
            for i in range(1,nruns):
                #skip lines with nstreams, ndepths, etc.
                for k in range(4):
                    next(f)
                # read values
                for j in range(ndepths):
                    irradiances[j,:,i] = \
                        [float(n) for n in f.readline().split()]


        return nruns, nstreams, ndepths, nwavelengths, depths, wavelengths, irradiances



    def writefile(self,filename,output='matlab'):
        '''output is 'matlab'.
        Planned: HDF5 and NetCDF.'''


        if output == 'matlab':
            sio.savemat('{0}.mat'.format(filename),
                        dict(up=self.updata,
                             down=self.downdata,
                             nRuns=self.nruns,
                             nWavelengths=self.nwavelengths,
                             nDepths=self.ndepths,
                             nStreams=self.nstreams,
                             wavelengths=self.wavelengths,
                             depths=self.depths,
                             runvar=self.runvar,
                             modelversion=self.modelversion))

        elif output == 'netcdf':

            f = sio.netcdf_file(filename + '.nc','w')
            f.history = 'Output from AccuRT model, ' + self.modelversion


            f.createDimension('depth', self.ndepths)
            f.createDimension('wavelength', self.nwavelengths)
            f.createDimension('multirun',self.nruns)

            depths = f.createVariable('depths','float32', ('depth',))
            wavelengths = f.createVariable('wavelength','int32',('wavelength',))
            multirun = f.createVariable('runvar','float',('multirun',))

            depths[:] = self.depths
            depths.unit = 'm'
            depths.reference = 'Top of Atmosphere'
            wavelengths[:] = self.wavelengths
            if isinstance(self.runvar,str):
                multirun[:] = np.arange(self.updata.shape[2])
            else:
                multirun[:] = self.runvar

            upward_irradiance = f.createVariable('upward_irradiance','float',
                                          ('depth','wavelength','multirun'))

            upward_irradiance[:,:,:] = self.updata
            downward_irradiance = \
                f.createVariable('downward_irradiance','float',
                                 ('depth','wavelength','multirun'))

            downward_irradiance[:,:,:] = self.downdata

            f.close()


    def plot(self,profile=False,run=1,direction='down'):
        if direction=='up':
            data = self.updata[:,:,run-1]
        elif direction == 'down':
            data = self.downdata[:,:,run-1]


        if profile:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data,self.depths)
            ax.set_ylabel('Depth below TOA [m]')
            ax.set_xlabel('Irradiance [W/m2]')
            ax.invert_yaxis()
            ax.legend([str(l) for l in self.wavelengths],
                      loc='best',
                      title='Wavelength [nm]')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.wavelengths,data.T)
            ax.set_xlabel('Wavelength [nm]')
            ax.set_ylabel('Irradiance [W/m2]')
            ax.legend([str(l) for l in self.depths],
                      loc='best',
                      title='Depth below TOA [m]')
            
        
        return fig, ax



    def albedo(self, layer, integrated=False):
        '''Calculate albedo, return array.'''

        if layer == 'all':
            if integrated:
                a = np.trapz(self.updata / self.downdata,
                             x=self.wavelengths,axis=1)
            else:
                a = self.updata / self.downdata

        else:
            incident = self.downdata[layer,:,:]
            reflected = self.updata[layer,:,:]

            if integrated:
                a = np.trapz( reflected / incident, x=self.wavelengths, axis=0)
            else:
                a = reflected / incident

        return a

    def transmitted(self, layers, integrated=False):
        '''Calculate transmittance between levels given by 2-tuple layers.'''

        incident = self.downdata[layers[0],:,:]
        outgoing = self.downdata[layers[1],:,:]

        t = outgoing / incident

        if integrated:
            t = np.trapz(t,x=self.wavelengths,axis=0)

        return t

    def absorbed(self, layers, integrated=False):
        
        incoming = self.downdata[layers[0],:,:]
        outgoing = self.downdata[layers[1],:,:]
        reflected = self.updata[layers[1],:,:]

        a = (incoming - outgoing - reflected) / incoming

        if integrated:
            a = np.trapz(a,x=self.wavelengths,axis=0)

        return a
