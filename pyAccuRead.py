'''
Class for doing stuff with output from AccuRT.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

class PyAccu(object):
    '''Reads the output text files from AccuRT, and includes methods for
    calculating albedo and transmittance, and for simple plotting.

    Assumes that the number of detector depths and wavelengths
    does *not* change when using repeated runs.

    Positional argument:

    expname -- Name of the main config file.

    Keyword arguments:

    basefolder -- Folder where the main configfile is located.
        Default './'.

    direct -- Boolean. If True, read in the direct irradiance in addition
        to the diffuse. Default False.

    runvarfile -- Filename or list-like structure holding indices for repeated
        runs. Default None.

    scalar -- Boolean. If True, read in scalar irradiance in addition to
        diffuse irradiance. Default False.

    iops -- Boolean. If True, read in iops-file into a dict. Default False.


     Example:
     >>> a = pyAccu('atmOcean')
     >>> transm = a.transmittance(layers=(2,4),integrated=True)
     >>> a.plot()
     >>> a.plot(profile=True)
     '''

    def __init__(self,expname,basefolder='./',direct=False,
                 runvarfile=None, scalar=False,iops=False):
        '''See PyAccu for description of arguments.'''

        outputfolder =  expname + 'Output/'

        up_diffuse = 'cosine_irradiance_upward.txt'
        down_diffuse = 'cosine_irradiance_downward.txt'

        updiff = os.path.join(basefolder, outputfolder, up_diffuse)
        downdiff = os.path.joind(basefolder, outputfolder, down_diffuse)

        
        self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
            self.depths, self.wavelengths, self.updata = \
            self.readirradiance(updiff)
        *_, self.downdata  = \
            self.readirradiance(downdiff)

        if direct:
            up_direct = 'cosine_irradiance_direct_upward.txt'
            down_direct = 'cosine_irradiance_direct_downward.txt'
        
            updir = os.path.join(basefolder, outputfolder, up_direct)
            downdir = os.path.join(basefolder, outputfolder, down_direct)
            *_, self.downdirect  = \
                self.readirradiance(downdir)
            *_, self.updirect  = \
                self.readirradiance(updir)

        if scalar:
            supfile = 'scalar_irradiance_upward.txt'
            sdownfile = 'scalar_irradiance_downward.txt'

            *_, self.scalar_down = self.readirradiance(os.path.join(basefolder, outputfolder, sdownfile))
            *_, self.scalar_up = self.readirradiance(os.path.join(basefolder, outputfolder, supfile))



        if isinstance(runvarfile,str):
            try:
                self.runvar = np.loadtxt(os.path.join(basefolder, runvarfile))
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

        with open(os.path.join(basefolder, outputfolder, 'version.txt'),'r') as ver:
            self.modelversion = ver.readline()[:-1]

        if read_iops:
            filename = os.path.join(basefolder, outputfolder, 'iops.txt')
            self.iops = self.readiops(filename)


        


    def readirradiance(self,filename):
        '''Read output irradiance textfiles from AccuRT model.
        Returns number of runs, streams, detector depths and wavelengths,
        and numpy arrays of depths, wavelengths and irradiance'''


        with open(filename,'r') as f:

            # read number of runsm streams, depths, wavelengths
            # and lists of detector depths, wavelengths
            nruns = int(f.readline()) 
            nstreams = int(f.readline())
            ndepths, nwavelengths = [int(j) for j in f.readline().split()]
            depths = np.array([float(j) for j in f.readline().split()])
            wavelengths = np.array([float(j) for j in f.readline().split()])

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

    def readiops(self,filename):
        '''Read iops.txt, returns dict.'''


        with open(filename,'r') as f:
            nRuns = int(f.readline())
            nLayerDepths, nWavelengths, nPhaseMoments = [int(x) for x in f.readline().split()]

            totalOpticalDepth = np.empty((nLayerDepths,nWavelengths))
            absorptionCoefficients = np.empty((nLayerDepths,nWavelengths))
            scatteringCoefficients = np.empty((nLayerDepths,nWavelengths))
            scatteringScalingFactors = np.empty((nLayerDepths,nWavelengths))
            phaseMoments = np.empty((nLayerDepths,nWavelengths,nPhaseMoments))

            
        with open(filename,'r') as f:
            nRuns = int(f.readline())

            LayerDepths = []
            Wavelengths = []

            for i in range(nRuns):
                nLayerDepths, nWavelengths, nPhaseMoments = [int(x) for x in f.readline().split()]

                LayerDepths.append(np.array([float(x) for x in f.readline().split()]))
                Wavelengths.append(np.array([float(x) for x in f.readline().split()]))

                for j in range(nLayerDepths):
                    for k in range(nWavelengths):
                        d = f.readline().split()
                        totalOpticalDepth[j,k] = float(d.pop(0))
                        absorptionCoefficients[j,k] = float(d.pop(0))
                        scatteringCoefficients[j,k] = float(d.pop(0))
                        scatteringScalingFactors[j,k] = float(d.pop(0))
                        phaseMoments[j,k,:] = [float(k) for k in d]


            iops = dict(nRuns=nRuns,
                        nLayerDepths = nLayerDepths,
                        nWaveLenghts = nWavelengths,
                        nPhaseMoments = nPhaseMoments,
                        LayerDepths = LayerDepths,
                        Wavelengths = Wavelengths,
                        totalOpticalDepth = totalOpticalDepth,
                        absorptionCoefficients = absorptionCoefficients,
                        scatteringCoefficients = scatteringCoefficients,
                        scatteringScalingFactors = scatteringScalingFactors,
                        phaseMoments = phaseMoments)

            return iops
                        

                
            


    def writefile(self,filename,output='matlab'):
        '''output is 'matlab' or 'netcdf'.'''


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


    def plot(self,profile=False,run=1,direction='down',ax=None):
        '''Plots data from one of the runs, either as a vertical profile or
        as spectra. Either upwelling or downwelling irradiance.'''
        if ax is None:
            fig,ax = plt.subplots()
            
        if direction=='up':
            data = self.updata[:,:,run-1]
        elif direction == 'down':
            data = self.downdata[:,:,run-1]


        if profile:
            ax.plot(data,self.depths)
            ax.set_ylabel('Depth below TOA [m]')
            ax.set_xlabel('Irradiance [W/m2]')
            ax.invert_yaxis()
            ax.legend([str(l) for l in self.wavelengths],
                      loc='best',
                      title='Wavelength [nm]')
        else:
            ax.plot(self.wavelengths,data.T)
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
