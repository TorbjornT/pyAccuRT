'''
Class for doing stuff with output from AccuRT.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.ndimage.filters import gaussian_filter1d as gaussf

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

        self.has_direct = False
        self.has_scalar = False
        self.has_iops = False

        outputfolder =  expname + 'Output'

        diff_u_file = 'cosine_irradiance_total_upward.txt'
        diff_d_file = 'cosine_irradiance_total_downward.txt'
        
        diff_u_path = os.path.join(basefolder, outputfolder, diff_u_file)
        diff_d_path = os.path.join(basefolder, outputfolder, diff_d_file)

        
        self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
            self.depths, self.wavelengths, self.updata = \
            self.readirradiance(diff_u_path)
        *_, self.downdata = self.readirradiance(diff_d_path)

        if direct:
            self.has_direct = True
            dir_u_file = 'cosine_irradiance_direct_upward.txt'
            dir_d_file = 'cosine_irradiance_direct_downward.txt'
        
            dir_u_path = os.path.join(basefolder, outputfolder, dir_u_file)
            dir_d_path = os.path.join(basefolder, outputfolder, dir_d_file)
            *_, self.direct_down = self.readirradiance(dir_d_path)
            *_, self.direct_up = self.readirradiance(dir_u_path)

        if scalar:
            self.has_scalar = True
            sclr_u_file = 'scalar_irradiance_total_upward.txt'
            sclr_d_file = 'scalar_irradiance_total_downward.txt'

            sclr_u_path = os.path.join(basefolder, outputfolder, sclr_u_file)
            sclr_d_path = os.path.join(basefolder, outputfolder, sclr_d_file)

            *_, self.scalar_down = self.readirradiance(sclr_d_path)
            *_, self.scalar_up = self.readirradiance(sclr_u_path)


        if iops:
            self.has_iops = True
            iops_path = os.path.join(basefolder, outputfolder, 'iops.txt')
            self.iops = self.readiops(iops_path)


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



        


    def readirradiance(self,filename):
        '''Read output irradiance textfiles from AccuRT model.
        Returns number of runs, streams, detector depths and wavelengths,
        and numpy arrays of depths, wavelengths and irradiance'''


        with open(filename,'r') as f:

            # read number of runs, streams, depths, wavelengths
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
            
            totalOpticalDepth = []
            absorptionCoefficients = []
            scatteringCoefficients = []
            scatteringScalingFactors = []
            phaseMoments = []
            LayerDepths = []
            Wavelengths = []

            for i in range(nRuns):
                nLayerDepths, nWavelengths, nPhaseMoments = [int(x) for x in f.readline().split()]

                LayerDepths.append(np.array([float(x) for x in f.readline().split()]))
                Wavelengths.append(np.array([float(x) for x in f.readline().split()]))
                
                ToD = np.empty((nLayerDepths,nWavelengths))
                AC = np.empty((nLayerDepths,nWavelengths))
                SC = np.empty((nLayerDepths,nWavelengths))
                SSF = np.empty((nLayerDepths,nWavelengths))
                PM = np.empty((nLayerDepths,nWavelengths,nPhaseMoments))


                for j in range(nLayerDepths):
                    for k in range(nWavelengths):
                        d = f.readline().split()
                        ToD[j,k] = float(d.pop(0))
                        AC[j,k] = float(d.pop(0))
                        SC[j,k] = float(d.pop(0))
                        SSF[j,k] = float(d.pop(0))
                        PM[j,k,:] = np.array(d,dtype='float')
                        
                totalOpticalDepth.append(ToD.copy())
                absorptionCoefficients.append(AC.copy())
                scatteringCoefficients.append(SC.copy())
                scatteringScalingFactors.append(SSF.copy())
                phaseMoments.append(PM.copy())


            iops = dict(nRuns=nRuns,
                        LayerDepths = np.squeeze(LayerDepths),
                        Wavelengths = np.squeeze(Wavelengths),
                        totalOpticalDepth = np.squeeze(totalOpticalDepth),
                        absorptionCoefficients = np.squeeze(absorptionCoefficients),
                        scatteringCoefficients = np.squeeze(scatteringCoefficients),
                        scatteringScalingFactors = np.squeeze(scatteringScalingFactors),
                        phaseMoments = np.squeeze(phaseMoments))

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
                a = np.trapz(self.updata, x=self.wavelengths,axis=1)/\
                    np.trapz(self.downdata, x=self.wavelengths,axis=1)
            else:
                a = self.updata / self.downdata

        else:
            incident = self.downdata[layer,:,:]
            reflected = self.updata[layer,:,:]

            if integrated:
                a = np.trapz(reflected,x = self.wavelengths,axis=0)/\
                    np.trapz(incident,x=self.wavelengths,axis=0)
            else:
                a = reflected / incident

        return a

    def transmittance(self, layers, integrated=False):
        '''Calculate transmittance between levels given by 2-tuple layers.'''

        incident = self.downdata[layers[0],:,:]
        outgoing = self.downdata[layers[1],:,:]

        if integrated:
            t = np.trapz(outgoing,x=self.wavelengths,axis=0)/\
                np.trapz(incident,x=self.wavelengths,axis=0)
        else:
            t = outgoing / incident

        return t



    def gauss_smooth(self,n=5):
        '''Smooth data with a Gaussian filter.
        Todo: inplace or not'''

        self.downdata = gaussf(self.downdata,sigma=n,axis=1)
        self.updata = gaussf(self.updata,sigma=n,axis=1)

        try:
            self.direct_down = gaussf(self.direct_down,sigma=n,axis=1)
            self.direct_up = gaussf(self.direct_up,sigma=n,axis=1)
        except:
            pass
        try:
            self.scalar_down = gaussf(self.scalar_down,sigma=n,axis=1)
            self.scalar_up = gaussf(self.scalar_up,sigma=n,axis=1)
        except:
            pass


            

    def calc_heatingrate(self):
        '''Add test for scalar and iops'''
        Eabs = np.empty_like(np.squeeze(self.scalar_down))
        for k in range(Eabs.shape[2]):
            layerdepths = self.iops['LayerDepths'][k]
            abscoeff = self.iops['absorptionCoefficients'][k]
            layerind = [np.where(layerdepths>=dd)[0][0] for dd in self.depths]
            for i,j in enumerate(layerind):
                Eabs[i,:,k] = abscoeff[j] * (self.scalar_down[i,:,k] + self.scalar_up[i,:,k])
        Eabs[Eabs<0] = 0

        return Eabs
