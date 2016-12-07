'''
Class for doing stuff with output from AccuRT.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.ndimage.filters import gaussian_filter1d as gaussf

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

    direct -- Boolean. If True, read in the direct irradiance in addition
        to the diffuse. Default False.

    runvarfile -- Filename or list-like structure holding indices for repeated
        runs. Default None.

    scalar -- Boolean. If True, read in scalar irradiance in addition to
        diffuse irradiance. Default False.

    radiance -- Boolean. If True, read radiance. Default False.

    iops -- Boolean. If True, read in iops-file into a dict. Default False.


     Example:
     >>> a = pyAccu('atmOcean')
     >>> transm = a.transmittance(layers=(2,4),integrated=True)
     >>> a.plot()
     >>> a.plot(profile=True)
     '''

    def __init__(self,expname,basefolder='./',cosine=True,
                 diffuse=False,direct=False,
                 runvarfile=None, scalar=False,iops=False,
                 radiance=False,sine=False,material_profile=False):
        '''See PyAccu for description of arguments.'''

        self.has_cosine = False
        self.has_diffuse = False
        self.has_direct = False
        self.has_scalar = False
        self.has_iops = False

        outputfolder =  expname + 'Output'

        fn_fmt = '{0}_irradiance_{1}_{2}ward.txt'

        if cosine:
            self.has_cosine = True
            cos_u_file = fn_fmt.format('cosine','total','up')
            cos_d_file = fn_fmt.format('cosine','total','down')
        
            cos_u_path = os.path.join(basefolder, outputfolder, cos_u_file)
            cos_d_path = os.path.join(basefolder, outputfolder, cos_d_file)
        
            self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths, self.updata = \
                self.readirradiance(cos_u_path)
            *_, self.downdata = self.readirradiance(cos_d_path)

        if diffuse:
            self.has_diffuse = True
            diff_u_file = fn_fmt.format('cosine','diffuse','up')
            diff_d_file = fn_fmt.format('cosine','diffuse','down')
        
            diff_u_path = os.path.join(basefolder, outputfolder, diff_u_file)
            diff_d_path = os.path.join(basefolder, outputfolder, diff_d_file)

            *_, self.diffuse_down = self.readirradiance(diff_d_path)
            *_, self.diffuse_up = self.readirradiance(diff_u_path)

            if not hasattr(self,'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths = _


        if direct:
            self.has_direct = True
            dir_u_file = fn_fmt.format('cosine','direct','up')
            dir_d_file = fn_fmt.format('cosine','direct','down')
        
            dir_u_path = os.path.join(basefolder, outputfolder, dir_u_file)
            dir_d_path = os.path.join(basefolder, outputfolder, dir_d_file)
            *_, self.direct_down = self.readirradiance(dir_d_path)
            *_, self.direct_up = self.readirradiance(dir_u_path)

            if not hasattr(self,'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths = _

        if scalar:
            self.has_scalar = True
            sclr_u_file = fn_fmt.format('scalar','total','up')
            sclr_d_file = fn_fmt.format('scalar','total','down')

            sclr_u_path = os.path.join(basefolder, outputfolder, sclr_u_file)
            sclr_d_path = os.path.join(basefolder, outputfolder, sclr_d_file)

            *_, self.scalar_down = self.readirradiance(sclr_d_path)
            *_, self.scalar_up = self.readirradiance(sclr_u_path)

            if not hasattr(self,'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths = _
        
        if sine:
            sine_u_file = fn_fmt.format('sine','total','up')
            sine_d_file = fn_fmt.format('sine','total','down')

            sine_u_path = os.path.join(basefolder, outputfolder, sine_u_file)
            sine_d_path = os.path.join(basefolder, outputfolder, sine_d_file)

            *_, self.sine_down = self.readirradiance(sine_d_path)
            *_, self.sine_up = self.readirradiance(sine_u_path)

            if not hasattr(self,'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths = _

        if radiance:
            outputfolder =  expname + 'Output'
            filename = os.path.join(basefolder,outputfolder,'radiance.txt')
            self.radiance, self.polarangles,self.azimuthangles, *_ = \
                                        self.readradiance(filename)

            if not hasattr(self,'nruns'):
                self.nruns, self.nstreams, self.ndepths, self.nwavelengths, \
                self.depths, self.wavelengths = _


        if iops:
            self.has_iops = True
            iops_path = os.path.join(basefolder, outputfolder, 'iops.txt')
            self.iops = self.readiops(iops_path)

        if material_profile:
            mp_path = os.path.join(basefolder,outputfolder,'material_profile.txt')
            self.material_profile = self.read_material_profile(mp_path)


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

    def readradiance(self,filename):
        '''Read output radiance.txt from AccuRT.
        Returns number of runs, streams, detector depths and wavelengths,
        and numpy arrays of dephts, wavelengths, polar and azimuth angles
        and radiance.

        Dimensions of radiance array is
        (depth) x (wavelength) x (polar angle) x (azimuth angle) x (run number)'''

        
        with open(filename, 'r') as f:
            nruns = int(f.readline()) 
            nstreams = int(f.readline())
            ndepths, nwavelengths, npolarangles, nazimuthangles = [int(j) for j in f.readline().split()]
            depths = np.array([float(j) for j in f.readline().split()])
            wavelengths = np.array([float(j) for j in f.readline().split()])
            polarangles = np.array([float(j) for j in f.readline().split()])
            azimuthangles = np.array([float(j) for j in f.readline().split()])

            radiances = np.empty((ndepths,nwavelengths,npolarangles,nazimuthangles,nruns))

            rad = np.array([float(j) for j in f.readline().split()])
            radiances[:,:,:,:,0] = rad.reshape(ndepths,nwavelengths,npolarangles,nazimuthangles)

            for i in range(1,nruns):
                #skip lines with nstreams, ndepths, etc.
                for k in range(6):
                    next(f)
                rad = np.array([float(j) for j in f.readline().split()])
                # read values
                radiances[:,:,:,:,i] = rad.reshape(ndepths,nwavelengths,npolarangles,nazimuthangles)

        return radiances, polarangles, azimuthangles, nruns, nstreams, ndepths, nwavelengths, depths, wavelengths

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
                        

    def read_material_profile(self,filename):
        mp = []
        with open(filename) as MP:
            while True:
                line = MP.readline()
                if line.startswith('runNo'):
                    run = int(line.split()[1])
                    break
            endoffile = False
            while True:
                if endoffile:
                    break
                layer = -1
                mp.append([])
                while True:
                    line = MP.readline()
                    print(line)
                    if line.startswith('=') or \
                       line.startswith('-') or \
                       line.startswith('~'):
                        pass
                    elif len(line) == 0:
                        endoffile = True
                        break
                    elif line.startswith('runNo'):
                        run = int(line.split()[1])
                        break
                    elif line.startswith('Layer '):
                        layer += 1
                        mp[run].append(dict())
                        pass
                    elif line.startswith('Bottom depth'):
                        z = float(line.split()[-2])
                        mp[run][layer]['bottomdepth'] = z
                    else:
                        matname = line.replace(' ','').strip()
                        material = dict()
                        conc,conctype = MP.readline().split()
                        tau,ssa,g = [float(x) for x in MP.readline().split()]
                        atau,btau = [float(x) for x in MP.readline().split()]
                        a,b = [float(x) for x in MP.readline().split()]
                        df = float(MP.readline())

                        material['concentration'] = float(conc)
                        material['concentrationtype'] = conctype[1:-1]
                        material['opticaldepth'] = tau
                        material['singlescatteringalbedo'] = ssa
                        material['asymmetryfactor'] = g
                        material['absorptionoptdep'] = atau
                        material['scatteringoptdep'] = btau
                        material['absorption'] = a
                        material['scattering'] = b
                        material['deltafitscalingfactor'] = df

                        mp[run][layer][matname] = material

        return mp



    def writefile(self,filename,output=None):
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
                        nWavelengths=self.nwavelengths,
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

            
            sio.savemat('{0}.mat'.format(filename),data)

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

    def transmittance(self, layers, integrated=False, wlrange=None):
        '''Calculate transmittance between levels given by 2-tuple layers.'''

        incident = self.downdata[layers[0],:,:]
        outgoing = self.downdata[layers[1],:,:]

        if integrated:
            if wlrange is None:
                t = np.trapz(outgoing,x=self.wavelengths,axis=0)/\
                    np.trapz(incident,x=self.wavelengths,axis=0)
            else:
                J = np.abs(self.wavelengths-wlrange[0]).argmin()
                K = np.abs(self.wavelengths-wlrange[1]).argmin()
                if (self.wavelengths[J] > wlrange[0]) and (J > 0):
                    J -= 1
                if (self.wavelengths[K] < wlrange[1]):
                    K += 2
                else:
                    K += 1

                t = np.trapz(outgoing[J:K,:],x=self.wavelengths[J:K],axis=0)/\
                    np.trapz(incident[J:K,:],x=self.wavelengths[J:K],axis=0)
        else:
            t = outgoing / incident

        return t



    def gauss_smooth(self,n=5,inplace=False):
        '''Smooth data with a Gaussian filter.
        '''

        if inplace:
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
        else:
            self.downdata_sm = gaussf(self.downdata,sigma=n,axis=1)
            self.updata_sm = gaussf(self.updata,sigma=n,axis=1)

            try:
                self.direct_down_sm = gaussf(self.direct_down,sigma=n,axis=1)
                self.direct_up_sm = gaussf(self.direct_up,sigma=n,axis=1)
            except:
                pass
            try:
                self.scalar_down_sm = gaussf(self.scalar_down,sigma=n,axis=1)
                self.scalar_up_sm = gaussf(self.scalar_up,sigma=n,axis=1)
            except:
                pass



            

    def calc_heatingrate(self):
        if not (self.has_scalar and self.has_iops):
            raise AttributeError('Scalar irradiance and IOPs not available, you need scalar=True,iops=True')
        Eabs = np.empty_like(self.scalar_down)
        for k in range(Eabs.shape[2]):
            if Eabs.shape[2] == 1:
                layerdepths = self.iops['LayerDepths']
                abscoeff = self.iops['absorptionCoefficients']
            else:
                layerdepths = self.iops['LayerDepths'][k]
                abscoeff = self.iops['absorptionCoefficients'][k]
            layerind = [np.where(layerdepths>=dd)[0][0] for dd in self.depths]
            for i,j in enumerate(layerind):
                Eabs[i,:,k] = abscoeff[j] * (self.scalar_down[i,:,k] + self.scalar_up[i,:,k])
        Eabs[Eabs<0] = 0

        return Eabs

    def diffuse_attenuation(self,integrated=False):
        dz = np.diff(self.depths)
        if integrated:
            dint = np.trapz(self.downdata,axis=1,x=self.wavelengths)
            kd = 1/dz * np.log(dint[:-1,:]/dint[1:,:])
        else:
            kd = 1/dz * np.log(self.downdata[:-1,:,:]/self.downdata[1:,:,:])
        return kd