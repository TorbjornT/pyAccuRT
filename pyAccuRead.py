'''
Class for doing stuff with output from AccuRT.
'''

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
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
     >>> a = pyAccu('./atmOceanOutput')
     >>> a.plot_transmission(layers=(2,4))
     >>> transm = a.transmission(layers=(2,4),integrate=True)
     >>> a.plot()
     >>> a.plot(profile=True)
     >>> b = pyAccu('./atmoOut',direct=True)

     '''

    def __init__(self,basefolder='',outputfolder=None,mode='diffuse'):
        '''basefolder: where main config file is.
        outputfolder: by default, folder with 'Output' at the end of the name
        mode: 'diffuse' (default) or 'direct'
        '''

        if not outputfolder:
            outputfolder = glob('*Output/')[0]

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
            self.runvar = np.loadtxt(basefolder + runvarfile)
        else:
            try:
                iterator = iter(runvarfile)
            except TypeError:
                self.runvar = 'No multi-run info provided'
            else:
                self.runvar = np.array(runvarfile)


        


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


    def createarray(self,direction='down'):
        '''Return numpy-array with all irradiances. Size of array will be
        nDepths x nWavelengths x nRuns.'''

        if direction=='up':
            data = self.updata
        elif direction == 'down':
            data = self.downdata

        irrarray = np.ones((self.nDepths,self.nWavelengths,self.nRuns))

        for i in range(self.nRuns):
            irrarray[:,:,i] = data[i]['irradiances']

        return irrarray


    def writefile(self,filename='output',type='matlab'):
        '''type is 'matlab'.
        Planned: HDF5 and NetCDF.'''

        if type == 'matlab':
            up = {}
            down = {}

            up['allirradiances'] = self.createarray('up')
            down['allirradiances'] = self.createarray('down')

            try:
                up['runvar'] = self.runvar
                down['runvar'] = self.runvar
            except:
                print('No multirun-data found.')
            

            for i in range(self.nRuns):
                up['R{0}'.format(i)] = self.updata[i]
                down['R{0}'.format(i)] = self.downdata[i]

            sio.savemat('{0}.mat'.format(filename), dict(up=up,down=down))


    def plot(self,profile=False,run=1,direction='down'):
        if direction=='up':
            data = self.updata[run-1]['irradiances']
        elif direction == 'down':
            data = self.downdata[run-1]['irradiances']


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
            ax.set_xlabel('Wavelength [nm]]')
            ax.set_ylabel('Irradiance [W/m2]')
            ax.legend([str(l) for l in self.depths],
                      loc='best',
                      title='Depth below TOA [m]')
            
        
        return fig, ax



    def albedo(self, layer, integrated=False):
        '''Calculate albedo, return array.'''

        if layer == 'all':
            incident = self.createarray('down')
            reflected = self.createarray('up')
        else:
            incident = self.createarray('down')[layer,:,:]
            reflected = self.createarray('up')[layer,:,:]

        a = reflected / incident
        if integrated:
            a = np.trapz(a,x=self.wavelengths,axis=1)

        return a

    def transmitted(self, layers, integrated=False):
        '''Calculate transmittance between levels given by 2-tuple layers.'''

        dat = self.createarray('down')
        incident = dat[layers[0],:,:]
        outgoing = dat[layers[1],:,:]

        t = outgoing / incident

        if integrated:
            t = np.trapz(t,x=self.wavelengths,axis=1)

        return t

    def absorbed(self, layers, integrated=False):
        
        down = self.createarray('down')
        incoming = down[layers[0],:,:]
        outgoing = down[layers[1],:,:]
        reflected = self.createarray('up')[layers[1],:,:]

        a = (incoming - outgoing - reflected) / incoming

        if integrated:
            a = np.trapz(a,x=self.wavelengths,axis=1)

        return a
