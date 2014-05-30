'''
Class for doing stuff with output from AccuRT.
'''

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.io as sio

class pyAccu(object):
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
        
        self.updata, self.nRuns = \
            self.__readIrradiance__(up)
        self.downdata, _  = \
            self.__readIrradiance__(down)

        self.wavelengths = self.updata[0]['Wavelengths']
        self.depths = self.updata[0]['Depths']
        self.nWavelengths = self.updata[0]['nWavelengths']
        self.nDepths = self.updata[0]['nDepths']

        if runvarfile:
            self.runvar = np.loadtxt(basefolder + runvarfile)
        


    def __readIrradiance__(self,filename):
        '''Read output textfiles from AccuRT model.
        Return dict with data.'''


        f = open(filename,'r')

        nRuns = int(f.readline())

        data = {}

        for i in range(nRuns):
            data[i] = {}
            data[i]['nStreams'] = int(f.readline())
            dep, wav = [int(j) for j in f.readline().split()]
            data[i]['nDepths'] = dep
            data[i]['nWavelengths'] = wav
            data[i]['Depths'] = [float(j) for j in f.readline().split()]
            data[i]['Wavelengths'] = [float(j) for j in f.readline().split()]
            data[i]['irradiances'] = np.empty((dep,wav))
            for j in range(dep):
                data[i]['irradiances'][j] = \
                    [float(n) for n in f.readline().split()]

        f.close()

        return data, nRuns


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
