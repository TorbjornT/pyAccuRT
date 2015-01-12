import string
import subprocess
import pyAccuRT.pyAccuRead as PR
from pyAccuRT.__ATconfigs__ import templates
from pyAccuRT.__ATdefaults__ import defaults

class AccuRun(object):
    '''Setting up and running AccuRT.'''



    def __init__(self,expname,basefolder='.'):

        self._expname = expname
        self._basefolder = basefolder
        self.materials = ['main','ice']
#        self.defaults = defaults


    def setupdefaults(self,material):
        __fid = open(material + 'defaults.json','r')
        temp = json.load(__fid)
        __fid.close()
        return temp
        
    def reset(self,material):
        '''Reset config files to default.'''
        if material == 'all':
            for mat in self.materials:
                self.defaults[mat] = setupdefaults(mat)
        else:
            self.defaults[material] = setupdefaults(material)

    def run(self):
        '''Run model.'''
        _modelcall = ['AccuRT', self._basefolder + self._expname]
        subprocess.call(_modelcall)
