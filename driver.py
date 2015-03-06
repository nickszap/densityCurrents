#among many resources, http://scipy-lectures.github.io/advanced/image_processing/ seems like a useful introduction to image processing
# for a list of scipy.ndimage functions: http://docs.scipy.org/doc/scipy/reference/ndimage.html

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import ndimage, stats

Cp = 1004.5; Rd = 287.04; Rd_Cp = Rd/Cp; p0 = 1.e5; grav = 9.81

def calc_potentialTemperature(t,p):
  theta = t*(p0/p)**Rd_Cp
  return theta

def calc_refFields(fieldsIn, windowSpacing=50., gridSpacing=1.):
  #Input list of 2d fields.
  #calc regional values as reference "environmental" values, where convolutions define the regional values.
  #return list of 2d reference fields
  
  windowLen = int(windowSpacing/gridSpacing)
  
  wts = np.ones((windowLen,windowLen),dtype=float)
  countVal = np.ones(fieldsIn[0].shape, dtype=int)
  nVals = ndimage.filters.convolve(countVal, wts, output=None, mode='reflect')
  
  nFields = len(fieldsIn)
  refVals = []
  for iField in xrange(nFields):
    sumVals = ndimage.filters.convolve(fieldsIn[iField], wts, output=None, mode='reflect')
    refVals.append(sumVals/nVals)
    
  return refVals

def readTimeLevel(data):
  #return t,p,q,u,v from file and convert to useful units
  
  #first model level variables
  t = data.variables['T'][:,:]+273.15 #to K\
  p = data.variables['PSFC'][:,:]*100. #to Pa
  q = data.variables['Q'][:,:] #kg/kg
  u = data.variables['U0'][:,:] #m/s
  v = data.variables['V'][:,:] #m/s
  slp = data.variables['SLP'][:,:]*100. #Pa
  
  return (t,p,q,u,v,slp)

class MidpointNormalize(Normalize):
  #taken from http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
        
def plot_field_recentered(var, norm, title=' ', showFig=False):
  
  plt.figure()  
  plt.pcolormesh(var,norm=norm,cmap=plt.cm.RdBu_r)
  plt.colorbar()
  plt.title(title)
  
  if (showFig):
    plt.show()
  else:
    return plt
  
def demo():
  #read in data --------------------
  fDir = '/data01/densityCurrents/'
  f = fDir+'201407101500.nc'
  dxGrid = 1.e3
  
  data = netCDF4.Dataset(f,'r')
  
  t,p,q,u,v,slp = readTimeLevel(data)
  ny,nx = t.shape
  
  #calculate some variables --------------
  theta = calc_potentialTemperature(t,p)
  thetav = theta*(1+.61*q)
  
  fieldsRef = calc_refFields([thetav,slp], windowSpacing=50*dxGrid, gridSpacing=dxGrid)
  thetav_ref = fieldsRef[0]; slp_ref = fieldsRef[1]
  
  buoy = grav*(thetav-thetav_ref)/thetav_ref
  #gradb_dir = np.gradient(buoy);
  #gradb = np.maximum(gradb_dir[0],gradb_dir[1])
  gradb = ndimage.morphological_gradient(buoy, size=(5,5))
  
  pPerturb = slp-slp_ref
  gradP = ndimage.morphological_gradient(pPerturb, size=(5,5))
  
  du_dxy = np.gradient(u/dxGrid)
  dv_dxy = np.gradient(v/dxGrid)
  div = du_dxy[1]+dv_dxy[0]
  gradDiv = ndimage.morphological_gradient(div, size=(5,5))
  
  #plot some stuff --------------
  if (False):
    normBuoy = MidpointNormalize(midpoint=0)
    plot_field_recentered(buoy, normBuoy, title='buoyancy', showFig=False)
    
    plt.figure()
    plt.pcolormesh(thetav)
    plt.colorbar()
    
    normGrad = MidpointNormalize(midpoint=0)
    plot_field_recentered(gradb, normGrad, title='grad(buoyancy)',showFig=False)
    
    normP = MidpointNormalize(midpoint=0)
    plot_field_recentered(pPerturb, normP, title='perturbation pressure',showFig=False)
    
    normDiv = MidpointNormalize(midpoint=0)
    plot_field_recentered(div, normDiv,title='divergence', showFig=False)
    
    norm = MidpointNormalize(midpoint=0)
    plot_field_recentered(gradDiv, norm,title='grad(divergence)', showFig=False)
    
    plt.show()
    
  #signals should persist across different variables -> correlations
  signalThresh = (grav*1./300.)*50.
  signalFlow = -buoy*pPerturb
  norm = MidpointNormalize(midpoint=0)
  plot_field_recentered(signalFlow, norm, title='-buoy*pPerturb',showFig=False)
  
  if (False):
    signalFront = gradb*gradP
    norm = MidpointNormalize(midpoint=0)
    plot_field_recentered(signalFront, norm, title='gradb*gradp',showFig=False)
    
    candidates = (signalFlow>0)*(signalFront>signalThresh)
  else:
    candidates = signalFlow>signalThresh
  
  candidates = ndimage.morphology.binary_closing(candidates, structure=np.ones((5,5),dtype=int), iterations=1)
  objs, nObjs = ndimage.measurements.label(candidates)
  
  #i think background is always 0?
  plt.figure()
  plt.pcolormesh(objs, cmap=plt.cm.flag)
  plt.colorbar()
  plt.show()
  
  
if __name__=='__main__':
  demo()



