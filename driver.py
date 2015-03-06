import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import ndimage

Cp = 1004.5; Rd = 287.04; Rd_Cp = Rd/Cp; p0 = 1.e5
grav = 9.81

def calc_potentialTemperature(t,p):
  theta = t*(p0/p)**Rd_Cp
  return theta

def calc_buoyancy(theta):
  thetaMean = np.mean(theta)
  buoy = grav*(theta-thetaMean)/thetaMean
  return buoy

def calc_buoyancy_local(theta):
  #getting a more targeted idea for the "environmental" thetaMean
  #nearer to the feature (imagine a global domain -> thetaMean(everywhere) wouldn't mean much)
  
  windowSpacing = 50. #square's length
  gridSpacing = 4.
  windowLen = int(.5*windowSpacing/gridSpacing)
  
  ny,nx = theta.shape
  thetaMean = np.empty((ny,nx),dtype=float)
  for j in xrange(ny):
    jMin = max(0,j-windowLen)
    jMax = min(ny-1,j+windowLen)
    for i in xrange(nx):
      iMin = max(0,i-windowLen)
      iMax = min(nx-1,i+windowLen)
      #print j,i,jMin,jMax, iMin,iMax
      thetaMean[j,i] = np.mean(theta[jMin:jMax+1,iMin:iMax+1])
  
  buoy = grav*(theta-thetaMean)/thetaMean
  return buoy

def calc_pertPressure_local(theta):
  
  windowSpacing = 50. #square's length
  gridSpacing = 1.
  windowLen = int(.5*windowSpacing/gridSpacing)
  
  ny,nx = theta.shape
  thetaMean = np.empty((ny,nx),dtype=float)
  for j in xrange(ny):
    jMin = max(0,j-windowLen)
    jMax = min(ny-1,j+windowLen)
    for i in xrange(nx):
      iMin = max(0,i-windowLen)
      iMax = min(nx-1,i+windowLen)
      #print j,i,jMin,jMax, iMin,iMax
      thetaMean[j,i] = np.mean(theta[jMin:jMax+1,iMin:iMax+1])
  
  buoy = theta-thetaMean
  return buoy

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
        
def plot_field_recentered(var, norm, showFig=False):
  
  plt.figure()  
  plt.pcolormesh(var,norm=norm,cmap=plt.cm.RdBu_r)
  plt.colorbar()
  
  if (showFig):
    plt.show()
  else:
    return plt

def demo_compareConvolve():
  #read in data --------------------
  fDir = '/data01/densityCurrents/'
  f = fDir+'201407101500.nc'
  
  data = netCDF4.Dataset(f,'r')
  
  t,p,q,u,v,slp = readTimeLevel(data)
  ny,nx = t.shape
  
  #calculate some variables --------------
  theta = calc_potentialTemperature(t,p)
  thetav = theta*(1+.61*q)
  
  windowLen = 51
  wts = np.ones((windowLen,windowLen),dtype=float)
  countVal = np.ones(t.shape, dtype=int)
  
  sumVals = ndimage.filters.convolve(t, wts, output=None, mode='reflect')
  nVals = ndimage.filters.convolve(countVal, wts, output=None, mode='reflect')
  meanVal = sumVals/nVals
  
  plt.figure()
  plt.pcolormesh(meanVal)
  plt.colorbar()
  plt.show()
  
def demo():
  #read in data --------------------
  fDir = '/data01/densityCurrents/'
  f = fDir+'201407101500.nc'
  
  data = netCDF4.Dataset(f,'r')
  
  t,p,q,u,v,slp = readTimeLevel(data)
  ny,nx = t.shape
  
  #calculate some variables --------------
  theta = calc_potentialTemperature(t,p)
  thetav = theta*(1+.61*q)
  
  #buoy = calc_buoyancy(thetav); print buoy
  buoy = calc_buoyancy_local(thetav);
  gradb_dir = np.gradient(buoy); #print gradb
  gradb = np.maximum(gradb_dir[0],gradb_dir[1])
  
  pPerturb = calc_pertPressure_local(slp)
  
  du_dxy = np.gradient(u)
  dv_dxy = np.gradient(v)
  div = du_dxy[1]+dv_dxy[0]
  
  #plot some stuff --------------
  if (False):
    normBuoy = MidpointNormalize(midpoint=0)
    plot_field_recentered(buoy, normBuoy, showFig=False)
    
    plt.figure()
    plt.pcolormesh(thetav)
    plt.colorbar()
    
    normGrad = MidpointNormalize(midpoint=0)
    plot_field_recentered(gradb, normGrad, showFig=False)
    
    normP = MidpointNormalize(midpoint=0)
    plot_field_recentered(pPerturb, normP, showFig=False)
    
    normDiv = MidpointNormalize(midpoint=0)
    plot_field_recentered(div, normDiv, showFig=False)
    
    plt.show()
    
  #hope signal persists across different variables -> correlations
  corr = gradb*pPerturb
  
  normCorr = MidpointNormalize(midpoint=0)
  plot_field_recentered(corr, normCorr, showFig=True)

if __name__=='__main__':
  #demo()
  demo_compareConvolve()



