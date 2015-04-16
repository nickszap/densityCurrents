#among many resources, http://scipy-lectures.github.io/advanced/image_processing/ seems like a useful introduction to image processing
# for a list of scipy.ndimage functions: http://docs.scipy.org/doc/scipy/reference/ndimage.html

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import math
from matplotlib.colors import Normalize
from scipy import ndimage, stats
from mpl_toolkits.basemap import Basemap
from numpy.linalg import eig, inv

Cp = 1004.5; Rd = 287.04; Rd_Cp = Rd/Cp; p0 = 1.e5; grav = 9.81; pi=np.pi

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
  #from a trimmed wrfout file
  #return t,p,q,u,v from file and convert to useful units
  
  #first model level variables
  t = data.variables['T'][:,:]+273.15 #to K\
  p = data.variables['PSFC'][:,:]*100. #to Pa
  q = data.variables['Q'][:,:] #kg/kg
  u = data.variables['U0'][:,:] #m/s
  v = data.variables['V'][:,:] #m/s
  slp = data.variables['SLP'][:,:]*100. #Pa
  latitude = data.variables['LAT'][:,:]
  longitude = data.variables['LON'][:,:]
  
  
  return (t,p,q,u,v,slp,latitude,longitude)

def readTimeLevel_fullData(data):
  #read a real wrfout file
  #return t,p,q,u,v from file and convert to useful units
  
  iTime = 0
  
  #first model level variables
  t = data.variables['T2'][iTime,:,:]+273.15 #to K\
  p = data.variables['PSFC'][iTime,:,:] #*100. #to Pa
  q = data.variables['Q2'][iTime,:,:] #kg/kg
  u = data.variables['U10'][iTime,:,:] #m/s
  v = data.variables['V10'][iTime,:,:] #m/s
  
  latitude = data.variables['XLAT'][iTime,:,:]
  longitude = data.variables['XLONG'][iTime,:,:]
  
  #calc slp in Pa
  hgt = data.variables['HGT'][iTime,:,:]
  H = 8.5e3 #scale height of 8.5km for T ~ 290K
  #tv = t*(1+.61*q)
  #H = tv*Rd/grav; print H
  slp = p*np.exp(hgt/H)
  
  return (t,p,q,u,v,slp,latitude,longitude)

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

def makeMapCoords(lat,lon,objs,nObjs):
  #return x,y coordinates on a map from lat/lon in degrees
  m = Basemap(projection='ortho',lat_0=(np.amax(lat)+np.amin(lat))/2,lon_0=(np.amax(lon)+np.amin(lon))/2, resolution='l')
  return m(lon,lat)

def makegridcoords(dims):
    ny = dims[0]
    nx = dims[1]
    x,y  = np.meshgrid(range(nx),range(ny))
    return x,y

def find_index_of_nearest_xy(polyarray, objectarray):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = numpy.where(distance==distance.min())
    return idy[0],idx[0]

def demo():
  #read in data --------------------
  #fDir = '/data02/densityCurrents/cases/'
  #f = fDir+'2014071011.nc'
  #dxGrid = 1.e3
  fDir = '/data02/densityCurrents/cases/coldpoolexamples/'
  f = fDir+'wrfout.2014060402'
  dxGrid = 4.e3
  
  data = netCDF4.Dataset(f,'r')
  
  t,p,q,u,v,slp,latitude,longitude = readTimeLevel_fullData(data)
  #t,p,q,u,v,slp,latitude,longitude = readTimeLevel(data)
  ny,nx = t.shape
  
  plt.pcolormesh(slp); plt.colorbar()
  
  #calculate some variables --------------
  theta = calc_potentialTemperature(t,p)
  thetav = theta*(1+.61*q)
  
  fieldsRef = calc_refFields([thetav,slp], windowSpacing=50.e3, gridSpacing=dxGrid)
  thetav_ref = fieldsRef[0]; slp_ref = fieldsRef[1]
  
  buoy = grav*(thetav-thetav_ref)/thetav_ref
  gradb_dir = np.gradient(buoy);
  gradb = np.maximum(gradb_dir[0],gradb_dir[1])
  #gradb = ndimage.morphological_gradient(buoy, size=(5,5))
  
  pPerturb = slp-slp_ref
  #gradP = ndimage.morphological_gradient(pPerturb, size=(5,5))
  
  du_dxy = np.gradient(u/dxGrid)
  dv_dxy = np.gradient(v/dxGrid)
  div = du_dxy[1]+dv_dxy[0]
  gradDiv = ndimage.morphological_gradient(div, size=(5,5))
  maxDiv = ndimage.filters.maximum_filter(div, size=(7,7))
  minDiv = ndimage.filters.minimum_filter(div, size=(7,7))
  divDiff = maxDiv-minDiv
  
  #temperature advection
  dt_dxy = np.gradient(thetav/dxGrid)
  tAdvect = u*dt_dxy[0]+v*dt_dxy[1]
  
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
  #print signalThresh
  signalFlow = -buoy*pPerturb
  if (False):
    norm = MidpointNormalize(midpoint=0)
    plot_field_recentered(signalFlow, norm, title='-buoy*pPerturb',showFig=False)
  
  if (False):
    signalFront = gradb*gradP
    norm = MidpointNormalize(midpoint=0)
    plot_field_recentered(signalFront, norm, title='gradb*gradp',showFig=False)
    
    candidates = (signalFlow>0)*(signalFront>signalThresh)
  if (False):
    candidates = (signalFlow>signalThresh)*(divDiff>1.e-3)
  elif (False):
    candidates = (signalFlow>signalThresh)*(tAdvect<0)
  else:
    candidates = signalFlow>signalThresh
  
  candidates = ndimage.morphology.binary_closing(candidates, structure=np.ones((5,5),dtype=int), iterations=1)
  objs, nObjs = ndimage.measurements.label(candidates)
  print "largest label: ", np.max(objs), nObjs
  
  #threshold objects by size
  lenThreshold = 50.e3 #50 km
  backgroundThreshold = min(nx,ny)*dxGrid*.8 #suspiciously hacky, eh?
  labelForTooSmall = -1
  x,y = makegridcoords(objs.shape)
  for objlabel in xrange(nObjs+1):
    xobj = x[objs == objlabel]
    yobj = y[objs == objlabel]
    
    iMin = np.argmin(xobj); iMax = np.argmax(xobj);
    if (iMin == iMax):
      #the line is vertical so use y for endpts
      iMin = np.argmin(yobj); iMax = np.argmax(yobj);
    dx = xobj[iMax]-xobj[iMin]; dy = yobj[iMax]-yobj[iMin];
    dx *= dxGrid; dy *= dxGrid; 
    objLength = np.sqrt(dx*dx+dy*dy) #in km
    if (objLength<lenThreshold or objLength>backgroundThreshold):
      objs[objs == objlabel] = labelForTooSmall
    else:
      print "object passes length test: ", xobj[0], yobj[0], objLength
  
  #reorder the object labels so too small-> -1 a    
  uniqueObjs = np.unique(objs); uniqueObjs.sort(); #so -1 for objects that were too small to be first element
  nUniqueObjs = len(uniqueObjs)
  newLabels = objs.copy() #to not overwrite
  for iObj in xrange(nUniqueObjs):
    oldLabel = uniqueObjs[iObj]
    newLabels[objs==oldLabel] = iObj-1
  nObjs = nUniqueObjs -1 #-1 since we toss objects that were too small
  objs = newLabels
  
  if (True):
    #i think background is always obj=0?
    plt.figure()
    plt.pcolormesh(objs, cmap=plt.cm.Set1)
    plt.colorbar()
    plt.show()
  
  isDensityCurrent = np.ones(nObjs, dtype=int)
  x,y = makegridcoords(objs.shape)
  for objlabel in xrange(nObjs):
      xobj = x[objs == objlabel]
      yobj = y[objs == objlabel]
      uobj = u[objs == objlabel]
      vobj = v[objs == objlabel]
      
      p = np.polyfit(xobj,yobj,2)
      dydxobj = 2*p[0]*xobj + p[1]
      dydx = 2*p[0]*x + p[1]
      normal = -1/dydx
      normalobj = -1/dydxobj
      #plt.figure()
      #plt.plot(xobj,yobj)
      #plt.plot(xobj,np.polyval(p,xobj))
      xobjPlot = np.array([np.min(xobj), round(np.mean(xobj)), np.max(xobj)])
      
      lenNormal = 20.
      dydxPlot = 2*p[0]*xobjPlot+p[1]
      normalPlot = -1/dydxPlot
      #print normalPlot
      #print math.atan(1/normalPlot[0])*(180/pi)
      normx = 1
      mag = np.sqrt(normx*normx+normalPlot*normalPlot)
      newXplus = xobjPlot+normx*lenNormal/mag
      newYplus = np.polyval(p,xobjPlot)+normalPlot*lenNormal/mag
      newXminus = xobjPlot-normx*lenNormal/mag
      newYminus = np.polyval(p,xobjPlot)-normalPlot*lenNormal/mag
      newX = [newXplus, newXminus]
      newY = [newYplus, newYminus]
      gridXsound = [np.around(newXplus),np.around(newXminus)]
      gridYsound = [np.around(newYplus),np.around(newYminus)]
      
      if (True):
        #plt.plot(xobj, yobj, 'bo')
        plt.plot(newX,newY)
        #plt.quiver(xobj, yobj, uobj, vobj)
        #tObj = t.copy(); tObj[objs != objlabel] = np.mean(tObj[objs == objlabel])
        tObj = tAdvect.copy(); tObj[objs != objlabel] = np.median(tObj[objs == objlabel])
        plt.pcolormesh(x, y, tObj); plt.colorbar()
        plt.show()
  
  
if __name__=='__main__':
  demo()



