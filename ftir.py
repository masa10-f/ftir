import numpy as np
from scipy.signal import find_peaks
import copy

class FTIR:
  """
  Executing FTIR from CW reference data & measurement data
  
  attributes
  ==============
  input: numpy.ndarray
      input data including spectral information
  ref: numpy.ndarray
      reference data (sine wave)
  added: numpy.ndarray
      averaged input data
  origin: list
      the original point position for peaks
  sections: dict of list
      index: origin point for each section
      data: taple of start point and end point i.e. {origin: (start, end)}
  rev: bool
      if even index data is re-orderd, rev = True. initial num is False.
  """

  def __init__(self, input, ref):
    self.input = copy.copy(input)
    self.ref = copy.copy(ref)
    self.added = np.array([])
    self.origin = []
    self.sections = {}
    self.rev = False
    # self.corrected = False
  
  def highpass_data(self, threshold = 5):
    """Apply high pass filter for removing DC component in reference data. this process is necessary for _find_zerocross().

    Parameters
    -------
    threshold: int(optional)
      remove low freq data below threshold(>=0)
    """
    assert threshold >= 0
    # ref
    ref_fft = np.fft.fft(self.ref)
    ref_fft[:threshold] = 0
    ref_fft[-threshold:-1] = 0
    self.ref = np.fft.ifft(ref_fft)

    # # input
    # input_fft = np.fft.fft(self.input)
    # input_fft[:threshold] = 0
    # self.input = np.fft.ifft(input_fft)

  def _find_zerocross(self, distance):
    """Find zerocross in reference data for FTIR.
    Parameters
    -------
    distance: int
      Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    
    Returns
    -------
    zerocross: list of ints
      storage index of zerocross points
    """
    assert distance >= 1
    ref_ref = abs(copy.copy(self.ref))
    zerocross, _ = find_peaks(-ref_ref, distance = distance)
    return zerocross

  def resampling(self, distance = 10):
    """Resampling data by using zerocross

    Parameters
    -------
    distance: int(optional)
      This is used in scipy.signal.find_peaks().Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    """
    zerocross = self._find_zerocross(distance)
    self.input = self.input[zerocross]
    self.ref = self.ref[zerocross]

  # find origin points by searching maximum points of each peak
  def find_origin_points(self,threshold = 0, distance = 500):
    """Find origins for each peak. index of origins will be stored in self.origin.

    Parameters
    -------
    threshold: int(optional)
       Required threshold of peaks, the vertical distance to its neighboring samples. Either a number, None, an array matching x or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required threshold.
       If threshold is specified, calculation time will be shorter.
    distance: int(optional)
      This is used in scipy.signal.find_peaks().Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Psuedopeaks are removed first until the condition is fulfilled for all remaining peaks.

    """
    assert threshold > 0
    # ref_input = copy.copy(self.input)
    # ref_input[ref_input < threshold] = 0
    # origin_args = argrelmax(ref_input, order = order)[0]
    origin_args, _ = find_peaks(self.input, threshold, distance = distance)
    self.origin = origin_args
  

  # separate data
  def _sep_data(self):
    """Separate data for accumulation. use data between first and last peak. divide rigion by the middle points of each peak. the number of regions is (number of peaks) - 2"""
    assert len(self.origin) > 2
    for i in range(0,len(self.origin)-2):
      start = (self.origin[i+1] - self.origin[i])//2 + self.origin[i]
      end = (self.origin[i+2] - self.origin[i+1])//2 + self.origin[i+1] - 1
      self.sections[self.origin[i+1]] = [start, end]

  # reverse the order of even iter data
  def _reverse_data(self):
    """Reorder odd index data. this function is used especially for repeated system(i.e. mirror goes to and back repeatedly)
    """
    if not self.sections:
      self._sep_data()
    for i in range((len(self.sections)+1)//2):
      target = self.origin[2*i + 1]
      start, end = self.sections[target]
      rev_data = np.zeros(end-start)
      for j in range(end-start):
        rev_data[j] = self.input[end-j]
        if end - j == target:
          self.origin[2*i+1] = start + j
          self.sections[start+j] = [start, end]
          del self.sections[target]
      self.input[start:end] = rev_data
    self.rev = True

  # serach minimum width availanble for data addition
  def _search_opt_width(self):
    """Find optimal width form accumulation.
    Returns
    ------
    min_l: int
      max left width for accumulation
    min_r: int
      max right width for accumulation
    """
    if not self.rev:
      self._reverse_data()
    min_l = len(self.input)
    min_r = len(self.input)
    for origin in self.sections.keys():
      width_l = origin - self.sections[origin][0]
      width_r = self.sections[origin][1] - origin
      if width_l < min_l:
        min_l = width_l
      if width_r < min_r:
        min_r = width_r
    return min_l, min_r

  # accumulate data
  def accumulate_data(self):
    """Accumulate data. output will be storaged in self.added
    """
    if not self.rev:
      self._reverse_data()
    assert len(self.sections) > 0
    min_l, min_r = self._search_opt_width()
    extracted = np.zeros([len(self.sections.keys()), min_l + min_r])
    for i in range(1, len(self.origin)-1):
      extracted[i-1] = self.input[self.origin[i] - min_l : self.origin[i] + min_r]
    self.added = extracted.mean(axis=0)


  # def _correct_xaxis(self, zerosections):
  #   for i in range(zerosections - 1):
  #     zero_n = zerosections[i]
  #     zero_np1 = zerosections[i + 1]
  #     mid = self.time[zero_n : zero_np1]
  #     step = (mid[-1]-mid[0])/(len(mid))
  #     for j in range(1, len(mid)):
  #       self.time[mid[j]] = self.time[mid[j-1]] + step

  # resize width mainly used when copared with other data
  def resize(self, width_l, width_r):
    """Resize data for comparison with other data. for example, one compares data with sample and without sample for calculationg transmission.
    Parameters
    -------
    width_l: int
      resize parameter for the left width
    width_r: int
      resize parameter for the right width
    """
    min_l, min_r = self._search_opt_width()
    _can_be_resize = True
    if min_l < width_l:
      print("width_L of this data is small than other's")
      _can_be_resize = False
    if min_r < width_r:
      print("width_R of this data is small than other's")
      _can_be_resize = False
    if _can_be_resize:
      shrinked = np.zeros(width_l + width_r + 1)
      L = len(self.added)
      shrinked = self.added[min_l - width_l : L-(min_r-width_r)]
      self.added = shrinked
      assert shrinked[-1] != 0
    
  # main function
  def FFT(self):
    """Execute FFT for data. if there exists accumulated data, this will be transfered.

    Returns
    -------
    freq_data: numpy.ndarray 1d
      fourier transformed data
    """
    if len(self.added):
      _to_be_fft = self.added
    else:
      _to_be_fft = self.input
    _to_be_fft = _to_be_fft * np.hamming(len(_to_be_fft))
    freq_data = np.fft.fft(_to_be_fft)
    return freq_data

  # export stored data
  def export_data(self):
    return self.input, self.ref

  # export origin points data
  def export_origins(self):
    return self.origin
  
  # return length of input data
  def count_L(self):
    L_input = len(self.input)
    L_ref = len(self.ref)
    if L_input != L_ref:
      print("length of input and reference is not samae!")
    return L_input
  
  # length of added data
  def length_addition(self):
    return len(self.added)

def wavenumber(lam):
  """
  calculate wavenumber from wavelength
  nm to cm^{-1}

  attributes
  ============
  lam: float
      wavelength(nm)
  ------------
  return: float
      wavenumber(cm^{-1})
  """
  return 1/lam * 10**(7)

def x_axis(L, unit):
  """
  return wavenumber array

  attributes
  ===========
  L: int
      length of data
  unit: float
      unit wavenumber calculated from wavenumber func.
  """
  return np.linspace(0,unit, L)