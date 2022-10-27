import numpy as np
from scipy.signal import argrelmin, argrelmax, find_peaks
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
  sections: dict
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
  
  # remove DC component
  def highpass_data(self, threshold = 5):
    # ref
    ref_fft = np.fft.fft(self.ref)
    ref_fft[:threshold] = 0
    self.ref = np.fft.ifft(ref_fft)

    # # input
    # input_fft = np.fft.fft(self.input)
    # input_fft[:threshold] = 0
    # self.input = np.fft.ifft(input_fft)

  def _find_zerocross(self, order):
    ref_ref = abs(copy.copy(self.ref))
    zerocross = argrelmin(ref_ref, order = order)[0]
    return zerocross

  def resampling(self, order = 10):
    zerocross = self._find_zerocross(order)
    self.input = self.input[zerocross]
    self.ref = self.ref[zerocross]

  # find origin points by searching maximum points of each peak
  def find_origin_points(self,threshold, order = 500):
    assert threshold > 0
    # ref_input = copy.copy(self.input)
    # ref_input[ref_input < threshold] = 0
    # origin_args = argrelmax(ref_input, order = order)[0]
    origin_args, _ = find_peaks(self.input, threshold, distance = order)
    self.origin = origin_args
  

  # separate data
  def _sep_data(self):
    start = 0
    for i in range(len(self.origin)-1):
      end = (self.origin[i+1] - self.origin[i])//2 + self.origin[i]
      self.sections[self.origin[i]] = [start, end]
      start = end + 1
    end = len(self.input) - 1
    self.sections[self.origin[-1]] = [start, end]

  # reverse the order of even iter data
  def _reverse_data(self):
    if not self.sections:
      self._sep_data()
    reverse_origins = []
    for i in range(len(self.origin)//2):
      reverse_origins.append(self.origin[2*i + 1])
    for target in reverse_origins:
      start, end = self.sections[target]
      rev_data = np.zeros(end-start+1)
      for i in range(end-start+1):
        rev_data[i] = self.input[end-i]
    self.rev = True

  # serach minimum width availanble for data addition
  def _search_opt_width(self):
    if not self.rev:
      self._reverse_data()
    min_l = len(self.input)
    min_r = len(self.input)
    for origin in self.origin:
      width_l = origin - self.sections[origin][0]
      width_r = self.sections[origin][1] - origin
      if width_l < min_l:
        min_l = width_l
      if width_r < min_r:
        min_r = width_r
    return min_l, min_r

  # addition data
  def addition_data(self):
    if not self.rev:
      self._reverse_data()
    min_l, min_r = self._search_opt_width()
    extracted = np.zeros([len(self.origin), min_l + min_r])
    for i in range(len(self.origin)):
      extracted[i] = self.input[self.origin[i] - min_l : self.origin[i] + min_r]
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
    min_l, min_r = self._search_opt_width()
    _can_be_resize = True
    if min_l < width_l:
      print("width_L of this data is small than other's")
      _can_be_resize = False
    if min_r < width_r:
      print("width_R of this data is small than other's")
      _can_be_resize = False
    if _can_be_resize:
      shrinked = np.zeros(width_l + width_r)
      L = len(self.input)
      shrinked = self.added[min_l - width_l : L-(min_r-width_r)]
      self.added = shrinked
    
  # main function
  def FFT(self):
    if len(self.added):
      _to_be_fft = self.added
    else:
      _to_be_fft = self.input
    _to_be_fft = _to_be_fft * np.hamming(len(_to_be_fft))
    return np.fft.fft(_to_be_fft)

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