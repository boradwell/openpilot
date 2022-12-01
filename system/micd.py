#!/usr/bin/env python3
import sounddevice as sd
import numpy as np

from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.realtime import Ratekeeper
from system.swaglog import cloudlog

RATE = 10
DT_MIC = 1. / RATE
REFERENCE_SPL = 2 * 10 ** -5  # newtons/m^2


STEPS_PER_DB = 100
MIN_DB = -100
frequencies = np.linspace(0, 22050, num=22050)
a_weighting = 20 * np.log10(frequencies / 10**3)
a_weighting[np.isinf(a_weighting)] = MIN_DB


a_weight = [1.00000000e+00, -3.98107171e+00, 5.94603023e+00, -3.93304134e+00,
            9.78581702e-01, -1.37889712e-15]
b_weight = [2.00000000e-05, 1.00000000e-05, 2.00000000e-05, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00]

b = np.array([0.0997773295537, 0.0997773295537])
a = np.array([1.000000000000, -0.809511377552])


def f(samples):
  filtered_samples = np.convolve(b, samples, mode="same")

  # calculate the RMS value of the filtered samples
  rms = np.sqrt(np.mean(filtered_samples ** 2))
  spl = 20 * np.log10(rms / 20e-6)

  print(f"The A-weighted SPL is {spl:.2f} dB")


def calc_loudness(data):
  # apply the filter to the input data
  data_a = np.lfilter(b_weight, a_weight, data)
  # calculate the root mean square (RMS) of the filtered data
  rms = np.sqrt(np.mean(data_a ** 2))
  # return the perceived loudness in decibels (dB)
  return 20 * np.log10(rms)

def function(indata):
  # compute the spectrum of the audio data
  spectrum = np.abs(np.fft.rfft(indata))
  print(spectrum)

  # compute the loudness of each frequency in the spectrum
  loudness = 10 * np.log10(spectrum)

  # apply the A-weighting curve to the loudness values
  loudness = loudness + a_weighting

  # map the loudness values onto a scale from 0 to 1
  loudness = np.maximum(loudness, MIN_DB) / MIN_DB

  # sum the loudness values to get the perceived loudness of the sound
  perceived_loudness = np.sum(loudness) / len(loudness)

  # output the perceived loudness
  print('perceived loudness', perceived_loudness)


class Mic:
  def __init__(self, pm):
    self.pm = pm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.empty(0)
    self.spl_filter = FirstOrderFilter(0, 4, DT_MIC, initialized=False)

  def update(self):
    # self.measurements contains amplitudes from -1 to 1 which we use to
    # calculate an uncalibrated sound pressure level
    # print(self.measurements)
    if len(self.measurements) > 0:
      f(self.measurements)
      # print('calc loudness', calc_loudness(self.measurements))
      # https://www.engineeringtoolbox.com/sound-pressure-d_711.html
      sound_pressure = np.sqrt(np.mean(self.measurements ** 2))  # RMS of amplitudes
      sound_pressure_level = 20 * np.log10(sound_pressure / REFERENCE_SPL) if sound_pressure > 0 else 0  # dB
      self.spl_filter.update(sound_pressure_level)
    else:
      sound_pressure = 0
      sound_pressure_level = 0
    print(sound_pressure_level)


    self.measurements = np.empty(0)

    msg = messaging.new_message('microphone')
    msg.microphone.soundPressure = float(sound_pressure)
    msg.microphone.soundPressureDb = float(sound_pressure_level)
    msg.microphone.filteredSoundPressureDb = float(self.spl_filter.x)

    self.pm.send('microphone', msg)
    self.rk.keep_time()

  def callback(self, indata, frames, time, status):
    self.measurements = np.concatenate((self.measurements, indata[:, 0]))

  def micd_thread(self, device=None):
    if device is None:
      device = "sysdefault"

    with sd.InputStream(device=device, channels=1, samplerate=44100, callback=self.callback) as stream:
      cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}")
      while True:
        self.update()


def main(pm=None, sm=None):
  if pm is None:
    pm = messaging.PubMaster(['microphone'])

  mic = Mic(pm)
  mic.micd_thread()


if __name__ == "__main__":
  main()
