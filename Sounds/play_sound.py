import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import soundfile as sf
import sofa
import librosa
from scipy import signal
import sounddevice as sd
import time

SOFA = [
    "HRTFsets/SOFA Far-Field/HRIR_FULL2DEG.sofa",
    "HRTFsets/SOFA Far-Field/HRIR_L2702.sofa",
    "HRTFsets/SOFA Far-Field/HRIR_CIRC360.sofa",
]

# function to find the closest array value to a given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_sound(sound_path, angle, elevation, set_index=0, target_fs=48000):
    # initialization
    HRTF = sofa.Database.open(SOFA[set_index])
    fs_H = HRTF.Data.SamplingRate.get_values()[0]
    positions = HRTF.Source.Position.get_values(system='spherical')

    H = np.zeros([HRTF.Dimensions.N, 2])
    Stereo3D = np.zeros([HRTF.Dimensions.N, 2])

    # need to adjust angle to match CCW convention
    angle_label = angle
    angle = (360 - angle) % 360 # for edge case, we want 360 to map to 0

    [az, az_idx] = find_nearest(positions[:, 0], angle)
    subpositions = positions[np.where(positions[:, 0] == az)]
    [elev, sub_idx] = find_nearest(subpositions[:, 1], elevation)

    # we now have the indicies for the azimuth and elevation of the source
    # closest to the HRIR data for the random direction

    # get the HRTF data for the left and right ears
    H[:, 0] = HRTF.Data.IR.get_values(indices={"M": az_idx + sub_idx, "R": 0, "E": 0})
    H[:, 1] = HRTF.Data.IR.get_values(indices={"M": az_idx + sub_idx, "R": 1, "E": 0})

    if fs_H != target_fs:
            H = librosa.core.resample(H.transpose(), orig_sr=fs_H, target_sr=target_fs).transpose()

    # pick random sources
    [x, fs_x] = sf.read(sound_path)
    if x.shape[1] > 1:
        x = np.mean(x, axis=1)
    if fs_x != target_fs:
        x = librosa.core.resample(x.transpose(), orig_sr=fs_x, target_sr=target_fs).transpose()

    # convolve and add LR signals to final array (general pointwise normalization)
    rend_L = signal.fftconvolve(x, H[:, 0])
    rend_R = signal.fftconvolve(x, H[:, 1])

    M = np.max([np.abs(rend_L), np.abs(rend_R)])
    if len(Stereo3D) < len(rend_L):
        diff = len(rend_L) - len(Stereo3D)
        Stereo3D = np.append(Stereo3D, np.zeros([diff, 2]), axis=0)
    Stereo3D[0:len(rend_L), 0] += (rend_L / M)
    Stereo3D[0:len(rend_R), 1] += (rend_R / M)

    # print operation
    print("Rendered at azimuth: ", angle_label, "degrees, elevation: ", elev, "degrees")

    return Stereo3D

fs = 48000

def boop(yaw, pitch, depth, delay=0.5):
    SOUND_DAMPENING_CONSTANT = 4*3.14

    volume = 1/(depth*SOUND_DAMPENING_CONSTANT)
         
    binaural = get_sound("boop.wav", yaw, pitch)
    sd.play(binaural, fs)
    time.sleep(delay)


if __name__ == "__main__":

    for i in range(5):
        boop(60, 50, 1)
        boop(60, 0, 1)
        

    # for angle in range(0, 360, 45):
    #     # generate random display
    #     binaural = get_sound("boop.wav", angle, 0)

    #     sd.play(binaural, fs)

    #     time.sleep(0.5)