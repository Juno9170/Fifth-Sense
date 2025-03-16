# boop.py - Modified version with thread-safe audio system
import numpy as np
import soundfile as sf
import sofa
import librosa
from scipy import signal
import sounddevice as sd
import time
from threading import Lock, Thread
from collections import deque
from play_tts import audio_lock  # Import the shared lock

# Keep your original constants
DB_REDUCTION_MAX = -60
DB_REDUCTION_MIN = -6
SOFA = [
    "HRTFsets/SOFA Far-Field/HRIR_FULL2DEG.sofa",
    "HRTFsets/SOFA Far-Field/HRIR_L2702.sofa",
    "HRTFsets/SOFA Far-Field/HRIR_CIRC360.sofa",
]

# Keep your utility functions
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def adjust_volume_db(audio, target_db):
    amplitude_ratio = 10 ** (target_db / 20.0)
    return audio * amplitude_ratio

# Add the new AudioSystem class
class AudioSystem:
    def __init__(self):
        self.fs = 48000
        self.queue = deque(maxlen=3)
        self.lock = Lock()
        self.worker_thread = None
        self.hrtf_cache = {}
        self._preload_hrtfs()
        
        
        # SoundDevice settings
        sd.default.samplerate = self.fs
        sd.default.device = None
        sd.default.dtype = 'float32'
        
        # Load boop sound once
        self.boop_sound, _ = sf.read("boop.wav", always_2d=True)
        if self.boop_sound.shape[1] > 1:
            self.boop_sound = np.mean(self.boop_sound, axis=1)
            
    def _preload_hrtfs(self):
        """Pre-load all HRTF data at initialization"""
        for path in SOFA:
            self.hrtf_cache[path] = {
                'obj': sofa.Database.open(path),
                'positions': None,
                'fs': None
            }
            hrtf = self.hrtf_cache[path]
            hrtf['fs'] = hrtf['obj'].Data.SamplingRate.get_values()[0]
            hrtf['positions'] = hrtf['obj'].Source.Position.get_values(system='spherical')

    def _get_sound(self, angle, elevation, set_index=0):
        """Thread-safe sound generation"""
        path = SOFA[set_index]
        hrtf = self.hrtf_cache[path]
        positions = hrtf['positions']
        
        angle = (360 - angle) % 360
        az, az_idx = find_nearest(positions[:, 0], angle)
        subpositions = positions[positions[:, 0] == az]
        elev, sub_idx = find_nearest(subpositions[:, 1], elevation)
        
        H = np.zeros([hrtf['obj'].Dimensions.N, 2])
        H[:, 0] = hrtf['obj'].Data.IR.get_values(indices={"M": az_idx + sub_idx, "R": 0, "E": 0})
        H[:, 1] = hrtf['obj'].Data.IR.get_values(indices={"M": az_idx + sub_idx, "R": 1, "E": 0})

        if hrtf['fs'] != self.fs:
            H = librosa.resample(H.T, orig_sr=hrtf['fs'], target_sr=self.fs).T
            
        x = self.boop_sound.copy()
        rend_L = signal.fftconvolve(x, H[:, 0])
        rend_R = signal.fftconvolve(x, H[:, 1])

        M = np.max([np.abs(rend_L), np.abs(rend_R)])
        return np.column_stack((rend_L/M, rend_R/M))

    def _worker(self):
        """Dedicated audio processing thread"""
        while True:
            try:
                if self.queue:
                    with self.lock:
                        task = self.queue.popleft()
                    
                    if task is None:
                        break

                    yaw, pitch, depth = task

                    if depth == 0:
                        db_reduction = 9999999999
                    else:
                        db_reduction = np.interp(depth, [0, 1], [DB_REDUCTION_MIN, DB_REDUCTION_MAX])
                    
                    print(f"DB Reduction: {db_reduction}")
                    binaural = self._get_sound(yaw, pitch)
                    binaural = adjust_volume_db(binaural, db_reduction)
                    
                    try:
                        # Acquire the same lock before playing boops
                        with audio_lock:
                            sd.play(binaural, self.fs)
                            time.sleep(0.2)
                            # sd.wait()
                    except Exception as e:
                        print(f"Error playing sound: {e}")
                        
            except Exception as e:
                print(f"Audio system error: {str(e)}")
                time.sleep(1)

    def start(self):
        """Start the audio system"""
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def add_sound(self, yaw, pitch, depth):
         with self.lock:
            # deque(maxlen) handles this automatically, but we'll add logging
            if len(self.queue) < self.queue.maxlen:
                self.queue.append((yaw, pitch, depth))
            

# Keep your test code but modify it to use AudioSystem
if __name__ == "__main__":
    audio = AudioSystem()
    audio.start()
    
    test_params = [
        (60, 50, 1), (60, 0, 1), (60, -50, 1),
        (60, 50, 0.5), (60, 0, 0.5), (60, -50, 0.5)
    ]
    
    for params in test_params:
        audio.add_sound(*params)
        time.sleep(1)  # Add small delay between sounds