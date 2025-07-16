import numpy as np

class Limiter:
    def __init__(self, threshold, attack_ms, release_ms, sample_rate):
        self.threshold = threshold
        self.attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
        self.release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))
        self.envelope = 0.0
        self.gain = 1.0

    def process(self, audio_buffer):
        output_buffer = np.zeros_like(audio_buffer)

        for i, sample in enumerate(audio_buffer):
            # 1. Envelope Follower: Track the absolute value of the signal
            abs_sample = abs(sample)
            if abs_sample > self.envelope:
                self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_sample
            else:
                self.envelope = self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_sample

            # 2. Gain Calculation: Determine the necessary gain reduction
            if self.envelope > self.threshold:
                target_gain = self.threshold / self.envelope
            else:
                target_gain = 1.0

            # 3. Apply Gain Smoothing (using attack/release for gain too)
            self.gain = self.attack_coeff * self.gain + (1.0 - self.attack_coeff) * target_gain

            # 4. Apply Gain to the Audio Sample
            output_buffer[i] = sample * self.gain

        return output_buffer

"""
def limit_gain_calc(data, envelope, threshold, attack_ms, release_ms, fs):

    Y = np.zeros_like(envelope)

    env = 0.0

    attack_coeff = np.exp(-1.0 / (fs * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (fs * release_ms / 1000.0))

    for i, sample in enumerate(envelope):
        # 1. Gain Calculation: Determine the necessary gain reduction
        if env > threshold:
            target_gain = threshold / envelope
        else:
            target_gain = 1.0

        # 3. Apply Gain Smoothing (using attack/release for gain too)
        gain = attack_coeff * gain + (1.0 - attack_coeff) * target_gain

        # 4. Apply Gain to the Audio Sample
        Y[i] = sample * gain

    return output_buffer

"""
