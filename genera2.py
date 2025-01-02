import numpy as np
import soundfile as sf
import os

class SimpleGenerator:
    def __init__(self):
        self.sample_rate = 22050  # Reduced sample rate for smaller files
        self.notes = {
            'DO': 261.63,
            'RE': 293.66,
            'MI': 329.63,
            'FA': 349.23,
            'SOL': 392.00
        }

    def generate_tone(self, frequency, duration, volume=0.5):
        # Generate frames
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, endpoint=False)
        tone = np.sin(2.0 * np.pi * frequency * t)

        # Add simple envelope to avoid clicks
        envelope = np.ones(frames)
        attack = int(0.02 * frames)  # 20ms attack
        release = int(0.02 * frames)  # 20ms release
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        
        tone *= envelope
        return (tone * volume * 32767).astype(np.int16)  # Convert to 16-bit audio

    def generate_melody(self, filename="melody.wav"):
        # Melody sequence
        melody = ['DO', 'RE', 'MI', 'FA', 'SOL', 'FA', 'MI', 'RE']
        duration = 0.5  # Duration of each note in seconds
        
        # Generate complete melody
        full_audio = np.array([], dtype=np.int16)
        for note in melody:
            tone = self.generate_tone(self.notes[note], duration)
            full_audio = np.concatenate((full_audio, tone))

        # Save as WAV
        print("Saving WAV file...")
        sf.write(filename, full_audio, self.sample_rate)
        print(f"WAV file saved as: {filename}")

        # Convert to MP3
        mp3_filename = filename.replace('.wav', '.mp3')
        try:
            # Convert to MP3 with basic settings
            command = f'ffmpeg -i "{filename}" -codec:a libmp3lame -b:a 64k "{mp3_filename}" -y'
            print("Converting to MP3...")
            result = os.system(command)
            
            if result == 0:
                os.remove(filename)  # Remove WAV file after successful conversion
                print(f"MP3 file saved as: {mp3_filename}")
            else:
                print("Error during MP3 conversion")
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            print("WAV file has been kept.")

if __name__ == "__main__":
    print("Starting melody generator...")
    generator = SimpleGenerator()
    generator.generate_melody("simple_melody.mp3")