import numpy as np
import random
import re
import soundfile as sf
import os

class LatinBalladGenerator:
    def __init__(self):
        self.sample_rate = 22050  # Reduced sample rate
        
        # Generate notes with octaves
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.base_frequencies = {}
        for octave in range(2, 7):
            for i, note in enumerate(note_names):
                note_name = f"{note}{octave}"
                n = i + (octave - 4) * 12
                freq = 440 * (2 ** (n / 12))
                self.base_frequencies[note_name] = freq

        # Harmonic minor scale for Latin ballad
        self.scale = ['A', 'B', 'C', 'D', 'E', 'F', 'G#']
        self.chord_progression = [
            ['Am', 'Dm', 'E7', 'Am'],
            ['F', 'G', 'Em', 'Am'],
            ['Dm', 'G', 'C', 'F'],
            ['Bm7b5', 'E7', 'Am', 'Am']
        ]
        self.current_progression = random.choice(self.chord_progression)

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, endpoint=False)
        wave = np.zeros(frames)

        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'triangle':
            wave = 2 * np.abs(2 * (frequency * t - np.floor(0.5 + frequency * t))) - 1
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (frequency * t - np.floor(0.5 + frequency * t))
        elif wave_type == 'noise':
            wave = np.random.uniform(-1, 1, frames)
        else:
            wave = np.sin(2 * np.pi * frequency * t)

        # ADSR envelope
        attack = int(0.05 * frames)
        decay = int(0.1 * frames)
        sustain_level = 0.7
        release = int(0.1 * frames)
        sustain = frames - (attack + decay + release)
        if sustain < 0:
            sustain = 0

        envelope = np.ones(frames)
        envelope[:attack] = np.linspace(0, 1, attack, endpoint=False)
        envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay, endpoint=False)
        if sustain > 0:
            envelope[attack + decay:attack + decay + sustain] = sustain_level
        envelope[-release:] = np.linspace(sustain_level, 0, release, endpoint=False)

        wave *= envelope
        return (wave * volume * (2 ** 15 - 1)).astype(np.int16)

    def get_chord_frequencies(self, chord_name):
        chord_types = {
            'maj': [0, 4, 7],
            'm': [0, 3, 7],
            '7': [0, 4, 7, 10],
            'm7': [0, 3, 7, 10],
            'maj7': [0, 4, 7, 11],
            'dim': [0, 3, 6],
            'm7b5': [0, 3, 6, 10],
        }
        match = re.match(r'^([A-G]#?)(.*)$', chord_name)
        if match:
            root_note = match.group(1)
            chord_type = match.group(2) if match.group(2) else 'maj'
        else:
            root_note = 'C'
            chord_type = 'maj'
        root_freq = self.base_frequencies.get(f"{root_note}3", 261.63)
        intervals = chord_types.get(chord_type, [0, 4, 7])
        frequencies = [root_freq * (2 ** (interval / 12)) for interval in intervals]
        return frequencies

    def generate_chord(self, chord_name, duration):
        frequencies = self.get_chord_frequencies(chord_name)
        chord_wave = np.zeros(int(duration * self.sample_rate), dtype=np.float64)
        for freq in frequencies:
            wave = self.generate_wave(freq, duration, 'triangle', 0.4).astype(np.float64)
            chord_wave += wave
        chord_wave /= len(frequencies)
        return chord_wave.astype(np.int16)

    def generate_melody(self, duration):
        total_frames = int(duration * self.sample_rate)
        melody_wave = np.array([], dtype=np.int16)
        note_duration = duration / 16
        scale_notes = [f"{note}5" for note in self.scale]
        previous_note = None

        for _ in range(16):
            if previous_note and random.random() < 0.8:
                idx = scale_notes.index(previous_note)
                step = random.choice([-1, 0, 1])
                idx = (idx + step) % len(scale_notes)
                note = scale_notes[idx]
            else:
                note = random.choice(scale_notes)
            freq = self.base_frequencies[note]
            wave = self.generate_wave(freq, note_duration, 'sine', 0.6)
            melody_wave = np.concatenate((melody_wave, wave))
            previous_note = note

        if len(melody_wave) > total_frames:
            melody_wave = melody_wave[:total_frames]
        elif len(melody_wave) < total_frames:
            padding = np.zeros(total_frames - len(melody_wave), dtype=np.int16)
            melody_wave = np.concatenate((melody_wave, padding))

        return melody_wave

    def generate_percussion(self, duration):
        frames = int(duration * self.sample_rate)
        percussion_wave = np.zeros(frames, dtype=np.float32)
        beat_frames = frames // 16

        for i in range(16):
            start = i * beat_frames
            end = start + beat_frames

            hihat = self.generate_hihat(beat_frames)
            percussion_wave[start:end] += hihat

            if i == 4 or i == 12:
                snare = self.generate_snare(beat_frames)
                percussion_wave[start:end] += snare

            if i == 0 or i == 8:
                kick = self.generate_kick(beat_frames)
                percussion_wave[start:end] += kick

        max_val = np.max(np.abs(percussion_wave))
        if max_val > 0:
            percussion_wave = percussion_wave / max_val

        return (percussion_wave * (2 ** 15 - 1)).astype(np.int16)

    def generate_snare(self, frames):
        noise = np.random.uniform(-1, 1, frames)
        envelope = np.exp(-np.linspace(0, 1, frames) * 20)
        return noise * envelope * 0.5

    def generate_hihat(self, frames):
        noise = np.random.uniform(-1, 1, frames)
        envelope = np.exp(-np.linspace(0, 1, frames) * 30)
        return noise * envelope * 0.3

    def generate_kick(self, frames):
        t = np.linspace(0, frames / self.sample_rate, frames, endpoint=False)
        freq = np.linspace(150, 50, frames)
        envelope = np.exp(-t * 20)
        return np.sin(2 * np.pi * freq * t) * envelope * 0.8

    def add_reverb(self, wave, decay=0.5, delay=0.05):
        delay_samples = int(delay * self.sample_rate)
        reverb_wave = np.zeros(len(wave) + delay_samples, dtype=np.float32)
        reverb_wave[:len(wave)] += wave
        reverb_wave[delay_samples:] += wave * decay
        return reverb_wave[:len(wave)]

    def generate_bassline(self, duration):
        total_frames = int(duration * self.sample_rate)
        bass_wave = np.array([], dtype=np.int16)
        note_duration = duration / len(self.current_progression)

        for chord_name in self.current_progression:
            root_note = chord_name[:-1] if chord_name.endswith(('m', '7', 'b5')) else chord_name
            freq = self.base_frequencies.get(f"{root_note}2", 130.81)
            wave = self.generate_wave(freq, note_duration, 'sine', 0.5)
            bass_wave = np.concatenate((bass_wave, wave))

        if len(bass_wave) > total_frames:
            bass_wave = bass_wave[:total_frames]
        elif len(bass_wave) < total_frames:
            padding = np.zeros(total_frames - len(bass_wave), dtype=np.int16)
            bass_wave = np.concatenate((bass_wave, padding))

        return bass_wave

    def generate_section(self, duration):
        total_frames = int(duration * self.sample_rate)
        section_wave = np.zeros(total_frames, dtype=np.float32)
        chord_duration = duration / len(self.current_progression)

        chord_wave = np.array([], dtype=np.int16)
        for chord_name in self.current_progression:
            wave = self.generate_chord(chord_name, chord_duration)
            chord_wave = np.concatenate((chord_wave, wave))

        if len(chord_wave) > total_frames:
            chord_wave = chord_wave[:total_frames]
        elif len(chord_wave) < total_frames:
            padding = np.zeros(total_frames - len(chord_wave), dtype=np.int16)
            chord_wave = np.concatenate((chord_wave, padding))

        melody_wave = self.generate_melody(duration)
        percussion_wave = self.generate_percussion(duration)
        bass_wave = self.generate_bassline(duration)

        chord_wave = chord_wave.astype(np.float32) / (2 ** 15 - 1)
        melody_wave = melody_wave.astype(np.float32) / (2 ** 15 - 1)
        percussion_wave = percussion_wave.astype(np.float32) / (2 ** 15 - 1)
        bass_wave = bass_wave.astype(np.float32) / (2 ** 15 - 1)

        section_wave += chord_wave * 0.5
        section_wave += melody_wave * 0.8
        section_wave += percussion_wave * 0.7
        section_wave += bass_wave * 0.5

        section_wave = self.add_reverb(section_wave)

        max_val = np.max(np.abs(section_wave))
        if max_val > 0:
            section_wave = section_wave / max_val

        # Convert to mono for smaller file size
        return (section_wave * (2 ** 15 - 1)).astype(np.int16)

   def save_to_wav(self, filename="latin_ballad.wav", duration=16, num_sections=4):
        print("Generating music...")
        sections = []
        for i in range(num_sections):
            print(f"Generating section {i+1} of {num_sections}...")
            section = self.generate_section(duration)
            sections.append(section)
        
        all_sections = np.concatenate(sections)

        print("Saving WAV file...")
        sf.write(filename, all_sections, self.sample_rate)
        print(f"WAV file saved as: {filename}")

        # Convert to MP3 using afconvert (built into macOS)
        mp3_filename = filename.replace('.wav', '.m4a')
        try:
            # Use afconvert (native macOS tool) for conversion
            command = f'afconvert -f m4af -d aac -b 64000 "{filename}" "{mp3_filename}"'
            print(f"Running command: {command}")
            result = os.system(command)
            
            if result == 0:
                os.remove(filename)
                print(f"Audio file saved as: {mp3_filename}")
            else:
                print("Error during conversion")
                
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            print("WAV file has been kept.")

if __name__ == "__main__":
    print("Starting Latin ballad generator...")
    generator = LatinBalladGenerator()
    # Generate a song with 4 sections (64 seconds total)
    generator.save_to_wav("latin_ballad.wav", duration=16, num_sections=4)