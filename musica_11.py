import pygame
import numpy as np
import random
import re

class RomanticSoundGenerator:
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
        pygame.init()
        self.sample_rate = 44100

        # Generar notas con octavas
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.base_frequencies = {}
        for octave in range(0, 8):
            for i, note in enumerate(note_names):
                note_name = f"{note}{octave}"
                n = i + (octave - 4) * 12 - 9
                freq = 440 * (2 ** (n / 12))
                self.base_frequencies[note_name] = freq

        self.scale = ['C', 'D', 'E', 'G', 'A']  # Escala pentatónica mayor
        self.chord_progression = [
            ['Cmaj7', 'Am7', 'Dm7', 'G7'],
            ['Fmaj7', 'Em7', 'Dm7', 'G7'],
            ['Cmaj7', 'Em7', 'Fmaj7', 'G7']
        ]
        self.current_progression = random.choice(self.chord_progression)

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        wave = np.zeros(frames)

        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == 'triangle':
            wave = 2 * np.abs(2 * (frequency * t - np.floor(0.5 + frequency * t))) - 1
        elif wave_type == 'sawtooth':
            wave = 2 * (frequency * t - np.floor(0.5 + frequency * t))
        elif wave_type == 'noise':
            wave = np.random.uniform(-1, 1, frames)
        else:
            wave = np.sin(2 * np.pi * frequency * t)

        # Envelope ADSR
        attack = int(0.05 * frames)
        decay = int(0.1 * frames)
        sustain_level = 0.7
        release = int(0.2 * frames)
        sustain = frames - (attack + decay + release)

        envelope = np.ones(frames)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
        envelope[attack + decay:attack + decay + sustain] = sustain_level
        envelope[-release:] = np.linspace(sustain_level, 0, release)

        wave *= envelope
        return (wave * volume * (2 ** 15 - 1)).astype(np.int16)

    def get_chord_frequencies(self, chord_name):
        chord_types = {
            'maj7': [0, 4, 7, 11],
            'm7': [0, 3, 7, 10],
            '7': [0, 4, 7, 10],
            'm': [0, 3, 7],
            '': [0, 4, 7]
        }
        match = re.match(r'^([A-G]#?)(.*)$', chord_name)
        if match:
            root_note = match.group(1)
            chord_type = match.group(2)
        else:
            root_note = 'C'
            chord_type = ''
        root_freq = self.base_frequencies.get(f"{root_note}4", 261.63)
        intervals = chord_types.get(chord_type, [0, 4, 7])
        frequencies = [root_freq * (2 ** (interval / 12)) for interval in intervals]
        return frequencies

    def generate_chord(self, chord_name, duration):
        frequencies = self.get_chord_frequencies(chord_name)
        chord_wave = np.zeros(int(duration * self.sample_rate), dtype=np.float64)
        for freq in frequencies:
            chord_wave += self.generate_wave(freq, duration, 'triangle', 0.4)
        chord_wave /= len(frequencies)
        return chord_wave.astype(np.int16)

    def generate_melody(self, duration):
        melody_wave = np.array([], dtype=np.int16)
        note_duration = duration / 8  # Dividir el tiempo en 8 notas
        scale_notes = [f"{note}5" for note in self.scale]
        previous_note = None

        for _ in range(8):
            if previous_note and random.random() < 0.8:
                idx = scale_notes.index(previous_note)
                step = random.choice([-1, 1])
                idx = (idx + step) % len(scale_notes)
                note = scale_notes[idx]
            else:
                note = random.choice(scale_notes)
            freq = self.base_frequencies[note]
            wave = self.generate_wave(freq, note_duration, 'sine', 0.6)
            melody_wave = np.concatenate((melody_wave, wave))
            previous_note = note

        return melody_wave

    def generate_percussion(self, duration):
        frames = int(duration * self.sample_rate)
        percussion_wave = np.zeros(frames, dtype=np.float32)
        beat_frames = frames // 16  # Dividir en 16 golpes

        for i in range(16):
            start = i * beat_frames
            end = start + beat_frames

            # Kick en los tiempos fuertes
            if i % 4 == 0:
                kick = self.generate_kick(beat_frames)
                percussion_wave[start:end] += kick

            # Snare en el tiempo 2 y 4
            if i % 4 == 2:
                snare = self.generate_snare(beat_frames)
                percussion_wave[start:end] += snare

            # Hi-hat en cada tiempo
            hihat = self.generate_hihat(beat_frames)
            percussion_wave[start:end] += hihat

        # Normalizar
        max_val = np.max(np.abs(percussion_wave))
        if max_val > 0:
            percussion_wave = percussion_wave / max_val

        return (percussion_wave * (2 ** 15 - 1)).astype(np.int16)

    def generate_kick(self, frames):
        t = np.linspace(0, frames / self.sample_rate, frames, False)
        freq = np.linspace(150, 50, frames)
        envelope = np.exp(-t * 20)
        kick = np.sin(2 * np.pi * freq * t) * envelope * 0.8
        return kick

    def generate_snare(self, frames):
        noise = np.random.uniform(-1, 1, frames)
        envelope = np.exp(-np.linspace(0, 1, frames) * 10)
        snare = noise * envelope * 0.5
        return snare

    def generate_hihat(self, frames):
        noise = np.random.uniform(-1, 1, frames)
        envelope = np.exp(-np.linspace(0, 1, frames) * 30)
        hihat = noise * envelope * 0.3
        return hihat

    def add_reverb(self, wave, decay=0.4, delay=0.03):
        delay_samples = int(delay * self.sample_rate)
        reverb_wave = np.zeros(len(wave) + delay_samples, dtype=np.float32)
        reverb_wave[:len(wave)] += wave
        reverb_wave[delay_samples:] += wave * decay
        return reverb_wave[:len(wave)]

    def generate_section(self, duration):
        section_wave = np.zeros(int(duration * self.sample_rate), dtype=np.float32)
        chord_duration = duration / len(self.current_progression)
        start = 0

        for chord_name in self.current_progression:
            chord_wave = self.generate_chord(chord_name, chord_duration)
            melody_wave = self.generate_melody(chord_duration)
            percussion_wave = self.generate_percussion(chord_duration)

            end = start + len(chord_wave)
            section_wave[start:end] += chord_wave.astype(np.float32) / (2 ** 15 - 1) * 0.5
            section_wave[start:end] += melody_wave.astype(np.float32) / (2 ** 15 - 1) * 0.7
            section_wave[start:end] += percussion_wave.astype(np.float32) / (2 ** 15 - 1) * 0.5

            start = end

        # Añadir reverb
        section_wave = self.add_reverb(section_wave)

        # Normalizar
        max_val = np.max(np.abs(section_wave))
        if max_val > 0:
            section_wave = section_wave / max_val

        return (section_wave * (2 ** 15 - 1)).astype(np.int16)

    def play(self):
        # Generar una sección de 16 segundos
        section_wave = self.generate_section(16)
        sound = pygame.sndarray.make_sound(section_wave)
        sound.play()

        # Esperar hasta que termine la reproducción
        pygame.time.wait(int(16 * 1000))

    def start(self):
        try:
            while True:
                self.play()
        except KeyboardInterrupt:
            pygame.quit()
            print("Composición detenida.")

# Ejecutar la composición
if __name__ == "__main__":
    generator = RomanticSoundGenerator()
    generator.start()