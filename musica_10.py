import pygame
import numpy as np
import random
import re

class RomanticSoundtrackGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.sample_rate = 44100

        # Generar una lista completa de notas con octavas
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.base_frequencies = {}
        for octave in range(0, 8):
            for i, note in enumerate(note_names):
                note_name = f"{note}{octave}"
                n = i + (octave - 4) * 12 - 9
                freq = 440 * (2 ** (n / 12))
                self.base_frequencies[note_name] = freq

        self.current_mood = "romantic"
        self.mood_scales = {
            "romantic": ['C', 'D', 'E', 'G', 'A'],  # Pentatónica Mayor
        }
        self.channels = {mood: pygame.mixer.Channel(i) for i, mood in enumerate(self.mood_scales)}
        self.chord_progressions = {
            "romantic": [
                ['Cmaj7', 'Am7', 'Dm7', 'G7'],
                ['Fmaj7', 'Em7', 'Dm7', 'G7'],
                ['Cmaj7', 'Em7', 'Fmaj7', 'G7'],
            ]
        }

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        wave = np.zeros(frames)

        # Crear una onda más suave y cálida
        if wave_type == 'sine':
            wave += np.sin(2 * np.pi * frequency * t)
            wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)  # Añadir segundo armónico
        elif wave_type == 'triangle':
            wave += 2 * np.abs(2 * (frequency * t - np.floor(0.5 + frequency * t))) - 1
        elif wave_type == 'piano':
            # Simular un sonido de piano básico
            wave += np.sin(2 * np.pi * frequency * t) * np.exp(-t * 5)
            wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t * 5)
        else:
            wave += np.sin(2 * np.pi * frequency * t)

        # Envelope ADSR
        attack = int(0.05 * frames)
        decay = int(0.1 * frames)
        sustain_level = 0.8
        release = int(0.2 * frames)
        sustain = frames - (attack + decay + release)

        envelope = np.ones(frames)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
        envelope[attack + decay:attack + decay + sustain] = sustain_level
        envelope[-release:] = np.linspace(sustain_level, 0, release)

        wave *= envelope
        return (wave * volume * (2 ** 15 - 1)).astype(np.int16)

    def get_chord_frequencies(self, chord):
        chord_types = {
            '': [1, 5/4, 3/2],              # Mayor
            'm': [1, 6/5, 3/2],             # Menor
            '7': [1, 5/4, 3/2, 7/4],        # Séptima dominante
            'maj7': [1, 5/4, 3/2, 15/8],    # Séptima mayor
            'm7': [1, 6/5, 3/2, 7/4],       # Séptima menor
            '6': [1, 5/4, 3/2, 5/3],        # Sexta
            '9': [1, 5/4, 3/2, 7/4, 9/4],   # Novena
        }

        match = re.match(r'^([A-G][#b]?)(.*)$', chord)
        if match:
            root_note = f"{match.group(1)}4"  # Octava por defecto
            chord_type = match.group(2)
        else:
            root_note = 'C4'
            chord_type = ''
        root_freq = self.base_frequencies.get(root_note, 261.63)
        intervals = chord_types.get(chord_type, [1, 5/4, 3/2])  # Mayor por defecto
        frequencies = [root_freq * interval for interval in intervals]
        return frequencies

    def generate_chord(self, chord, duration, wave_type='piano'):
        frequencies = self.get_chord_frequencies(chord)
        chord_wave = np.zeros(int(duration * self.sample_rate), dtype=np.float64)
        for freq in frequencies:
            chord_wave += self.generate_wave(freq, duration, wave_type, 0.5)
        return (chord_wave / len(frequencies)).astype(np.int16)

    def generate_melody(self, scale, duration, complexity=2):
        melody = np.array([], dtype=np.int16)
        note_duration = duration / 8  # Notas de 1/8
        previous_note = None
        scale_notes = [f"{note}5" for note in scale]  # Octava 5 para la melodía

        for _ in range(8):
            if previous_note and random.random() < 0.8:
                idx = scale_notes.index(previous_note)
                step = random.choice([-1, 1])
                idx = (idx + step) % len(scale_notes)
                note = scale_notes[idx]
            else:
                note = random.choice(scale_notes)
            wave = self.generate_wave(self.base_frequencies[note], note_duration, 'sine', 0.6)
            melody = np.concatenate((melody, wave))
            previous_note = note

        # Añadir legato (suavizar transiciones)
        melody = np.convolve(melody, np.ones(100) / 100, mode='same')
        return melody.astype(np.int16)

    def add_reverb(self, wave, decay=0.4, delay=0.03):
        delay_samples = int(delay * self.sample_rate)
        reverb_wave = np.zeros(len(wave) + delay_samples, dtype=np.float32)
        reverb_wave[:len(wave)] += wave
        reverb_wave[delay_samples:] += wave * decay
        return reverb_wave[:len(wave)]

    def generate_simple_rhythm(self, duration):
        frames = int(duration * self.sample_rate)
        rhythm = np.zeros(frames, dtype=np.float32)
        beat_frames = frames // 4  # Compás de 4/4

        # Simular un suave golpe de percusión
        for i in range(4):
            start = i * beat_frames
            end = start + int(0.05 * self.sample_rate)  # Duración corta
            envelope = np.linspace(1, 0, end - start)
            rhythm[start:end] += envelope * 0.3

        return (rhythm * 32767).astype(np.int16)

    def generate_section(self, mood, duration=16):
        chord_progression = random.choice(self.chord_progressions[mood])
        section = np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        for idx, chord in enumerate(chord_progression):
            chord_duration = duration / len(chord_progression)
            chord_wave = self.generate_chord(chord, chord_duration)
            melody = self.generate_melody(self.mood_scales[mood], chord_duration)
            rhythm = self.generate_simple_rhythm(chord_duration)

            start = int((idx * chord_duration) * self.sample_rate)
            end = int(((idx + 1) * chord_duration) * self.sample_rate)
            section[start:end] += chord_wave.astype(np.float32) / 32767 * 0.6
            section[start:end] += melody[:len(chord_wave)].astype(np.float32) / 32767 * 0.8
            section[start:end] += rhythm[:len(chord_wave)].astype(np.float32) / 32767 * 0.2

        # Añadir reverb
        section = self.add_reverb(section)

        # Normalizar y convertir de vuelta a int16
        max_val = np.max(np.abs(section))
        if max_val > 0:
            section = section / max_val
        return (section * 32767).astype(np.int16)

    def generate_soundtrack(self, mood, length=4):
        soundtrack = np.array([], dtype=np.int16)
        for _ in range(length):
            section = self.generate_section(mood, duration=16)
            soundtrack = np.concatenate((soundtrack, section))
        return np.column_stack((soundtrack, soundtrack))

    def play_mood(self, mood, volume=0.5):
        soundtrack = self.generate_soundtrack(mood)
        sound = pygame.sndarray.make_sound(soundtrack)
        sound.set_volume(volume)
        self.channels[mood].play(sound, loops=-1)

    # Como es un ejemplo de música romántica, no cambiaremos el estado
    # pero puedes implementar cambios similares si lo deseas.

# Ejemplo de uso
if __name__ == "__main__":
    pygame.init()
    soundtrack = RomanticSoundtrackGenerator()

    # Iniciar con música romántica
    soundtrack.play_mood("romantic", volume=0.7)

    print("Generador de banda sonora romántica iniciado. Presiona Ctrl+C para detener.")

    try:
        while True:
            pygame.time.wait(1000)  # Esperar 1 segundo
    except KeyboardInterrupt:
        print("\nGenerador de banda sonora detenido.")
        pygame.quit()