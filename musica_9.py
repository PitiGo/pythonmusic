import pygame
import numpy as np
import random
import re

class AdvancedSoundtrackGenerator:
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

        self.current_mood = "neutral"
        self.mood_scales = {
            "happy": ['C', 'D', 'E', 'G', 'A'],
            "sad": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            "tense": ['B', 'C', 'D#', 'E', 'F#', 'G', 'A'],
            "neutral": ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        }
        self.channels = {mood: pygame.mixer.Channel(i) for i, mood in enumerate(self.mood_scales)}
        self.chord_progressions = {
            "happy": [['C', 'G', 'Am7', 'Fmaj7'], ['F', 'G7', 'C', 'Em7']],
            "sad": [['Am', 'F', 'C', 'G'], ['Dm7', 'G7', 'Cmaj7', 'Am']],
            "tense": [['Bdim', 'F#7', 'D#m7b5', 'G#m'], ['F#7', 'C#', 'D#m', 'B7']],
            "neutral": [['Cmaj7', 'G7', 'Am7', 'Em7'], ['Fmaj7', 'C', 'G', 'Am7']]
        }

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        wave = np.zeros(frames)

        # Añadir armónicos para enriquecer el sonido
        if wave_type == 'sine':
            wave += np.sin(2 * np.pi * frequency * t)
            for i in range(2, 6):
                wave += (1 / i) * np.sin(2 * np.pi * frequency * i * t)
        elif wave_type == 'square':
            for i in range(1, 10, 2):
                wave += (1 / i) * np.sin(2 * np.pi * frequency * i * t)
        elif wave_type == 'sawtooth':
            for i in range(1, 10):
                wave += (1 / i) * np.sin(2 * np.pi * frequency * i * t)
        elif wave_type == 'triangle':
            for i in range(1, 10, 2):
                wave += ((-1) ** ((i - 1) // 2) * (1 / i ** 2)) * np.sin(2 * np.pi * frequency * i * t)
        elif wave_type == 'noise':
            wave += np.random.uniform(-1, 1, frames)
        else:
            wave += np.sin(2 * np.pi * frequency * t)

        # Filtro de paso bajo simple para suavizar el sonido
        wave = np.convolve(wave, np.ones(5) / 5, mode='same')

        # Envelope ADSR
        attack = int(0.01 * frames)
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

    def get_chord_frequencies(self, chord):
        chord_types = {
            '': [1, 5/4, 3/2],          # Mayor
            'm': [1, 6/5, 3/2],         # Menor
            '7': [1, 5/4, 3/2, 7/4],    # Séptima dominante
            'maj7': [1, 5/4, 3/2, 15/8],# Séptima mayor
            'm7': [1, 6/5, 3/2, 7/4],   # Séptima menor
            'dim': [1, 6/5, 7/5],       # Disminuido
            'sus4': [1, 4/3, 3/2],      # Suspensión cuarta
            'aug': [1, 5/4, 8/5]        # Aumentado
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

    def generate_chord(self, chord, duration, wave_type='sine'):
        frequencies = self.get_chord_frequencies(chord)
        chord_wave = np.zeros(int(duration * self.sample_rate), dtype=np.float64)
        for freq in frequencies:
            chord_wave += self.generate_wave(freq, duration, wave_type, 0.3)
        return (chord_wave / len(frequencies)).astype(np.int16)

    def generate_arpeggio(self, chord, duration, wave_type='sine'):
        frequencies = self.get_chord_frequencies(chord)
        arpeggio = np.array([], dtype=np.int16)
        note_duration = duration / len(frequencies)
        for freq in frequencies:
            arpeggio = np.concatenate((arpeggio, self.generate_wave(freq, note_duration, wave_type, 0.4)))
        return arpeggio

    def generate_bassline(self, chord, duration, wave_type='sawtooth'):
        root_freq = self.get_chord_frequencies(chord)[0] / 2  # Una octava más baja
        return self.generate_wave(root_freq, duration, wave_type, 0.5)

    def generate_kick(self, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        frequency = 150
        kick = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 70)
        return (kick * 0.7 * 32767).astype(np.int16)

    def generate_snare(self, duration):
        noise = np.random.uniform(-1, 1, int(self.sample_rate * duration))
        decay = np.exp(-np.linspace(0, 50, int(self.sample_rate * duration)))
        tone = np.sin(2 * np.pi * 180 * np.linspace(0, duration, int(self.sample_rate * duration)))
        snare = (noise * 0.5 + tone * 0.5) * decay
        return (snare * 0.7 * 32767).astype(np.int16)

    def generate_hihat(self, duration):
        noise = np.random.uniform(-1, 1, int(self.sample_rate * duration))
        decay = np.exp(-np.linspace(0, 200, int(self.sample_rate * duration)))
        hihat = noise * decay
        return (hihat * 0.3 * 32767).astype(np.int16)

    def generate_complex_rhythm(self, duration, complexity=2):
        frames = int(duration * self.sample_rate)
        rhythm = np.zeros(frames, dtype=np.float32)
        beat_frames = frames // 16  # Notas de 1/16

        kick = self.generate_kick(duration / 16)
        snare = self.generate_snare(duration / 16)
        hihat = self.generate_hihat(duration / 16)

        # Patrones de ritmo mejorados
        if complexity == 1:
            kick_pattern = [1, 0, 0, 0] * 4
            snare_pattern = [0, 0, 0, 0, 1, 0, 0, 0] * 2
            hihat_pattern = [1, 0, 1, 0] * 4
        elif complexity == 2:
            kick_pattern = [1, 0, 0, 1, 1, 0, 0, 0] * 2
            snare_pattern = [0, 0, 1, 0, 1, 0, 1, 0] * 2
            hihat_pattern = [1] * 16
        else:  # complexity 3
            kick_pattern = [1, 0, 1, 0, 1, 0, 0, 1] * 2
            snare_pattern = [0, 1, 0, 1, 0, 1, 0, 1] * 2
            hihat_pattern = [1] * 16

        for i in range(16):
            start = i * beat_frames
            end = (i + 1) * beat_frames
            if kick_pattern[i % len(kick_pattern)]:
                rhythm[start:end] += kick[:beat_frames].astype(np.float32) / 32767 * 0.8
            if snare_pattern[i % len(snare_pattern)]:
                rhythm[start:end] += snare[:beat_frames].astype(np.float32) / 32767 * 0.6
            if hihat_pattern[i % len(hihat_pattern)]:
                rhythm[start:end] += hihat[:beat_frames].astype(np.float32) / 32767 * 0.4

        # Normalizar y convertir de vuelta a int16
        max_val = np.max(np.abs(rhythm))
        if max_val > 0:
            rhythm = rhythm / max_val
        return (rhythm * 32767).astype(np.int16)

    def generate_melody(self, scale, duration, complexity=2):
        melody = np.array([], dtype=np.int16)
        note_duration = duration / 8  # Notas de 1/8
        previous_note = None
        scale_notes = [f"{note}5" for note in scale]  # Octava 5 para la melodía

        for _ in range(16):
            if random.random() < 0.2:
                melody = np.concatenate((melody, np.zeros(int(note_duration * self.sample_rate), dtype=np.int16)))
                previous_note = None
            else:
                if previous_note and random.random() < 0.7:
                    idx = scale_notes.index(previous_note)
                    if random.random() < 0.5 and idx > 0:
                        note = scale_notes[idx - 1]
                    elif idx < len(scale_notes) - 1:
                        note = scale_notes[idx + 1]
                    else:
                        note = random.choice(scale_notes)
                else:
                    note = random.choice(scale_notes)
                wave = self.generate_wave(self.base_frequencies[note], note_duration, 'sine', 0.4)
                melody = np.concatenate((melody, wave))
                previous_note = note

        if complexity > 1:
            # Añadir notas de gracia y variaciones
            for i in range(0, len(melody), int(note_duration * self.sample_rate)):
                if random.random() < 0.3:
                    grace_note = random.choice(scale_notes)
                    grace_duration = note_duration / 4
                    grace_wave = self.generate_wave(self.base_frequencies[grace_note], grace_duration, 'sine', 0.3)
                    melody[i:i + len(grace_wave)] += grace_wave[:len(melody[i:i + len(grace_wave)])]

        return melody

    def add_reverb(self, wave, decay=0.5, delay=0.02):
        delay_samples = int(delay * self.sample_rate)
        reverb_wave = np.zeros(len(wave) + delay_samples, dtype=np.float32)
        reverb_wave[:len(wave)] += wave
        reverb_wave[delay_samples:] += wave * decay
        return reverb_wave[:len(wave)]

    def generate_complex_section(self, mood, duration=32, complexity=2):
        chord_progression = random.choice(self.chord_progressions[mood])
        section = np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        for idx, chord in enumerate(chord_progression):
            chord_duration = duration / len(chord_progression)
            chord_wave = self.generate_chord(chord, chord_duration)
            arpeggio = self.generate_arpeggio(chord, chord_duration)
            bassline = self.generate_bassline(chord, chord_duration)
            rhythm = self.generate_complex_rhythm(chord_duration, complexity)
            melody = self.generate_melody(self.mood_scales[mood], chord_duration, complexity)

            start = int((idx * chord_duration) * self.sample_rate)
            end = int(((idx + 1) * chord_duration) * self.sample_rate)
            section[start:end] += chord_wave.astype(np.float32) / 32767 * 0.3
            section[start:end] += arpeggio.astype(np.float32) / 32767 * 0.2
            section[start:end] += bassline.astype(np.float32) / 32767 * 0.4
            section[start:end] += rhythm.astype(np.float32) / 32767 * 0.6
            section[start:end] += melody[:len(chord_wave)].astype(np.float32) / 32767 * 0.5

        # Añadir reverb
        section = self.add_reverb(section)

        # Normalizar y convertir de vuelta a int16
        max_val = np.max(np.abs(section))
        if max_val > 0:
            section = section / max_val
        return (section * 32767).astype(np.int16)

    def generate_advanced_soundtrack(self, mood, length=4, complexity=3):
        soundtrack = np.array([], dtype=np.int16)
        for _ in range(length):
            section = self.generate_complex_section(mood, 12, complexity)
            soundtrack = np.concatenate((soundtrack, section))
        return np.column_stack((soundtrack, soundtrack))

    def play_mood(self, mood, volume=0.5, complexity=2):
        soundtrack = self.generate_advanced_soundtrack(mood, 4, complexity)
        sound = pygame.sndarray.make_sound(soundtrack)
        sound.set_volume(volume)
        self.channels[mood].play(sound, loops=-1)

    def update_soundtrack(self, game_state):
        if game_state['danger_level'] > 0.7:
            new_mood = "tense"
            complexity = 3
        elif game_state['score'] > game_state['high_score']:
            new_mood = "happy"
            complexity = 2
        elif game_state['health'] < 0.3:
            new_mood = "sad"
            complexity = 1
        else:
            new_mood = "neutral"
            complexity = 2

        if new_mood != self.current_mood:
            self.fade_transition(self.current_mood, new_mood, complexity)
            self.current_mood = new_mood

    def fade_transition(self, old_mood, new_mood, complexity, transition_time=2000):
        steps = 20
        self.play_mood(new_mood, volume=0, complexity=complexity)
        for i in range(steps):
            old_volume = 1 - (i / steps)
            new_volume = i / steps
            self.channels[old_mood].set_volume(old_volume)
            self.channels[new_mood].set_volume(new_volume)
            pygame.time.wait(transition_time // steps)

# Ejemplo de uso
if __name__ == "__main__":
    pygame.init()
    soundtrack = AdvancedSoundtrackGenerator()
    
    # Iniciar con música triste
    soundtrack.play_mood("sad", complexity=3)

    # Simular cambios en el estado del juego
    game_states = [
        {"danger_level": 0.2, "score": 100, "high_score": 500, "health": 0.8},
        {"danger_level": 0.8, "score": 200, "high_score": 500, "health": 0.6},
        {"danger_level": 0.3, "score": 600, "high_score": 500, "health": 0.9},
        {"danger_level": 0.1, "score": 300, "high_score": 600, "health": 0.2}
    ]

    running = True
    state_index = 0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Cambiar el estado del juego cada 20 segundos
        if pygame.time.get_ticks() % 20000 < 100:
            soundtrack.update_soundtrack(game_states[state_index])
            state_index = (state_index + 1) % len(game_states)

        clock.tick(60)

    pygame.quit()

print("Generador de banda sonora iniciado. Presiona Ctrl+C para detener.")

try:
    while True:
        pygame.time.wait(1000)  # Esperar 1 segundo
except KeyboardInterrupt:
    print("\nGenerador de banda sonora detenido.")
    pygame.quit()