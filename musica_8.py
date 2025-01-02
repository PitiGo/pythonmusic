import pygame
import numpy as np
import random

class AdvancedSoundtrackGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.sample_rate = 44100
        self.base_frequencies = {note: 440 * (2 ** ((i - 9) / 12)) for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])}
        self.current_mood = "neutral"
        self.mood_scales = {
            "happy": ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            "sad": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            "tense": ['B', 'C', 'D#', 'E', 'F#', 'G', 'A'],
            "neutral": ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        }
        self.channels = {mood: pygame.mixer.Channel(i) for i, mood in enumerate(self.mood_scales)}
        self.chord_progressions = {
            "happy": [['C', 'G', 'Am', 'F'], ['F', 'G', 'C', 'Am']],
            "sad": [['Am', 'F', 'C', 'G'], ['Dm', 'G', 'C', 'Am']],
            "tense": [['B', 'F#', 'D#m', 'G#m'], ['F#', 'C#', 'D#m', 'B']],
            "neutral": [['C', 'G', 'Am', 'Em'], ['F', 'C', 'G', 'Am']]
        }

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif wave_type == 'triangle':
            wave = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
        elif wave_type == 'noise':
            wave = np.random.uniform(-1, 1, frames)
        else:
            wave = np.sin(2 * np.pi * frequency * t)
        
        # Envelope ADSR
        attack = int(0.01 * frames)
        decay = int(0.1 * frames)
        sustain_level = 0.7
        release = int(0.2 * frames)
        
        envelope = np.ones(frames)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
        envelope[-release:] = np.linspace(sustain_level, 0, release)
        
        wave *= envelope
        return (wave * volume * 32767).astype(np.int16)

    def get_chord_frequencies(self, chord):
        root = chord[0]
        if len(chord) == 1:  # Major chord
            return [self.base_frequencies[root], self.base_frequencies[root] * 5/4, self.base_frequencies[root] * 3/2]
        elif chord[1] == 'm':  # Minor chord
            return [self.base_frequencies[root], self.base_frequencies[root] * 6/5, self.base_frequencies[root] * 3/2]
        else:
            return [self.base_frequencies[root]]

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
        rhythm = np.zeros(frames, dtype=np.float32)  # Cambiado a float32 para mayor rango dinámico
        beat_frames = frames // 16  # 16th notes

        kick = self.generate_kick(duration/16)
        snare = self.generate_snare(duration/16)
        hihat = self.generate_hihat(duration/16)

        if complexity == 1:
            kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            snare_pattern = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            hihat_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        elif complexity == 2:
            kick_pattern = [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
            snare_pattern = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
            hihat_pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:  # complexity 3
            kick_pattern = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
            snare_pattern = [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
            hihat_pattern = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for i in range(16):
            start = i * beat_frames
            end = (i + 1) * beat_frames
            if kick_pattern[i]:
                rhythm[start:end] += kick[:beat_frames].astype(np.float32) / 32767 * 0.8
            if snare_pattern[i]:
                rhythm[start:end] += snare[:beat_frames].astype(np.float32) / 32767 * 0.6
            if hihat_pattern[i]:
                rhythm[start:end] += hihat[:beat_frames].astype(np.float32) / 32767 * 0.4

        # Normalizar y convertir de vuelta a int16
        max_val = np.max(np.abs(rhythm))
        if max_val > 0:
            rhythm = rhythm / max_val
        return (rhythm * 32767).astype(np.int16)

    def generate_melody(self, scale, duration, complexity=2):
        melody = np.array([], dtype=np.int16)
        note_duration = duration / 4  # 8th notes
        for _ in range(8):
            if random.random() < 0.2:  # 20% chance of rest
                melody = np.concatenate((melody, np.zeros(int(note_duration * self.sample_rate), dtype=np.int16)))
            else:
                note = random.choice(scale)
                wave_type = random.choice(['sine', 'square', 'sawtooth', 'triangle'])
                melody = np.concatenate((melody, self.generate_wave(self.base_frequencies[note], note_duration, wave_type, 0.4)))
        
        if complexity > 1:
            # Add some grace notes and variations
            for i in range(0, len(melody), int(note_duration * self.sample_rate)):
                if random.random() < 0.3:  # 30% chance of adding a grace note
                    grace_note = random.choice(scale)
                    grace_duration = note_duration / 4
                    grace_wave = self.generate_wave(self.base_frequencies[grace_note], grace_duration, 'sine', 0.3)
                    melody[i:i+len(grace_wave)] += grace_wave[:len(melody[i:i+len(grace_wave)])]
        
        return melody

    def generate_complex_section(self, mood, duration=32, complexity=2):
        chord_progression = random.choice(self.chord_progressions[mood])
        section = np.zeros(int(duration * self.sample_rate), dtype=np.float32)
        
        for chord in chord_progression:
            chord_duration = duration / len(chord_progression)
            chord_wave = self.generate_chord(chord, chord_duration)
            arpeggio = self.generate_arpeggio(chord, chord_duration)
            bassline = self.generate_bassline(chord, chord_duration)
            rhythm = self.generate_complex_rhythm(chord_duration, complexity)
            melody = self.generate_melody(self.mood_scales[mood], chord_duration, complexity)
            
            start = int((chord_progression.index(chord) * chord_duration) * self.sample_rate)
            end = int(((chord_progression.index(chord) + 1) * chord_duration) * self.sample_rate)
            section[start:end] += chord_wave.astype(np.float32) / 32767 * 0.3
            section[start:end] += arpeggio.astype(np.float32) / 32767 * 0.2
            section[start:end] += bassline.astype(np.float32) / 32767 * 0.4
            section[start:end] += rhythm.astype(np.float32) / 32767 * 0.6
            section[start:end] += melody[:len(chord_wave)].astype(np.float32) / 32767 * 0.5

        # Normalizar y convertir de vuelta a int16
        max_val = np.max(np.abs(section))
        if max_val > 0:
            section = section / max_val
        return (section * 32767).astype(np.int16)

    def generate_advanced_soundtrack(self, mood, length=4, complexity=3):
        soundtrack = np.array([], dtype=np.int16)
        for _ in range(length):
            section = self.generate_complex_section(mood,12 , complexity)
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
    
    # Iniciar con música tensa
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