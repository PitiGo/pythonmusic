import pygame
import numpy as np
import random

class ComplexSoundtrackGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        self.sample_rate = 44100
        self.base_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23,
            'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        self.current_mood = "neutral"
        self.mood_scales = {
            "happy": ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            "sad": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            "tense": ['B', 'C', 'D', 'E', 'F', 'G', 'A'],
            "neutral": ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        }
        self.channels = {mood: pygame.mixer.Channel(i) for i, mood in enumerate(self.mood_scales)}

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        else:
            wave = np.sin(2 * np.pi * frequency * t)  # Default to sine
        fade = int(0.05 * frames)  # 5% fade in/out
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        return (wave * volume * 32767).astype(np.int16)

    def generate_chord(self, root_freq, chord_type, duration, wave_type='sine'):
        if chord_type == 'major':
            frequencies = [root_freq, root_freq * 5/4, root_freq * 3/2]
        elif chord_type == 'minor':
            frequencies = [root_freq, root_freq * 6/5, root_freq * 3/2]
        elif chord_type == 'diminished':
            frequencies = [root_freq, root_freq * 6/5, root_freq * 7/6]
        else:
            frequencies = [root_freq]  # Default to just the root note

        chord = np.zeros(int(duration * self.sample_rate), dtype=np.float64)
        for freq in frequencies:
            chord += self.generate_wave(freq, duration, wave_type, 0.3)
        return (chord / len(frequencies)).astype(np.int16)

    def generate_rhythm(self, duration, pattern):
        rhythm = np.zeros(int(duration * self.sample_rate), dtype=np.int16)
        beat_duration = duration / len(pattern)
        for i, beat in enumerate(pattern):
            if beat == 1:
                start = int(i * beat_duration * self.sample_rate)
                end = int((i + 1) * beat_duration * self.sample_rate)
                rhythm[start:end] = self.generate_wave(880, beat_duration, 'square', 0.1)
        return rhythm

    def generate_complex_melody(self, mood, length=8, measure_duration=2):
        scale = self.mood_scales[mood]
        melody = np.array([], dtype=np.int16)
        for _ in range(length):
            chord_root = random.choice(scale)
            chord_type = 'major' if random.random() > 0.3 else 'minor'
            chord = self.generate_chord(self.base_frequencies[chord_root], chord_type, measure_duration)
            
            arpeggio = np.array([], dtype=np.int16)
            for _ in range(4):  # 4 notes per measure
                note = random.choice(scale)
                arpeggio = np.concatenate((arpeggio, self.generate_wave(self.base_frequencies[note], measure_duration/4, 'sine', 0.4)))
            
            rhythm = self.generate_rhythm(measure_duration, [1, 0, 1, 1, 0, 1, 1, 0])
            
            measure = chord + arpeggio + rhythm
            melody = np.concatenate((melody, measure))

        return np.column_stack((melody, melody))

    def play_mood(self, mood, volume=0.5):
        melody = self.generate_complex_melody(mood)
        sound = pygame.sndarray.make_sound(melody)
        sound.set_volume(volume)
        self.channels[mood].play(sound, loops=-1)

    def update_soundtrack(self, game_state):
        if game_state['danger_level'] > 0.7:
            new_mood = "tense"
        elif game_state['score'] > game_state['high_score']:
            new_mood = "happy"
        elif game_state['health'] < 0.3:
            new_mood = "sad"
        else:
            new_mood = "neutral"

        if new_mood != self.current_mood:
            self.fade_transition(self.current_mood, new_mood)
            self.current_mood = new_mood

    def fade_transition(self, old_mood, new_mood, transition_time=2000):
        steps = 20
        self.play_mood(new_mood, volume=0)
        for i in range(steps):
            old_volume = 1 - (i / steps)
            new_volume = i / steps
            self.channels[old_mood].set_volume(old_volume)
            self.channels[new_mood].set_volume(new_volume)
            pygame.time.wait(transition_time // steps)

# Ejemplo de uso
if __name__ == "__main__":
    pygame.init()
    soundtrack = ComplexSoundtrackGenerator()
    
    # Iniciar con música tensa
    soundtrack.play_mood("happy")

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

        # Cambiar el estado del juego cada 10 segundos
        if pygame.time.get_ticks() % 10000 < 100:
            soundtrack.update_soundtrack(game_states[state_index])
            state_index = (state_index + 1) % len(game_states)

        clock.tick(60)

    pygame.quit()