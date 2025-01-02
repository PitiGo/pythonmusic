import pygame
import numpy as np
import random

class SoundtrackGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        self.sample_rate = 44100
        self.base_frequencies = {
            'DO': 261.63, 'RE': 293.66, 'MI': 329.63, 'FA': 349.23,
            'SOL': 392.00, 'LA': 440.00, 'SI': 493.88
        }
        self.current_mood = "neutral"
        self.mood_scales = {
            "happy": ['DO', 'RE', 'MI', 'FA', 'SOL', 'LA', 'SI'],
            "sad": ['LA', 'SI', 'DO', 'RE', 'MI', 'FA', 'SOL'],
            "tense": ['SI', 'DO', 'RE', 'MI', 'FA', 'SOL', 'LA'],
            "neutral": list(self.base_frequencies.keys())
        }
        self.channels = {mood: pygame.mixer.Channel(i) for i, mood in enumerate(self.mood_scales)}

    def generate_tone(self, frequency, duration, volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, False)
        tone = np.sin(2 * np.pi * frequency * t) * volume
        fade = 100  # Número de frames para el fade in/out
        tone[:fade] = tone[:fade] * np.linspace(0, 1, fade)
        tone[-fade:] = tone[-fade:] * np.linspace(1, 0, fade)
        return (tone * 32767).astype(np.int16)

    def generate_melody(self, mood, length=16, note_duration=0.2):
        scale = self.mood_scales[mood]
        melody = np.array([], dtype=np.int16)
        for _ in range(length):
            note = random.choice(scale)
            tone = self.generate_tone(self.base_frequencies[note], note_duration)
            melody = np.concatenate((melody, tone))
        # Convertir a array 2D para sonido estéreo
        return np.column_stack((melody, melody))

    def play_mood(self, mood, volume=0.5):
        melody = self.generate_melody(mood)
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
    soundtrack = SoundtrackGenerator()
    
    # Iniciar con música neutral
    soundtrack.play_mood("tense")

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