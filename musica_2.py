import pygame
import numpy as np
import random

class SoundtrackGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.sample_rate = 44100
        self.base_frequencies = {
            'DO': 261.63, 'RE': 293.66, 'MI': 329.63, 'FA': 349.23,
            'SOL': 392.00, 'LA': 440.00, 'SI': 493.88
        }
        self.current_melody = []

    def generate_tone(self, frequency, duration, volume=0.5):
        frames = int(duration * self.sample_rate)
        arr = np.array([4096 * volume * np.sin(2.0 * np.pi * frequency * x / self.sample_rate) for x in range(frames)]).astype(np.int16)
        return pygame.sndarray.make_sound(np.c_[arr,arr])

    def generate_chord(self, frequencies, duration, volume=0.5):
        frames = int(duration * self.sample_rate)
        chord = np.zeros(frames, dtype=np.float32)
        for freq in frequencies:
            chord += np.sin(2.0 * np.pi * freq * np.arange(frames) / self.sample_rate)
        chord = (chord / len(frequencies) * 4096 * volume).astype(np.int16)
        return pygame.sndarray.make_sound(np.c_[chord,chord])

    def generate_melody(self, length, note_duration=0.5):
        self.current_melody = [random.choice(list(self.base_frequencies.keys())) for _ in range(length)]
        return [self.generate_tone(self.base_frequencies[note], note_duration) for note in self.current_melody]

    def play_soundtrack(self, length=8, loops=-1):
        melody = self.generate_melody(length)
        for sound in melody:
            sound.play()
            pygame.time.wait(int(500))  # 500 ms entre notas
        
    def update_soundtrack(self, game_state):
        # Aquí puedes añadir lógica para cambiar la música basándote en el estado del juego
        pass

# Ejemplo de uso
if __name__ == "__main__":
    pygame.init()
    soundtrack = SoundtrackGenerator()
    soundtrack.play_soundtrack()
    
    # Mantén el programa corriendo para escuchar la música
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()