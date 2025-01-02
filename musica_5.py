import pygame
import numpy as np
import random

class AdvancedSoundtrackGenerator:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)  # Reducir el buffer para menor latencia
        self.sample_rate = 44100
        self.base_frequencies = {note: 440 * (2 ** ((i - 9) / 12)) for i, note in enumerate(['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'])}
        self.current_mood = "neutral"
        self.mood_scales = {
            "happy": ['C', 'D', 'E', 'G', 'A'],
            "sad": ['A', 'B', 'C', 'E', 'F'],
            "tense": ['B', 'C', 'D#', 'F', 'F#'],
            "neutral": ['C', 'D', 'E', 'F', 'G', 'A']
        }
        self.channels = {mood: pygame.mixer.Channel(i) for i, mood in enumerate(self.mood_scales)}
        self.chord_progressions = {
            "happy": [['C', 'G', 'Am', 'F'], ['F', 'G', 'C', 'Am']],
            "sad": [['Am', 'F', 'C', 'G'], ['Dm', 'G', 'C', 'Am']],
            "tense": [['Bm', 'F#', 'D#m', 'G#m'], ['F#', 'C#', 'D#m', 'Bm']],
            "neutral": [['C', 'G', 'Am', 'Em'], ['F', 'C', 'G', 'Am']]
        }

    def generate_wave(self, frequency, duration, wave_type='sine', volume=0.5):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, endpoint=False)
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
        
        # Aplicar filtro de paso bajo para suavizar el sonido y eliminar altas frecuencias no deseadas
        wave = self.low_pass_filter(wave, cutoff=2000)
        
        # Envelope ADSR con ajustes para mayor claridad
        attack = int(0.01 * frames)
        decay = int(0.05 * frames)
        sustain_level = 0.8
        release = int(0.1 * frames)
        
        envelope = np.ones(frames)
        envelope[:attack] = np.linspace(0, 1, attack, endpoint=False)
        envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay, endpoint=False)
        if frames - release - (attack + decay) > 0:
            envelope[attack+decay:frames - release] = sustain_level
        envelope[frames - release:] = np.linspace(sustain_level, 0, release, endpoint=False)
        
        wave *= envelope
        return (wave * volume * 32767).astype(np.int16)

    def low_pass_filter(self, data, cutoff=2000):
        # Filtro de paso bajo simple usando FFT
        fft_data = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(len(data), d=1/self.sample_rate)
        fft_data[frequencies > cutoff] = 0
        filtered_data = np.fft.irfft(fft_data)
        return filtered_data

    def high_pass_filter(self, data, cutoff=5000):
        # Filtro de paso alto usando FFT
        fft_data = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(len(data), d=1/self.sample_rate)
        fft_data[frequencies < cutoff] = 0
        filtered_data = np.fft.irfft(fft_data)
        return filtered_data

    def get_chord_frequencies(self, chord):
        root = chord[0]
        if len(chord) > 1 and chord[1] == 'b':
            root += 'b'
        if 'm' in chord:
            # Acorde menor
            return [self.base_frequencies[root], self.base_frequencies[root] * (6/5), self.base_frequencies[root] * (3/2)]
        else:
            # Acorde mayor
            return [self.base_frequencies[root], self.base_frequencies[root] * (5/4), self.base_frequencies[root] * (3/2)]

    def generate_chord(self, chord, duration, wave_type='sine'):
        frames = int(duration * self.sample_rate)
        frequencies = self.get_chord_frequencies(chord)
        chord_wave = np.zeros(frames, dtype=np.float64)
        for freq in frequencies:
            wave = self.generate_wave(freq, duration, wave_type, 0.3).astype(np.float64)
            chord_wave += wave[:frames]
        chord_wave /= len(frequencies)
        return chord_wave.astype(np.int16)

    def generate_arpeggio(self, chord, duration, wave_type='sine'):
        frames = int(duration * self.sample_rate)
        frequencies = self.get_chord_frequencies(chord)
        note_duration = duration / len(frequencies)
        note_frames = int(note_duration * self.sample_rate)
        arpeggio = np.array([], dtype=np.int16)
        for freq in frequencies:
            wave = self.generate_wave(freq, note_duration, wave_type, 0.4)
            arpeggio = np.concatenate((arpeggio, wave))
        # Asegurar que arpeggio tenga la longitud correcta
        if len(arpeggio) > frames:
            arpeggio = arpeggio[:frames]
        elif len(arpeggio) < frames:
            padding = np.zeros(frames - len(arpeggio), dtype=np.int16)
            arpeggio = np.concatenate((arpeggio, padding))
        return arpeggio

    def generate_bassline(self, chord, duration, wave_type='sine'):
        frames = int(duration * self.sample_rate)
        root_freq = self.get_chord_frequencies(chord)[0] / 2  # Una octava más baja
        bass_wave = self.generate_wave(root_freq, duration, wave_type, 0.5)
        bass_wave = bass_wave[:frames]
        return bass_wave

    def generate_kick(self, duration):
        frames = int(duration * self.sample_rate)
        t = np.linspace(0, duration, frames, endpoint=False)
        # Kick drum: una onda sinusoidal con un barrido descendente de frecuencia
        freq = np.linspace(150, 50, frames)
        wave = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-t * 40)  # Decaimiento rápido
        wave *= envelope
        return (wave * 0.9 * 32767).astype(np.int16)

    def generate_snare(self, duration):
        frames = int(duration * self.sample_rate)
        # Snare drum: ruido blanco con decaimiento rápido
        noise = np.random.uniform(-1, 1, frames)
        t = np.linspace(0, duration, frames, endpoint=False)
        envelope = np.exp(-t * 30)
        wave = noise * envelope
        return (wave * 0.7 * 32767).astype(np.int16)

    def generate_hihat(self, duration):
        frames = int(duration * self.sample_rate)
        # Hi-hat: ruido blanco de alta frecuencia con decaimiento muy rápido
        noise = np.random.uniform(-1, 1, frames)
        t = np.linspace(0, duration, frames, endpoint=False)
        envelope = np.exp(-t * 60)
        wave = noise * envelope
        # Aplicar filtro de paso alto para enfatizar frecuencias altas
        wave = self.high_pass_filter(wave, cutoff=5000)
        return (wave * 0.5 * 32767).astype(np.int16)

    def generate_complex_rhythm(self, duration, complexity=2):
        frames = int(duration * self.sample_rate)
        rhythm = np.zeros(frames, dtype=np.float32)
        beat_frames = frames // 16  # 16th notes

        # Definir patrones
        kick_pattern = [1, 0, 0, 0] * 4
        snare_pattern = [0, 0, 0, 0, 1, 0, 0, 0] * 2
        hihat_pattern = [1, 1, 1, 1] * 4

        if complexity > 1:
            kick_pattern = [1, 0, 0, 1] * 4
            snare_pattern = [0, 0, 1, 0] * 4
            hihat_pattern = [1, 0, 1, 0] * 4

        for i in range(16):
            start = i * beat_frames
            end = start + beat_frames
            if end > frames:
                end = frames
            duration_beat = (end - start) / self.sample_rate
            if kick_pattern[i]:
                wave = self.generate_kick(duration_beat)
                rhythm[start:end] += wave[:end - start] / 32767.0
            if snare_pattern[i]:
                wave = self.generate_snare(duration_beat)
                rhythm[start:end] += wave[:end - start] / 32767.0
            if hihat_pattern[i]:
                wave = self.generate_hihat(duration_beat)
                rhythm[start:end] += wave[:end - start] / 32767.0

        # Limitar el rango de valores para evitar distorsiones
        rhythm = np.clip(rhythm, -1, 1)
        return (rhythm * 32767).astype(np.int16)

    def generate_melody(self, scale, duration, complexity=2):
        frames = int(duration * self.sample_rate)
        note_duration = duration / 16  # 16th notes
        note_frames = int(note_duration * self.sample_rate)
        melody = np.array([], dtype=np.int16)
        for _ in range(16):
            if random.random() < 0.2:  # 20% de probabilidad de silencio
                wave = np.zeros(note_frames, dtype=np.int16)
            else:
                note = random.choice(scale)
                wave_type = 'sine'  # Usar onda sinusoidal para mayor claridad
                freq = self.base_frequencies[note]
                wave = self.generate_wave(freq, note_duration, wave_type, 0.5)
            melody = np.concatenate((melody, wave))
        # Asegurar que melody tenga la longitud correcta
        if len(melody) > frames:
            melody = melody[:frames]
        elif len(melody) < frames:
            padding = np.zeros(frames - len(melody), dtype=np.int16)
            melody = np.concatenate((melody, padding))
        return melody

    def generate_complex_section(self, mood, duration=8, complexity=2):
        total_frames = int(duration * self.sample_rate)
        chord_progression = random.choice(self.chord_progressions[mood])
        section = np.zeros(total_frames, dtype=np.float64)
        chord_duration = duration / len(chord_progression)
        chord_frames = int(chord_duration * self.sample_rate)
        
        for idx, chord in enumerate(chord_progression):
            start = idx * chord_frames
            end = start + chord_frames
            if end > total_frames:
                end = total_frames
            chord_wave = self.generate_chord(chord, chord_duration)
            arpeggio = self.generate_arpeggio(chord, chord_duration)
            bassline = self.generate_bassline(chord, chord_duration)
            rhythm = self.generate_complex_rhythm(chord_duration, complexity)
            melody = self.generate_melody(self.mood_scales[mood], chord_duration, complexity)
            
            # Asegurar que todos los arrays tengan la misma longitud
            min_length = min(len(chord_wave), len(arpeggio), len(bassline), len(rhythm), len(melody), end - start)
            combined = (chord_wave[:min_length].astype(np.float64) * 0.5 +
                        arpeggio[:min_length].astype(np.float64) * 0.5 +
                        bassline[:min_length].astype(np.float64) * 0.5 +
                        rhythm[:min_length].astype(np.float64) * 1.0 +
                        melody[:min_length].astype(np.float64) * 0.7)
            
            section[start:start+min_length] += combined
        
        # Normalizar la sección
        max_val = np.max(np.abs(section))
        if max_val > 0:
            section /= max_val
        return (section * 32767).astype(np.int16)

    def generate_advanced_soundtrack(self, mood, length=4, complexity=2):
        soundtrack = np.array([], dtype=np.int16)
        for _ in range(length):
            section = self.generate_complex_section(mood, 8, complexity)
            soundtrack = np.concatenate((soundtrack, section))
        return np.column_stack((soundtrack, soundtrack))

    def play_mood(self, mood, volume=0.5, complexity=2):
        soundtrack = self.generate_advanced_soundtrack(mood, 2, complexity)
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
            old_volume = max(1 - (i / steps), 0)
            new_volume = min(i / steps, 1)
            self.channels[old_mood].set_volume(old_volume)
            self.channels[new_mood].set_volume(new_volume)
            pygame.time.wait(transition_time // steps)

# Ejemplo de uso
if __name__ == "__main__":
    pygame.init()
    soundtrack = AdvancedSoundtrackGenerator()
    
    # Iniciar con música neutral
    soundtrack.play_mood("sad", complexity=2)
    
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