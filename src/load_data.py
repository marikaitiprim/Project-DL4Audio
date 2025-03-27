import os
import torch 
import librosa
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pretty_midi
import numpy as np

def create_dataset(audio_dir, annotation_dir):      #pair paths from audio and annotations
    audio_paths = []
    annotation_paths = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                annotation_path = os.path.join(annotation_dir, file.replace('.wav', '.mid'))
                if os.path.exists(annotation_path):
                    audio_paths.append(audio_path)
                    annotation_paths.append(annotation_path)

    return audio_paths, annotation_paths

class MelExtractDataset(Dataset):

    def __init__(self, audio_paths, annotation_paths):
        self.audio_paths, self.annotation_paths = audio_paths, annotation_paths
        self.hop_length =512
        self.win_length = 2048
        self.sample_rate = 22050
        self.mel_bins = 128
        self.max_length = int(self.sample_rate * 9)  # 9 seconds for each audio file


    def mel_spectrogram(self, audio_path):

        waveform, sr = torchaudio.load(audio_path) #original sr = 44100
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform) #resample to 22050 Hz - (channels, time_steps)
        waveform = waveform[:, :self.max_length]    #cut the audio to 9 seconds

        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.win_length, hop_length=self.hop_length, n_mels=self.mel_bins)(waveform) #(channels, mel_bins, num_frames)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
        return log_mel_spec #log_mel_spec[..., np.newaxis]


    def load_annotations(self, annotation_path):
        midi_data = pretty_midi.PrettyMIDI(annotation_path) #load midi file
        midi_events = []             # List to hold the events (start time, note pitch, velocity, end time)
    
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes: #extract the notes
                    midi_events.append([note.start,  note.end, note.pitch]) 
        
        annotations = torch.tensor(midi_events, dtype=torch.float32)
        annotations = annotations[annotations[:, 0] < 9] # Filter annotations to only include notes that start within the first 9 seconds

        return annotations

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        annotation_path = self.annotation_paths[idx]

        mel_spec = self.mel_spectrogram(audio_path) 
        annotations = self.load_annotations(annotation_path)  #(num_notes, 3)

        # print(audio_path)
        # print('mel spec ',mel_spec.shape)
        # print(annotations.shape)
        # print(waveform.shape)

        time_grid = librosa.times_like(mel_spec, sr=self.sample_rate, hop_length=self.hop_length)  # Generate time grid
        labels = torch.zeros((mel_spec.shape[1], len(time_grid)), dtype=torch.float32) 

        for note in annotations:
            start_idx = np.searchsorted(time_grid, note[0])
            end_idx = np.searchsorted(time_grid, note[1])
            if 0 <= note[2] < mel_spec.shape[1]:
                labels[int(note[2]), start_idx:end_idx] = 1.0
        
        return torch.tensor(mel_spec, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def load_data(audio_dir, annotation_dir, batch_size=1):

    # Create dataset
    audio_paths, annotation_paths = create_dataset(audio_dir, annotation_dir)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(audio_paths, annotation_paths, test_size=0.2, random_state=42)

    # Create training and test datasets
    train_dataset = MelExtractDataset(X_train, y_train)
    test_dataset = MelExtractDataset(X_test, y_test)

    # import pdb; pdb.set_trace()
    # print(train_dataset[0][1])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


# if __name__ == "__main__":
#     audio_dir = "./Orchset/audio/stereo"
#     annotation_dir = "./Orchset/midi"

#     train, test = load_data(audio_dir, annotation_dir)