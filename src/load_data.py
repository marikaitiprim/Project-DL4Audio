import os
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pretty_midi

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
        self.hop_length = 512
        self.win_length = 1024
        self.sample_rate = 16000
        self.mel_bins = 128

    def load_annotations(self, annotation_path):
        midi_data = pretty_midi.PrettyMIDI(annotation_path) #load midi file
        midi_events = []             # List to hold the events (start time, note pitch, velocity, end time)
    
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes: #extract the notes
                    midi_events.append([note.start,  note.end, note.pitch]) 

        return torch.tensor(midi_events, dtype=torch.float32)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        annotation_path = self.annotation_paths[idx]

        waveform, sr = torchaudio.load(audio_path) #original sr = 44100

        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform) #resample to 16000 Hz - (channels, time_steps)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.win_length, hop_length=self.hop_length)(waveform) #(channels, mel_bins, num_frames)
        annotations = self.load_annotations(annotation_path)  #(num_notes, 4)

        # print('mel spec ',mel_spec.shape)
        # print(annotations.shape)
        # print(waveform.shape)

        labels = torch.zeros((mel_spec.shape[2], mel_spec.shape[1]), dtype=torch.float32)
        for annotation in annotations:
            start_time, end_time, pitch = annotation
            start_idx = int(start_time * mel_spec.shape[2])
            end_idx = int(end_time * mel_spec.shape[2])
            labels[start_idx:end_idx, int(pitch)] = 1.0 #mark the notes of the melody in the time grid

        # grid = torch.zeros(waveform.shape[1]*mel_spec.shape[2]) #initialize grid 
        # print(grid.shape)

        # labels = torch.zeros((len(grid), 128), dtype=torch.float32) #initialize matrix to hold 0 and 1s

        # print(labels.shape)

        # # Mark melody notes in the time grid
        # for note in annotations:
        #     start_idx = torch.searchsorted(grid, note[0])  # Get the nearest time step using searchsorted equivalent in torch
        #     end_idx = torch.searchsorted(grid, note[1])
        #     pitch = int(note[2])
        #     labels[start_idx:end_idx, pitch] = 1.0  #-> killed
        
        return mel_spec, annotations


def load_data(audio_dir, annotation_dir, batch_size=1):

    # Create dataset
    audio_paths, annotation_paths = create_dataset(audio_dir, annotation_dir)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(audio_paths, annotation_paths, test_size=0.2, random_state=42)

    # Create training and test datasets
    train_dataset = MelExtractDataset(X_train, y_train)
    test_dataset = MelExtractDataset(X_test, y_test)

    # print(train_dataset[0])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


# if __name__ == "__main__":
#     audio_dir = "/Users/marikaitiprimenta/Desktop/Deep Learning in Music/project/Project-DL4Audio/Orchset/audio/mono"
#     annotation_dir = "/Users/marikaitiprimenta/Desktop/Deep Learning in Music/project/Project-DL4Audio/Orchset/midi"

#     train, test = load_data(audio_dir, annotation_dir)