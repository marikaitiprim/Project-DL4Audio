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

class ExtractDataset(Dataset):

    def __init__(self, audio_paths, annotation_paths):
        self.audio_paths, self.annotation_paths = audio_paths, annotation_paths
        self.hop_length = 160
        self.nfft = 2048
        self.win_length = 768
        self.sample_rate = 16000
        self.max_length = int(self.sample_rate * 9)  # 9 seconds for each audio file


    def spectrogram(self, audio_path):
        '''
        Create the spectrogram for the audio file using torchaudio
        args: path to the audio file
        return: spectrogram with shape (channels, freq_bins, num_frames)
        '''

        waveform, sr = torchaudio.load(audio_path) #original sr = 44100
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform) #resample to 16000 Hz - (channels, time_steps)
        waveform = waveform[:, :self.max_length]    #cut the audio to 9 seconds
        
        spec = torchaudio.transforms.Spectrogram(n_fft=self.nfft, hop_length=self.hop_length, win_length=self.win_length)(waveform)

        # Apply log compression (helps with dynamic range)
        spec = torch.log1p(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8) #normalization
        
        return spec 


    def load_annotations(self, annotation_path):
        '''
        Load ground truth and extract the start, end times and pitch of the melody notes
        args: path to the midi file
        return: a tensor with the shape (num_notes, 3) where each note is represented by (start_time, end_time, pitch)
        '''

        midi_data = pretty_midi.PrettyMIDI(annotation_path) #load midi file
        midi_events = []             # List to hold the events (start time, note pitch, velocity, end time)
    
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes: #extract the notes
                    midi_events.append([note.start,  note.end, note.pitch]) 
        
        annotations = torch.tensor(midi_events, dtype=torch.float32)
        annotations = annotations[annotations[:, 0] < 9] # Filter annotations to only include notes that start within the first 9 seconds

        return annotations #(num_notes, 3) (15, 3)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        annotation_path = self.annotation_paths[idx]

        spec = self.spectrogram(audio_path).squeeze(0)
        annotations = self.load_annotations(annotation_path)  

        time_grid = librosa.times_like(spec, sr=self.sample_rate, hop_length=self.hop_length)  # Generate time grid
        labels = torch.zeros((601, spec.shape[1]), dtype=torch.float32) # (freq_bins, num_frames)

        for note in annotations:
            start_idx = np.searchsorted(time_grid, note[0])
            end_idx = np.searchsorted(time_grid, note[1])
            if start_idx < len(time_grid) and end_idx <= len(time_grid):
                labels[int(note[2]), start_idx:end_idx] = 1.0

        labels = torch.argmax(labels, dim=0).long()  # Shape: (num_frames, time_steps)
        
        return spec, labels


def load_data(audio_dir, annotation_dir, batch_size=8):
    '''Create the dataset and dataloaders for training and testing'''

    # Create dataset
    audio_paths, annotation_paths = create_dataset(audio_dir, annotation_dir)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(audio_paths, annotation_paths, test_size=0.2, random_state=42)

    # Create training and test datasets
    train_dataset = ExtractDataset(X_train, y_train)
    test_dataset = ExtractDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader