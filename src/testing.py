from audio2midi import CNNMidi
import torch
import torchaudio
import pretty_midi
import matplotlib.pyplot as plt
import subprocess

def save_wav(midi_filepath, output_filepath='output.wav'):
    wav_filepath = output_filepath
    process = subprocess.Popen(f"fluidsynth soundfont.sf -g 1.0 -r 44100 --quiet --no-shell {midi_filepath} -T wav -F {wav_filepath} > /dev/null", shell=True)
    process.wait()
    return wav_filepath

def spectrogram(audio_path):
    '''
    Create the mel spectrogram for the audio file using torchaudio
    args: path to the audio file
    return: mel spectrogram with shape (channels, mel_bins, num_frames)
    '''

    hop_length = 160
    nfft = 2048
    win_length = 768
    sample_rate = 16000

    waveform, sr = torchaudio.load(audio_path) #original sr = 44100
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform) #resample to 16000 Hz - (channels, time_steps)

    # mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.win_length, hop_length=self.hop_length, n_mels=self.mel_bins, win_length=self.win_length)(waveform) 
    
    spec = torchaudio.transforms.Spectrogram(n_fft=nfft, hop_length=hop_length, win_length=win_length)(waveform)

    # Apply log compression (helps with dynamic range)
    spec = torch.log1p(spec)
    spec = (spec - spec.mean()) / (spec.std() + 1e-8) #normalization
    
    return spec 

def convert_model_output_to_midi(predictions, fs=44100, hop_length=512, output_file='output.mid'):
    """
    Convert model predictions to MIDI file
    
    Args:
        predictions: numpy array of pitch predictions
        fs: sample rate of original audio
        hop_length: hop length used in feature extraction
        output_file: output MIDI file path
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an Instrument instance
    instrument = pretty_midi.Instrument(program=0)  # Program 0 is acoustic grand piano
    
    # Convert frame indices to time in seconds
    time_per_frame = hop_length / fs
    
    # Process the predictions
    current_note = None
    note_start_time = 0
    
    for i, pitch in enumerate(predictions):
        time_in_seconds = i * time_per_frame
        
        # Special values handling
        if pitch == 0:  # No melody
            # End current note if there was one
            if current_note is not None:
                note = pretty_midi.Note(
                    velocity=100,  # Default velocity
                    pitch=current_note,
                    start=note_start_time,
                    end=time_in_seconds
                )
                instrument.notes.append(note)
                current_note = None
        elif pitch == 1402 or pitch == 2353:  # Special values in your data
            # Handle special values (these might be specific to your model)
            # For example, if these are special pitch classes or articulations
            # You might want to customize this based on your specific model's output
            if pitch == 1402:
                midi_pitch = 80  # Arbitrary high note for special value
            else:
                midi_pitch = 84  # Different arbitrary note for other special value
                
            # Add as short notes
            note = pretty_midi.Note(
                velocity=100,
                pitch=midi_pitch,
                start=time_in_seconds,
                end=time_in_seconds + time_per_frame
            )
            instrument.notes.append(note)
        else:
            # Regular MIDI notes
            # If it's a continuation of the same note, extend it
            if current_note == pitch:
                continue
                
            # End previous note if there was one
            if current_note is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=current_note,
                    start=note_start_time,
                    end=time_in_seconds
                )
                instrument.notes.append(note)
            
            # Start new note
            current_note = pitch
            note_start_time = time_in_seconds
    
    # Add the final note if needed
    if current_note is not None:
        note = pretty_midi.Note(
            velocity=100,
            pitch=current_note,
            start=note_start_time,
            end=len(predictions) * time_per_frame
        )
        instrument.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    
    # Write out the MIDI data
    midi.write(output_file)
    
    return midi


# Optional: visualize the notes
def plot_piano_roll(midi_file, start_time=0, end_time=10):
    """Plot a piano roll visualization of the MIDI file"""
    plt.figure(figsize=(12, 6))
    for instrument in midi_file.instruments:
        for note in instrument.notes:
            if start_time <= note.start <= end_time:
                plt.plot([note.start, note.end], [note.pitch, note.pitch], linewidth=5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pitch')
    plt.title('Piano Roll')
    plt.grid(True)
    plt.show()



def test_audio(audio_path, checkpoint_path, device='cpu'):

    model = CNNMidi().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    spec = spectrogram(audio_path)
    spec = spec.squeeze(1).permute(0,2,1).to(device) #change the shape to (batch_size, time_steps, freq_bins)

    # Forward pass
    with torch.no_grad():
        outputs = model(spec)  # Shape: (1, time_frames, num_classes)
    
    # Get predicted pitch indices
    pred_pitch = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()  # Shape: (time_frames,)
    return pred_pitch

if __name__ == '__main__':

    # use GPU if available, otherwise, use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using ", device , ":")

    audio_path = "./Strauss-BlueDanube-ex1.wav"
    checkpoint_path = "./best_model.pth"
    output_midi_path = "./output.mid"
    output_wav_path = "./output.wav"

    # Test the audio file
    predictions = test_audio(audio_path, checkpoint_path, device=device)

    print(predictions)

    # Convert to MIDI
    midi_file = convert_model_output_to_midi(predictions)

   # Visualize the MIDI
    plot_piano_roll(midi_file)

    save_wav(output_midi_path)