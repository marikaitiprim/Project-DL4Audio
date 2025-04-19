import torch 
import torch.nn as nn
import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# class CNNMidi(nn.Module):
#     def __init__(self, num_classes=128, hidden_size=256, num_layers=2):
#         super(CNNMidi, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
#         # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         # Adaptive Pooling to reduce frequency dimension to 1
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
#         # LSTM for temporal modeling
#         self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
#         # Fully connected layer to predict 128 MIDI classes
#         self.fc = nn.Linear(hidden_size * 2, num_classes) # Bidirectional LSTM has 2x hidden size

#     def forward(self, x):
#         # Convolutional blocks
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
        
#         # Global average pooling over frequency axis
#         x = self.global_avg_pool(x)
#         x = x.squeeze(2)  # Shape: (Batch, 128, Time)
#         x = x.permute(0, 2, 1)  # Shape: (Batch, Time, 128)
        
#         # LSTM for temporal context
#         x, _ = self.lstm(x)  # Output shape: (Batch, Time, Hidden*2)
        
#         # Predict per time step
#         x = self.fc(x)  # Shape: (Batch, Time, 128)

#         x = x.permute(0, 2, 1) # Shape: (Batch, 128, Time)
        
#         return x

class CNNMidi(nn.Module):
    def __init__(self, input_size=1025, hidden_size=2000, proj_size=500, output_size=601, 
                 num_layers=1, dropout=0.2, device='cuda'):
        """
        LSTM-RNN for melody extraction with harmonic sum loss as described in the paper
        
        Args:
            input_size: dimension of input features (1025 from STFT as mentioned in paper)
            hidden_size: number of LSTM cells (2000 as per paper)
            proj_size: size of projection layer (500 as per paper)
            output_size: number of pitch classes (600 pitch bins + 1 for no melody = 601)
            num_layers: number of LSTM layers
            dropout: dropout probability
            device: 'cuda' or 'cpu'
        """
        super(CNNMidi, self).__init__()
        
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.output_size = output_size
        self.device = device
        self.projection = nn.Linear(hidden_size, proj_size)
        
        # Using LSTM with projection as described in the paper
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Output layer (projection to pitch classes)
        self.fc = nn.Linear(proj_size, output_size)
        
        # Initialize weights (as mentioned in paper: uniform [-0.05, 0.05])
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights uniformly between -0.05 and 0.05 as specified in the paper"""
        for param in self.parameters():
            nn.init.uniform_(param, -0.05, 0.05)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input sequence [batch_size, seq_len, input_size]
        
        Returns:
            y_pred: pitch prediction probabilities [batch_size, seq_len, output_size]
        """
        # LSTM with projection layer
        output, _ = self.lstm(x)
        output = self.projection(output)
        
        # Apply fully connected layer
        y_pred = self.fc(output)
        
        return y_pred
    

def evaluate(model, data_loader, criterion):
    model.eval()
    
    total_frames = 0
    correct_pitch = 0  # For RPA
    correct_chroma = 0  # For RCA
    correct_voicing = 0  # For MDA
    total_loss = 0.0  # To accumulate the loss
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x = batch_x.squeeze(1).permute(0,2,1) #change the shape to (batch_size, time_steps, freq_bins)  
            
            # Forward pass
            y_pred = model(batch_x)
            
            # Calculate loss
            loss = criterion(y_pred, batch_y)
            total_loss += loss.item()
            
            # Get predictions
            pred_pitch = torch.argmax(y_pred, dim=1)  # Shape: (Batch, Time)
            
            # Calculate metrics
            total_frames += batch_y.numel()
            
            # RPA - Raw Pitch Accuracy
            correct_pitch += torch.sum((pred_pitch == batch_y)).item()
            
            # RCA - Raw Chroma Accuracy (ignoring octave errors)
            # Convert to chroma (pitch class)
            pred_chroma = pred_pitch % 12
            true_chroma = batch_y % 12
            correct_chroma += torch.sum((pred_chroma == true_chroma) | 
                                         ((pred_pitch == 0) & (batch_y == 0))).item()
            
            # MDA - Melody Detection Accuracy
            # Voicing detection (melody or no melody)
            pred_voicing = (pred_pitch > 0)
            true_voicing = (batch_y > 0)
            correct_voicing += torch.sum(pred_voicing == true_voicing).item()
    
    # Calculate final metrics
    rpa = correct_pitch / total_frames
    rca = correct_chroma / total_frames
    mda = correct_voicing / total_frames
    avg_loss = total_loss / num_batches  # Average loss over all batches
    
    # print(f"RPA (Raw Pitch Accuracy): {rpa:.4f}")
    # print(f"RCA (Raw Chroma Accuracy): {rca:.4f}")
    # print(f"MDA (Melody Detection Accuracy): {mda:.4f}")
    # print(f"Validation Loss: {avg_loss:.4f}")
    
    return avg_loss, rpa, rca, mda


# def evaluate(model, data_loader, criterion):
#     model.eval()
#     num_batches = len(data_loader)
#     epoch_loss = 0.
#     precision = 0.
#     recall = 0.
#     f1_accuracy = 0.

#     with torch.no_grad():
#         all_predictions = []
#         all_labels = []
#         for batch_inputs, batch_labels in data_loader:
#             batch_inputs = batch_inputs.to(device)
#             batch_labels = batch_labels.to(device)

#             batch_inputs = batch_inputs.squeeze(1) #remove the channel dimension
#             batch_inputs = batch_inputs.permute(0,2,1) #change the shape to (batch_size, time_steps, freq_bins)  

#             batch_outputs = model(batch_inputs)

#             # Convert outputs and labels to integer class indices
#             batch_binary_indices = torch.argmax(batch_outputs, dim=1)  # Shape: (Batch, Time)
    
#             # Collect predictions and labels for metrics
#             all_predictions.append(batch_binary_indices.cpu().numpy().flatten())
#             all_labels.append(batch_labels.cpu().numpy().flatten())
           
#             epoch_loss += criterion(batch_outputs, batch_labels).item()

#     # Concatenate all predictions and labels
#     all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (Total_Time,)
#     all_labels = np.concatenate(all_labels, axis=0)  # Shape: (Total_Time,)

#     # Calculate metrics
#     precision = precision_score(all_labels, all_predictions, average='macro')
#     recall = recall_score(all_labels, all_predictions, average='macro')
#     f1_accuracy = f1_score(all_labels, all_predictions, average='macro')
           
#     epoch_loss /= num_batches

#     return epoch_loss, precision, recall, f1_accuracy


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_model, evaluate_every_n_epochs=1):
    model.train()
    num_batches = len(train_loader)
    best_valid_acc = 0.0
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_inputs, batch_labels in tqdm(train_loader):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)  

            batch_inputs = batch_inputs.squeeze(1).permute(0,2,1) #change the shape to (batch_size, time_steps, freq_bins)   

            outputs = model(batch_inputs)    

            # print(outputs.shape)   
            loss = criterion(outputs, batch_labels) #calculate loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        epoch_loss /= num_batches

        # scheduler.step()

        # print training loss
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        train_losses.append(epoch_loss)
        
        # evaluate the network on the validation data
        # if((epoch+1) % evaluate_every_n_epochs == 0):
        #     valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion)
        #     print(f'Validation loss: {valid_loss:.6f}')
        #     print(f'Validation Precision: {100*valid_precision:.2f}% | Recall: {100*valid_recall:.2f}% | F1: {100*valid_f1:.2f}%')
        #     valid_losses.append(valid_loss)
        #     valid_accuracies.append(valid_f1)

        # Evaluate the network on the validation data
        if (epoch + 1) % evaluate_every_n_epochs == 0:
            valid_loss, rpa, rca, mda = evaluate(model, valid_loader, criterion)
            print(f"Validation Loss: {valid_loss:.6f}")
            print(f"Validation RPA: {100 * rpa:.2f}% | RCA: {100 * rca:.2f}% | MDA: {100 * mda:.2f}%")
            valid_losses.append(valid_loss)
            valid_accuracies.append(rpa)
            
        # if the best validation performance so far, save the network to file 
        # if(valid_f1 >= best_valid_acc):
        #     best_valid_acc = valid_f1
        #     print('Saving best model')
        #     torch.save(model.state_dict(), saved_model)

        if(rpa >= best_valid_acc):
            best_valid_acc = rpa
            print('Saving best model')
            torch.save(model.state_dict(), saved_model)

    return train_losses, valid_losses, valid_accuracies

def plot_metrics(train_losses, valid_losses, valid_accuracies):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_tight_layout(True)
    ax1.plot(train_losses, label='Training')
    ax1.plot(valid_losses, label='Validation')
    ax1.set_xlabel('epochs')
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(valid_accuracies)
    ax2.set_xlabel('epochs')
    ax2.set_title('Validation Accuracy')
    plt.show()
    plt.savefig("best_model_metrics.png")

if __name__ == '__main__':

    # use GPU if available, otherwise, use CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using ", device , ":")

    audio_dir = "./Orchset/audio/mono"  #change to the path of your audio data
    annotation_dir = "./Orchset/midi" #change to the path of your annotation data

    train_loader, test_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=8) 

    model = CNNMidi().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

    train_losses, valid_losses, valid_accuracies = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, saved_model='best_model.pth')
    plot_metrics(train_losses, valid_losses, valid_accuracies)