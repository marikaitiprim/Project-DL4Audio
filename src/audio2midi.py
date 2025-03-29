import torch 
import torch.nn as nn
import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class CNNMidi(nn.Module):
    def __init__(self, num_classes=128, hidden_size=256, num_layers=2):
        super(CNNMidi, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Adaptive Pooling to reduce frequency dimension to 1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layer to predict 128 MIDI classes
        self.fc = nn.Linear(hidden_size * 2, num_classes) # Bidirectional LSTM has 2x hidden size

    def forward(self, x):
        # Convolutional blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Global average pooling over frequency axis
        x = self.global_avg_pool(x)
        x = x.squeeze(2)  # Shape: (Batch, 128, Time)
        x = x.permute(0, 2, 1)  # Shape: (Batch, Time, 128)
        
        # LSTM for temporal context
        x, _ = self.lstm(x)  # Output shape: (Batch, Time, Hidden*2)
        
        # Predict per time step
        x = self.fc(x)  # Shape: (Batch, Time, 128)
        
        return x

def peak_picking(batch_outputs, device):        #function for post-processing. Peak picking for converting the probabilities of the output of the model into binary representation
    batch_outputs_cpu = batch_outputs.cpu().numpy()
    batch_binary_outputs = np.zeros(batch_outputs_cpu.shape)  # Initialize with all zeros

    for frame_set_id in range(batch_outputs_cpu.shape[0]): #for every frame set in the batch
        detected_peaks_indices, _ = scipy.signal.find_peaks(batch_outputs_cpu[frame_set_id, :], distance=30) #post-processing peak picking for extracting the beats of the set
        batch_binary_outputs[frame_set_id, detected_peaks_indices] = 1  # Set beats to 1 for the specific set

    return torch.tensor(batch_binary_outputs, dtype=torch.float32).to(device)
    
def calculate_precision(true_positives, false_positives):
    precision = 0.
    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    return precision

def calculate_recall(true_positives, false_negatives):
    recall = 0.
    if true_positives + false_negatives != 0:
        recall = true_positives / (true_positives + false_negatives)
    return recall

def calculate_f1(precision, recall):
    f1_accuracy = 0.
    if precision + recall != 0:
        f1_accuracy = 2*precision*recall / (precision + recall)
    return f1_accuracy

def evaluate(model, data_loader, criterion):
    model.eval()
    num_batches = len(data_loader)
    epoch_loss = 0.
    precision = 0.
    recall = 0.
    f1_accuracy = 0.
    true_positives = 0.
    false_positives = 0.
    false_negatives = 0.
    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            batch_outputs = model(batch_inputs)

            batch_binary_outputs = torch.argmax(batch_outputs, dim=2)  # Shape: (Batch, Time) 
        
            true_positives += ((batch_binary_outputs == batch_labels) & (batch_binary_outputs == 1)).sum().item() #All beat predictions - TP
            false_positives += ((batch_binary_outputs != batch_labels) & (batch_binary_outputs == 1)).sum().item() #FP
            false_negatives += ((batch_binary_outputs != batch_labels) & (batch_binary_outputs == 0)).sum().item() #FN
            epoch_loss += criterion(batch_outputs.view(-1, 128), batch_labels.view(-1)).item()
           
    epoch_loss /= num_batches

    precision = calculate_precision(true_positives=true_positives, false_positives=false_positives)
    recall = calculate_recall(true_positives=true_positives, false_negatives=false_negatives)
    f1_accuracy = calculate_f1(precision=precision, recall=recall)

    return epoch_loss, precision, recall, f1_accuracy

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

            # import pdb; pdb.set_trace()           

            outputs = model(batch_inputs)        

            # Convert binary labels to class indices
            labels_class = torch.argmax(outputs, dim=2)  # Shape: (Batch, Time) 

            # Calculate loss
            loss = criterion(outputs.view(-1, 128), labels_class.view(-1)) #softmax

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        epoch_loss /= num_batches

        scheduler.step()

        # print training loss
        print(f'[{epoch+1}] loss: {epoch_loss:.6f}')
        train_losses.append(epoch_loss)
        
        # evaluate the network on the validation data
        if((epoch+1) % evaluate_every_n_epochs == 0):
            valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion)
            print(f'Validation loss: {valid_loss:.6f}')
            print(f'Validation Precision: {100*valid_precision:.2f}% | Recall: {100*valid_recall:.2f}% | F1: {100*valid_f1:.2f}%')
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_f1)
            
            # if the best validation performance so far, save the network to file 
            if(valid_f1 >= best_valid_acc):
                best_valid_acc = valid_f1
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

    train_loader, test_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=1) 

    model = CNNMidi().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs


    train_losses, valid_losses, valid_accuracies = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=2, saved_model='best_model.pth')
    plot_metrics(train_losses, valid_losses, valid_accuracies)