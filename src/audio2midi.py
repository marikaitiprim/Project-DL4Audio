import torch 
import torch.nn as nn
import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class LSTMMidi(nn.Module):
    def __init__(self, input_size=1025, hidden_size=2000, proj_size=500, output_size=601, 
                 num_layers=1, dropout=0.2, device='cuda'): 
        """
        LSTM-RNN for melody extraction with harmonic sum loss as described in the paper
        
        Args:
            input_size: dimension of input features 
            hidden_size: number of LSTM cells 
            proj_size: size of projection layer (500)
            output_size: number of pitch classes (600 pitch bins + 1 for no melody = 601)
            num_layers: number of LSTM layers
            dropout: dropout probability
            device: 'cuda' or 'cpu'
        """
        super(LSTMMidi, self).__init__()
        
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
        """Initialize weights uniformly between -0.05 and 0.05"""
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
    

def evaluate(model, val_loader, criterion):
    '''Evaluate the model on the validation set'''
    model.eval()
    
    total_frames = 0
    correct_pitch = 0  # For RPA
    correct_chroma = 0  # For RCA
    correct_voicing = 0  # For MDA
    num_batches = len(val_loader)
    epoch_loss = 0.0
    
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(val_loader):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            batch_inputs = batch_inputs.permute(0,2,1) #change the shape to (batch_size, time_steps, freq_bins)  
            
            outputs = model(batch_inputs)

            outputs = outputs.permute(0, 2, 1) 
            
            epoch_loss += criterion(outputs, batch_labels).item()
            
            pred_pitch = torch.argmax(outputs, dim=1)  # Shape: (Batch, freq_bins, Time_steps)
            
            total_frames += batch_labels.numel()
            
            # RPA - Raw Pitch Accuracy
            correct_pitch += torch.sum((pred_pitch == batch_labels)).item()
            
            # RCA - Raw Chroma Accuracy (ignoring octave errors)
            # Convert to chroma (pitch class)
            pred_chroma = pred_pitch % 12
            true_chroma = batch_labels % 12
            correct_chroma += torch.sum((pred_chroma == true_chroma) | 
                                         ((pred_pitch == 0) & (batch_labels == 0))).item()
            
            # MDA - Melody Detection Accuracy
            # Voicing detection (melody or no melody)
            pred_voicing = (pred_pitch > 0)
            true_voicing = (batch_labels > 0)
            correct_voicing += torch.sum(pred_voicing == true_voicing).item()
    
    # Calculate final metrics
    rpa = correct_pitch / total_frames
    rca = correct_chroma / total_frames
    mda = correct_voicing / total_frames
    epoch_loss /= num_batches  
    
    return epoch_loss, rpa, rca, mda


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
            batch_labels = batch_labels.to(device)  # (batch_size, time_steps)

            batch_inputs = batch_inputs.permute(0,2,1) #change the shape to (batch_size, time_steps, freq_bins)  

            outputs = model(batch_inputs)    
            outputs = outputs.permute(0, 2, 1) 

            # print(outputs.shape)   
            loss = criterion(outputs, batch_labels) #calculate loss

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

        # Evaluate the network on the validation data
        if (epoch + 1) % evaluate_every_n_epochs == 0:
            valid_loss, rpa, rca, mda = evaluate(model, valid_loader, criterion)
            print(f"Validation Loss: {valid_loss:.6f}")
            print(f"Validation RPA: {100 * rpa:.2f}% | RCA: {100 * rca:.2f}% | MDA: {100 * mda:.2f}%")
            valid_losses.append(valid_loss)
            valid_accuracies.append(rpa)

        if(rpa >= best_valid_acc): #get the best model based on RPA
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

    print("Using ", device , ":")
    
    audio_dir = "./Orchset/audio/mono"  #change to the path of your audio data
    annotation_dir = "./Orchset/midi" #change to the path of your annotation data

    train_loader, test_loader = load_data.load_data(audio_dir, annotation_dir, batch_size=16) 

    model = LSTMMidi().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

    train_losses, valid_losses, valid_accuracies = train(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, saved_model='best_model.pth')
    plot_metrics(train_losses, valid_losses, valid_accuracies)