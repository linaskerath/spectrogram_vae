from torchvision import  transforms # datasets,
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn

from tqdm import tqdm


##########################################################
def train(model, optimizer, data_loader, epochs, device, name):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
        
        torch.save(model, name + '.pth')
        torch.save(model.state_dict(), name + '_state_dict.pth') # state_dict
    progress_bar.close()

##########################################################
from vae import GaussianEncoder, GaussianDecoder, VAE, EncoderNet, DecoderNet 
from priors import MixtureOfGaussiansPrior

##########################################################
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),  # Convert the image to a tensor with values in [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] for each RGB channel
])

#dataset = ImageFolder(root='data/spectrogram_images', transform=transform)
#dataset = ImageFolder(root='/scratch/linsk/spectrograms/data/temp_specs/', transform=transform)
dataset = ImageFolder(root='/scratch/linsk/spectrograms/data/spectrograms/images/', transform=transform)
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
##########################################################


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 10  # Set your desired latent space dimension
encoder_net = EncoderNet(latent_dim=latent_dim)
decoder_net = DecoderNet(latent_dim=latent_dim)
#prior = GaussianPrior(latent_dim)
prior = MixtureOfGaussiansPrior(latent_dim, num_components=10)

# Initialize encoder, decoder, and model
encoder = GaussianEncoder(encoder_net)
decoder = GaussianDecoder(decoder_net)
model = VAE(prior, decoder, encoder).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

# Train model
epochs = 100
train(model, optimizer, train_loader, epochs, device, name = 'vae')       
