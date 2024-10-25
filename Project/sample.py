import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from IPython.display import display

#from main import EncoderNet, DecoderNet #, VAE

# Load the saved model
# model_path = 'vae.pth'  # Update with your model's file path if different
# model = torch.load(model_path)

from vae import GaussianEncoder, GaussianDecoder, VAE,  EncoderNet, DecoderNet
from priors import MixtureOfGaussiansPrior
import os
from datetime import datetime

latent_dim = 10

prior = MixtureOfGaussiansPrior(latent_dim, num_components=10)
encoder_net = EncoderNet(latent_dim=latent_dim)
decoder_net = DecoderNet(latent_dim=latent_dim)
encoder = GaussianEncoder(encoder_net)
decoder = GaussianDecoder(decoder_net)
model = VAE(prior, decoder, encoder)  # Define your model's architecture
#model.load_state_dict(torch.load('vae_state_dict.pth', weights_only=True))
model.load_state_dict(torch.load('vae_state_dict.pth', map_location=torch.device('cpu')))

model.eval()  # Set the model to evaluation model


# Directory where images will be saved
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Generate a single image and save it
with torch.no_grad():
    sample = model.sample(1).cpu()  # Generate one sample

# Convert the sample to a PIL image
image_pil = to_pil_image(sample.squeeze(0))  # Remove batch dimension if needed

# Generate a unique filename based on the current date and time
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'{output_dir}/generated_sample_{timestamp}.png'

# Save the image to a file
image_pil.save(filename)

print(f'Saved image: {filename}')