import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, log_var = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(0.5 * log_var) + 1e-6  # Make sure the std is not zero by adding a small epsilon
        return td.Independent(td.Normal(loc=mean, scale=std), 1) # log_prob not calculated yet.
        # # Without td.Independent, each dimension would be treated independently, and you'd get separate log-probabilities or samples for each dimension.
        # # This code's primary role is to return a probabilistic distribution that can later be used for sampling or calculating the log-probability. ( # rsample deals with reparametrization later)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes a tensor of dimension `(batch_size, M)` as
           input, where M is the dimension of the latent space, and outputs a tensor
           of dimension `(batch_size, feature_dim1, feature_dim2, 3)`.
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)
        std = torch.ones_like(mean) * 0.1 
        return td.Independent(td.Normal(loc=mean, scale=std), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        #elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        # elbo1 = self.decoder(z).log_prob(x)
        # elbo2 = td.kl_divergence(q, self.prior())
        # elbo = torch.mean(elbo1-elbo2, dim=0)
        RE = self.decoder(z).log_prob(x)
        KL = q.log_prob(z) - self.prior().log_prob(z)
        elbo = (RE - KL).mean()
        return elbo        



    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

        import torch.nn.functional as F

class EncoderNet(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # output size: 64x128x256
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # output size: 128x64x128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # output size: 256x32x64

        # Adjusting based on the convolutional output size
        self.fc = nn.Linear(256 * 32 * 64, latent_dim * 2)  # output size of conv3 is 256*32*64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)

class DecoderNet(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderNet, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 32 * 64)  # Adjust based on the final feature map size

        # Three deconvolution layers to upscale to 256x512
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 64x128
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 128x256
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)     # 256x512

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 256, 32, 64)  # Reshape to match the feature map size before deconvolutions
        z = F.relu(self.deconv1(z))         # 256 -> 128
        z = F.relu(self.deconv2(z))         # 128 -> 256
        z = torch.sigmoid(self.deconv3(z))  # 256 -> 512
        return z