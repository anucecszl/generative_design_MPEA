import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

gpu = False
device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

data_df = pd.read_excel('../dataset/MPEA_parsed_dataset.xlsx')
# Transform the dataset to a numpy array
data_np = data_df.to_numpy()[:, :]

# Identify the element molar ratios of the alloys and perform normalization
comp_data = data_np[:, 14:46].astype(float)
comp_min = comp_data.min(axis=0)
comp_max = comp_data.max(axis=0)
minmax_comp = (comp_data - comp_min) / comp_max

# Concatenate it with the processing data to produce the feature array X
proc_data = data_np[:, 46:53].astype(float)
X = np.concatenate((minmax_comp, proc_data), axis=1)


class GANTrainSet(Dataset):
    def __init__(self):
        """Initialize the dataset by converting the data array to a PyTorch tensor."""
        self.features = torch.from_numpy(X).float()
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        """Retrieve a specific data item from the dataset using its index."""
        return self.features[index]

    def __len__(self):
        """Return the total number of data items in the dataset."""
        return self.len


class Generator(nn.Module):
    """
    Defines the Generator network within the Generative Adversarial Network (GAN).
    """

    def __init__(self):
        """Initialize the Generator model with fully connected layers and activation functions."""
        super(Generator, self).__init__()

        # The model consists of three layers with ReLU activation functions.
        # It aims to generate sparse non-zero outputs, mimicking realistic data.
        self.model = nn.Sequential(
            nn.Linear(10, 39),
            nn.ReLU(),
            nn.Linear(39, 39),
            nn.ReLU(),
            nn.Linear(39, 39),
            nn.ReLU(),
        )

    def forward(self, noise):
        """Execute a forward pass through the Generator network using the input noise."""
        fake_formula = self.model(noise)
        return fake_formula


class Discriminator(nn.Module):
    """
    Defines the Discriminator network within the Generative Adversarial Network (GAN).
    """

    def __init__(self):
        """Initialize the Discriminator model with fully connected layers and activation functions."""
        super(Discriminator, self).__init__()

        # The model consists of three layers with LeakyReLU activation functions.
        self.model = nn.Sequential(
            nn.Linear(39, 39),
            nn.LeakyReLU(),
            nn.Linear(39, 39),
            nn.LeakyReLU(),
            nn.Linear(39, 1),
        )

    def forward(self, x):
        """Execute a forward pass through the Discriminator network using the input data."""
        reality = self.model(x)
        return reality


def compute_gradient_penalty(d_net, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Parameters:
        d_net (nn.Module): Discriminator neural network.
        real_samples (Tensor): Real data samples.
        fake_samples (Tensor): Generated (fake) data samples.

    Returns:
        gp (Tensor): Computed gradient penalty.
    """

    # Generate random weight term alpha for interpolation between real and fake samples.
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1)))

    # Perform the interpolation between real and fake samples.
    # The interpolated sample 'interpolates' is set to require gradient to enable backpropagation.
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Evaluate the interpolated samples using the Discriminator network.
    d_interpolates = d_net(interpolates)

    # Generate a tensor 'fake' to be used as grad_outputs for gradient computation.
    # It is set to not require gradient to prevent it from affecting loss minimization.
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False

    # Compute gradients of the Discriminator outputs with respect to the interpolated samples.
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Reshape gradients to 2D tensor for ease of computation.
    gradients = gradients.view(gradients.size(0), -1)

    # Compute the gradient penalty based on the L2 norm of the gradients.
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gp


if __name__ == "__main__":
    # Hyperparameter for gradient penalty
    lambda_gp = 0.01

    # Initialize Generator and Discriminator models
    generator = Generator()
    discriminator = Discriminator()

    # Transfer the models to the appropriate computation device
    generator.to(device)
    discriminator.to(device)

    # Define the optimizers for both models
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)

    # Prepare the data loader
    al_data_set = GANTrainSet()
    loader = DataLoader(dataset=al_data_set, batch_size=5, shuffle=True)

    # Epochs for training
    for epoch in range(10000):
        # Initialize losses
        loss_d_real = 0
        loss_d_fake = 0
        total_d_loss = 0

        # Batch processing
        for i, alloy in enumerate(loader):
            real_input = alloy

            # Discriminator training loop
            for j in range(5):
                # Generate fake data
                g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
                fake_alloy = generator(g_noise)
                fake_input = fake_alloy

                optimizer_D.zero_grad()

                # Compute Gradient Penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_input.data, fake_input.data)

                # Calculate Discriminator Loss and Update Discriminator
                d_loss = -torch.mean(discriminator(real_input)) + torch.mean(
                    discriminator(fake_input.detach())) + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizer_D.step()

                # Update Discriminator Loss Stats
                r, d, l = discriminator(real_input).sum().item(), discriminator(fake_input).sum().item(), d_loss.item()
                loss_d_real += r
                loss_d_fake += d
                total_d_loss += -l

            # Generate fake data for Generator training
            g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
            fake_alloy = generator(g_noise)
            fake_input = fake_alloy

            # Generator Update
            optimizer_G.zero_grad()
            g_loss = -torch.mean(discriminator(fake_input))
            g_loss.backward()
            optimizer_G.step()

        # Periodic Reporting and Visualization
        if epoch % 50 == 0:
            g_noise = torch.tensor(np.random.randn(3, 10)).float()
            fake_alloy = generator(g_noise)
            print(fake_alloy)

        # Compute the balance of the Discriminator
        balance = loss_d_real / (loss_d_real + loss_d_fake)

        # Logging
        if epoch < 500 or epoch % 20 == 0:
            print(epoch, "Discriminator balance:", balance, "D_loss:", total_d_loss)

    # Save the trained Generator model
    torch.save(generator.state_dict(), '../saved_models/generator_net_MPEA.pt')

