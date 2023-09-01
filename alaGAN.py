import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Since this is a small neural net, using cpu to run the model is faster.
gpu = False
device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

data_df = pd.read_excel('datasets/valid_Al_mechanical.xlsx')
# transform the dataset to a numpy array
data_np = data_df.to_numpy()

# identify the features of the alloys.
comp_data = data_np[:, 5:40].astype(float)
comp_min = np.min(comp_data.astype(float), axis=0)
comp_max = np.max(comp_data.astype(float), axis=0)

X = (comp_data - comp_min) / comp_max


class GANTrainSet(Dataset):
    def __init__(self):
        self.features = torch.from_numpy(X).float()
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.len


class Generator(nn.Module):
    """
    Build the Generator network
    """

    def __init__(self):
        super(Generator, self).__init__()

        # Set the model to have two latent layers both with LeakyReLU activation
        # The output layer used ReLU activation to get a sparse non-zero outputs
        self.model = nn.Sequential(
            nn.Linear(10, 35),
            nn.ReLU(),
            nn.Linear(35, 35),
            nn.ReLU(),
            nn.Linear(35, 35),
            nn.ReLU(),
        )

    def forward(self, noise):
        fake_formula = self.model(noise)
        return fake_formula


class Discriminator(nn.Module):
    """
    Build the Discriminator network
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Set the model to have two latent layers both with LeakyReLU activation
        self.model = nn.Sequential(
            nn.Linear(35, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 1),
        )

    def forward(self, x):
        reality = self.model(x)
        return reality


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == "__main__":
    lambda_gp = 0.0075

    generator = Generator()
    discriminator = Discriminator()

    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=1e-4, )
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4, )

    al_data_set = GANTrainSet()
    loader = DataLoader(dataset=al_data_set, batch_size=5, shuffle=True, )

    for epoch in range(10000):
        loss_d_real = 0
        loss_d_fake = 0
        total_d_loss = 0

        for i, alloy in enumerate(loader):
            real_input = alloy

            for j in range(5):
                # generate a batch of fake alloys and train the discriminator model.
                g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
                fake_alloy = generator(g_noise)
                fake_input = fake_alloy

                optimizer_D.zero_grad()
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_input.data, fake_input.data)

                d_loss = -torch.mean(discriminator(real_input)) + torch.mean(
                    discriminator(fake_input.detach())) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                r, d, l = discriminator(real_input).sum().item(), discriminator(fake_input).sum().item(), d_loss.item()
                loss_d_real += r
                loss_d_fake += d
                total_d_loss += -l

            # generate a batch of fake alloys and train the generator model.
            g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
            fake_alloy = generator(g_noise)
            fake_input = fake_alloy

            # calculate the loss function of generator and optimize.
            optimizer_G.zero_grad()
            g_loss = -torch.mean(discriminator(fake_input))
            g_loss.backward()
            optimizer_G.step()
        # print 3 alloys to show how does the model work in each epoch
        if epoch % 20 == 0:
            g_noise = torch.tensor(np.random.randn(6, 10)).float()
            fake_alloy = generator(g_noise)
            show_np = fake_alloy.detach().numpy()[:, -10:]
            print(np.sum(show_np, axis=1))
            print(show_np)
            print()
        balance = loss_d_real / (loss_d_real + loss_d_fake)

        if epoch < 5000 or epoch % 20 == 0:
            print(epoch, "discriminator balance:", balance, "d_loss:", total_d_loss)

        if epoch > 0 and epoch % 999 == 0:
            g_noise = torch.tensor(np.random.randn(1000, 10)).float()
            fake_alloy = generator(g_noise)
            show_np = fake_alloy.detach().numpy()[:, -10:]
            print('mean:', np.mean(np.sum(show_np, axis=1)))
            print('std:', np.std(np.sum(show_np, axis=1)))
    # after training, save the GAN generator model
    torch.save(generator.state_dict(), 'saved_models/generator_net_aluminium_gp.pt')

