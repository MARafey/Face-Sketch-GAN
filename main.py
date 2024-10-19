import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.autograd import Variable
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=1):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.init_size = 64 // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        gen_input = torch.cat((noise, condition), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, condition_dim=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3 + condition_dim, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, condition):
        condition = condition.view(condition.size(0), 1, 64, 64)
        d_in = torch.cat((img, condition), 1)
        out = self.conv_blocks(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class FaceSketchDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.photo_dir = os.path.join(root_dir, split, 'photos')
        self.sketch_dir = os.path.join(root_dir, split, 'sketches')

        self.photos = sorted([f for f in os.listdir(self.photo_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        photo_name = self.photos[idx]
        photo_path = os.path.join(self.photo_dir, photo_name)
        sketch_path = os.path.join(self.sketch_dir, photo_name)

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('L')

        if self.transform:
            photo = self.transform(photo)
            sketch = self.transform(sketch)

        return sketch, photo


def save_model(generator, discriminator, epoch, optimizer_G, optimizer_D, path='checkpoints'):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))


def load_model(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    return checkpoint['epoch']


def train_cgan(generator, discriminator, dataloader, num_epochs, device, save_interval=10):
    # Loss functions
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    adversarial_loss = adversarial_loss.to(device)

    for epoch in range(num_epochs):
        for i, (sketches, real_imgs) in enumerate(dataloader):
            batch_size = real_imgs.size(0)

            # Configure input
            real_imgs = real_imgs.to(device)
            sketches = sketches.to(device)

            # Valid and fake labels
            valid = Variable(torch.ones(batch_size, 1), requires_grad=False).to(device)
            fake = Variable(torch.zeros(batch_size, 1), requires_grad=False).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and generate images
            z = Variable(torch.randn(batch_size, generator.latent_dim)).to(device)
            gen_imgs = generator(z, sketches)

            # Calculate loss and backpropagate
            g_loss = adversarial_loss(discriminator(gen_imgs, sketches), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Calculate loss for real images
            real_loss = adversarial_loss(discriminator(real_imgs, sketches), valid)

            # Calculate loss for fake images
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), sketches), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

                # Save sample images
                if i % 500 == 0:
                    vutils.save_image(gen_imgs.data[:25],
                                      f'images/epoch_{epoch}_batch_{i}.png',
                                      normalize=True,
                                      nrow=5)

        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            save_model(generator, discriminator, epoch + 1, optimizer_G, optimizer_D)


def generate_sketch(model_path, image_path, output_path, device='cuda'):
    # Initialize model
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load trained model
    load_model(generator, discriminator, optimizer_G, optimizer_D, model_path)
    generator.to(device)
    generator.eval()

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    # Generate sketch
    with torch.no_grad():
        z = torch.randn(1, generator.latent_dim).to(device)
        generated = generator(z, image)

    # Save generated image
    vutils.save_image(generated, output_path, normalize=True)
    return generated


def main():
    # Hyperparameters
    latent_dim = 100
    batch_size = 64
    num_epochs = 200
    image_size = 64
    root_dir = "archive"  # Update with your dataset path

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories for saving results
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    train_dataset = FaceSketchDataset(root_dir, split='train', transform=transform)
    val_dataset = FaceSketchDataset(root_dir, split='val', transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Initialize models
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    # Train the model
    train_cgan(generator, discriminator, train_loader, num_epochs, device)


if __name__ == "__main__":
    # For training
    main()

    # For inference
    # model_path = "checkpoints/checkpoint_epoch_100.pth"
    # input_image = "path/to/input/image.jpg"
    # output_path = "path/to/output/sketch.jpg"
    # generate_sketch(model_path, input_image, output_path)