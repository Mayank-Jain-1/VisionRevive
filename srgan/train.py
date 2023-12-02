import os
import cv2
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader import MyImageFolder
import config
from Srgan import Generator, Discriminator
from VGGLoss import VGGLoss

def main():
  dataset = MyImageFolder('./Dataset/DIV2K_train_HR')
  loader = DataLoader(
      dataset, 
      batch_size=config.BATCH_SIZE,
      shuffle=True,
      pin_memory=True,
      num_workers=config.NUM_WORKERS,
  )

  generator = Generator(num_channels=config.NUM_CHANNELS,
                  num_blocks=config.NUM_BLOCKS).to(config.DEVICE)
  discriminator = Discriminator().to(config.DEVICE)

  gen_opt = torch.optim.Adam(generator.parameters(), lr=config.GEN_LR, betas=(0.9, 0.0999))

  disc_opt = torch.optim.Adam(discriminator.parameters(), lr=config.DISC_LR, betas=(0.9, 0.0999))

  mse = nn.MSELoss()
  bce = nn.BCEWithLogitsLoss()
  # VGGLoss = VGGLoss()

  train_loop(loader,generator, discriminator, mse, bce, gen_opt, disc_opt)


def train_with_gan_vgg(loader, generator, discriminator,mse, bce, gen_opt, disc_opt, VGGLoss):
  loop = tqdm(loader, leave=True)

  for idx,(hr_img, lr_img) in enumerate(loop):
    hr_img = hr_img.to(config.DEVICE)
    lr_img = lr_img.to(config.DEVICE)

    # Creating the Output 4x Higher res image
    generated = generator(lr_img)

    # Dicriminator Optimization

    # Discriminator Loss on Real HR IMage
    disc_real = discriminator(hr_img)
    disc_loss_real = bce(
        disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
    )

    # Discriminator Loss on Generated Image
    disc_fake = discriminator(generated.detach())
    disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))

    # Calculating total loss and optimizing
    total_loss_disc = disc_loss_fake + disc_loss_real

    disc_opt.zero_grad()
    total_loss_disc.backward()
    disc_opt.step()

    # Generator Optimization 

    disc_fake= discriminator(generated)
    l2_loss = mse(generated, hr_img)
    adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
    vggloss = 0.006 * VGGLoss(generated, hr_img)
    total_gen_loss = vggloss + adversarial_loss 

    gen_opt.zero_grad()
    total_gen_loss.backward()
    gen_opt.step()





    x = input("------------------------>")

def train_with_mse(loader, generator, discriminator,mse, bce, gen_opt, disc_opt, VGGLoss):
  loop = tqdm(loader, leave=True)

  for idx,(hr_img, lr_img) in enumerate(loop):
    hr_img = hr_img.to(config.DEVICE)
    lr_img = lr_img.to(config.DEVICE)

    # Creating the Output 4x Higher res image
    generated = generator(lr_img)

    # Generator Optimization 
    l2_loss = mse(generated, hr_img)
    total_gen_loss = l2_loss 

    gen_opt.zero_grad()
    total_gen_loss.backward()
    gen_opt.step()

    



if __name__ == '__main__':
  main()