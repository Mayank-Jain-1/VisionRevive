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

  generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.GEN_LR, betas=(0.9, 0.0999))

  discriminator_optmizer = torch.optim.Adam(discriminator.parameters(), lr=config.DISC_LR, betas=(0.9, 0.0999))

  mse = nn.MSELoss()
  bce = nn.BCEWithLogitsLoss()
  # vgg_loss = VGGLoss()

  train_loop(loader)


def train_loop(loader):
  loop = tqdm(loader, leave=True)

  for idx,(hr_img, lr_img) in enumerate(loop):
    cv2.imwrite('./Testing/lr_image.png', lr_img[0].numpy())
    cv2.imwrite('./Testing/hr_image.png', hr_img[0].numpy())

    x = input("enter somethign for next")

if __name__ == '__main__':
  main()