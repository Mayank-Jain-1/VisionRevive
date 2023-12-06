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
import utils
import time

def main():
  dataset = MyImageFolder('.\Dataset\DIV2K_train_hr', '.\Dataset\DIV2K_train_lr')
  loader = DataLoader(
      dataset, 
      batch_size=config.BATCH_SIZE,
      shuffle=True,
      pin_memory=True,
      num_workers=config.NUM_WORKERS,
  )
  
  torch.manual_seed(49)

  generator = Generator(num_channels=config.NUM_CHANNELS,
                  num_blocks=config.NUM_BLOCKS).to(config.DEVICE)
  # discriminator = Discriminator().to(config.DEVICE)
  # print(generator.state_dict()['initial.cnn.weight'][0][0][0])

  gen_opt = torch.optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.0999))
  # disc_opt = torch.optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.0999))

  mse = nn.MSELoss()
  # bce = nn.BCEWithLogitsLoss()
  # VGGLoss = VGGLoss()
  if config.LOAD == True:
    utils.load_checkpoint(os.path.join(config.GEN_CHK,config.CHEKPOINT_GEN),generator, gen_opt, config.LEARNING_RATE)

  # for name,params in generator.named_parameters():
  #   print("Generator :- min : ",name, torch.min(params))
  #   print("Generator :- max : ",name, torch.max(params))
    # print("Generator :- isinf : ",name,torch.any(torch.isinf(params)).item())
    # print("Generator :- isNaN : ",name,torch.any(torch.isnan(params)).item())

  for epoch in range(51,config.EPOCHS+1):
    train_with_mse(loader, generator, mse, gen_opt, epoch)


def train_with_mse(loader, generator, mse, gen_opt, epoch):
  loop = tqdm(loader, leave=True)
  scaler = torch.cuda.amp.GradScaler()


  for idx,(hr_img, lr_img) in enumerate(loop):

    hr_img = hr_img.to(config.DEVICE)
    lr_img = lr_img.to(config.DEVICE)

    # Creating the Output 4x Higher res image
    with torch.cuda.amp.autocast():
      generated = generator(lr_img)
      generated = torch.clamp(generator(lr_img), 0, 1)
      l2_loss = mse(generated, hr_img)
    # Generator Optimization 
    gen_opt.zero_grad()
    
    # l2_loss.backward()
    # gen_opt.step()

    scaler.scale(l2_loss).backward()
    scaler.step(gen_opt)
    scaler.update()
    torch.cuda.empty_cache()

    print(f"\nLoss={l2_loss}")

    save_idx = 3
    if idx%save_idx == 0:
      with torch.inference_mode():
        print(generated[0][0][:10])
        # cv2.imwrite(f"./Training_Results/Generated/{epoch}_epoch_{idx}_batch_res.png", generated[0].permute((1,2,0)).to('cpu').numpy()*255)
        # cv2.imwrite(f"./Training_Results/Truth/{epoch}_epoch_{idx}_batch_truth.png", hr_img[0].permute((1,2,0)).to('cpu').numpy()*255)
        cv2.imwrite(f"./Training_Results/Generated/res.png", generated[0].permute((1,2,0)).to('cpu').numpy()*255)
        cv2.imwrite(f"./Training_Results/Truth/truth.png", hr_img[0].permute((1,2,0)).to('cpu').numpy()*255)

      utils.save_checkpoint(generator, gen_opt, f"{config.GEN_CHK}/mse_{epoch}_epoch_{idx//save_idx}_gen.pth.tar")


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
    adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
    vggloss = 0.006 * VGGLoss(generated, hr_img)
    total_gen_loss = vggloss + adversarial_loss 

    gen_opt.zero_grad()
    total_gen_loss.backward()
    gen_opt.step()





    x = input("------------------------>")

    
if __name__ == '__main__':
  main()