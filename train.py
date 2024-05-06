import argparse
import os
import numpy as np
from dataloader import ganDataset
from models.generator import Generator
from models.discriminator import Discriminator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , random_split
from torchvision.utils import make_grid , save_image

def arguments():
    parser = argparse.ArgumentParser(description = "Define dynamic parameters for the model.")
    parser.add_argument('--epochs' , default = 20 , type = int , help = 'number of epochs')
    parser.add_argument('--data_dir' , default = './data' , type = str , help = 'data directory path')
    parser.add_argument('--batch_size' , default = 64 , type = int , help = 'define batch size')
    parser.add_argument("--latent_dims", default=100, type = int , help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--learning_rate", default=0.0002, type = float , help="learning rate")
    parser.add_argument("--beta1", default=0.5, type = float , help="value of beta1")
    parser.add_argument('--train_ratio' , default = 0.8 , type = float , help = "define_test_ratio")
    parser.add_argument("--weight_dir", default="./weight", type = str , help="weight directory path")
    parser.add_argument("--device", default="cuda:0", help="cuda device, i.e. 0 or cpu")
    return parser.parse_args() 

args = arguments()

def run():
    
    dataset = ganDataset(args.data_dir)
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset , val_dataset = random_split(dataset,[train_size,val_size]) 
    
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader =  DataLoader(dataset=val_dataset,batch_size=args.batch_size)
    
    noise = torch.randn(16,args.latent_dims,1,1).to(args.device)

    gen_model = Generator().to(args.device)
    disc_model = Discriminator().to(args.device)
    criterion = nn.BCELoss()
    opt_gen = torch.optim.Adam(gen_model.parameters(),lr = args.learning_rate,betas=(args.beta1,0.999))
    opt_disc = torch.optim.Adam(disc_model.parameters(),lr = args.learning_rate,betas=(args.beta1,0.999))
    

    # create folders
    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs('./result' , exist_ok=True)

    ## training loop

    best_val_disc_loss = 1000
    best_val_gen_loss = 1000
  
    for epoch in range(args.epochs):
        gen_model.train(True)
        disc_model.train(True)
        total_loss_disc_train = 0.0
        total_loss_gen_train = 0.0
        last_loss_disc_train = 0.0
        last_loss_gen_train = 0.0
        total_disc_loss_val = 0.0 
        gen_loss_val = 0.0
        
        for idx,batch_train in enumerate(train_loader):
            noise_batch = torch.randn(args.batch_size,args.latent_dims,1,1).to(args.device)
            batch_train = batch_train.to(args.device)
            fake_img = gen_model(noise_batch)
            fake_prediction = disc_model(fake_img)
            real_prediction = disc_model(batch_train)
            opt_disc.zero_grad()
            loss_real = criterion(real_prediction , torch.ones_like(real_prediction))
            loss_fake = criterion(fake_prediction , torch.zeros_like(fake_prediction))
            total_disc_loss = (loss_fake + loss_real) / 2
            
            total_disc_loss.backward(retain_graph = True)
            
            opt_disc.step()
            
            fake_pred = disc_model(fake_img)
            opt_gen.zero_grad()
            gen_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            gen_loss.backward()
            opt_gen.step()
            
            
            total_loss_disc_train += total_disc_loss.item()
            total_loss_gen_train += gen_loss.item()
            
            if idx % 100 == 99:
                last_loss_disc_train = total_loss_disc_train / 100
                last_loss_gen_train = total_loss_gen_train / 100
                total_loss_disc_train = 0.0
                total_loss_gen_train = 0.0
                print('batch {} disc_loss: {} gen_loss: {}'.format(idx + 1, last_loss_disc_train , last_loss_gen_train))
            
     
        
        disc_model.eval()
        gen_model.eval()
        with torch.no_grad():
            for idx_val , batch_val in enumerate(val_loader):
                batch_val = batch_val.to(args.device)
                fake_img_val = gen_model(noise_batch)
                fake_prediction_val = disc_model(fake_img_val)
                real_prediction_val = disc_model(batch_val)
                loss_real_val = criterion(real_prediction_val , torch.ones_like(real_prediction_val))
                loss_fake_val = criterion(fake_prediction_val , torch.zeros_like(fake_prediction_val))
                total_disc_loss_val = (loss_fake_val + loss_real_val) / 2
                
                
                
                fake_pred_val = disc_model(fake_img_val)
            
                gen_loss_val = criterion(fake_pred_val, torch.ones_like(fake_pred_val))
                
                
                
            ## save_image_grid
            generated_fake_img = gen_model(noise)
            grid_16 = make_grid(generated_fake_img,nrow=4,normalize=True)
            save_image(grid_16 , f'./result/generated_epoch{epoch}.jpg')
            
        
           
            
            
                
        disc_val_loss_per_epoch = total_disc_loss_val / len(val_loader)    
        gen_val_loss_per_epoch = gen_loss_val / len(val_loader)    
        if disc_val_loss_per_epoch < best_val_disc_loss and gen_val_loss_per_epoch < best_val_gen_loss:
            best_val_disc_loss = disc_val_loss_per_epoch
            best_val_gen_loss = gen_val_loss_per_epoch
            torch.save(disc_model.state_dict() , os.path.join(args.weight_dir , 'discriminator.pt'))
            torch.save(gen_model.state_dict() , os.path.join(args.weight_dir , 'generator.pt'))
            
            
                
          
        print(f'Epochs: {epoch + 1} | Val Discriminator Loss: {disc_val_loss_per_epoch: .3f} | Val Generator Loss: {gen_val_loss_per_epoch : .3f}' )
  
    
    
if __name__ == '__main__':
    
    run()