import torch
import numpy as np
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self , input_channels = 3 , output_channels_1 = 128 , num_blocks = 4):
        super(Discriminator, self).__init__()
        
        self.block1 = nn.Conv2d(input_channels,  output_channels_1, kernel_size = 4, stride = 2 , padding = 1, bias = False)
        self.act1 = nn.LeakyReLU(0.2)
        self.seq_blocks = nn.Sequential(*[self.unit_module(output_channels_1* i , output_channels_1 * 2 * i , 1, 2) for i in range(1,num_blocks+1) if i%2 == 0 or i == 1])
        self.block2 = nn.Conv2d(1024, 1, kernel_size=4 ,padding=0, stride= 1, bias = False)
        self.act = nn.Sigmoid()
        self.apply(self.__init_weights)
        
    def __init_weights(self, module):
         
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean = 0.0, std= 0.02)   
            
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight, mean = 0.0, std= 0.02)   
            
             
        
        
    def unit_module(self,in_chan,out_chan ,padding, stride):
        
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size = 4, stride = stride , padding = padding, bias = False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        x = self.block1(x)
        x = self.act1(x)
        x = self.seq_blocks(x)
        x = self.block2(x)
        #x = self.act(x)
        x = self.act(x.view(x.shape[0],-1))
        return x
    
# a = torch.rand(1,3,64,64).to('cuda:0')

# model = Discriminator().to('cuda:0')
# out = model(a)
# print(out)