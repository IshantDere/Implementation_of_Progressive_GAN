import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import os
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent = torch.randn(1,512)

class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size ,
                 stride ,
                 padding ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self,x):
        return self.act(self.norm(self.conv(x)))

class ConvT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size ,
                 stride ,
                 padding ):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self,x):
        return self.act(self.norm(self.conv(x)))

class Generator_block(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 up_sample = True
                 ):
        super().__init__()

        self.up_sample = up_sample

        self.conv1 = Conv(in_channels, hidden_dim,
                               (3,3), (1,1), 1)
        self.conv2 = Conv(hidden_dim, hidden_dim,
                               (3,3), (1,1), 1)
        if up_sample:
            self.up = nn.Upsample(scale_factor = 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.up_sample:
            x = self.up(x)
        return x

x = torch.randn(1, 512,2,2)
model = Generator_block(512, 512, up_sample = False)
model(x).shape

class ToRGB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels = 3,
                 kernel_size = (3,3),
                 stride = (1,1),
                 padding = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size,
                              stride,
                              padding)
        self.norm = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Generator(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 n_layers = 6):
        super().__init__()

        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                Generator_block(in_channels,
                                hidden_dim)
            )
            in_channels = hidden_dim
            hidden_dim = hidden_dim // 2

        self.gen_layers = nn.Sequential(*self.layers)
        self.out = ToRGB(in_channels)
        print(self.layers)

    def forward(self, x):
        return self.out(self.gen_layers(x))

x = torch.randn(1, 512,2,2).to(device)
model_gen = Generator(512, 512).to(device)
model_gen(x).shape

!nvidia-smi

class Discriminator(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 n_layers = 2):
        super().__init__()

        self.layers_disc = []
        
        
        for i in range (n_layers):
            self.layers_disc.append(
                Conv(in_channels,
                     hidden_dim,
                     (3,3),
                     (1,1),
                     1)
            )
            in_channels = hidden_dim
            hidden_dim = hidden_dim *2

            self.layers = nn.Sequential(*self.layers_disc)
            # print(self.layers_disc)

    def forward(self, x):
        return self.layers(x)

x = torch.randn(1, 3, 128, 128).to(device)
model_disc = Discriminator(3, 64).to(device)
model_disc(x).shape

!nvidia-smi

class Datasets(Dataset):
    def __init__(self,
                 root_dir):
        super().__init__()
        self.root_dir = root_dir

        self.files = os.listdir(os.path.join(root_dir, 'train')) * 10000

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.root_dir, 'train', self.files[idx])
        ).convert("RGB")

        latent = torch.randn(512,2,2)

        img = self.transform(img)

        return latent, img

dataset = datasets('/content/dataset')
dataloader = DataLoader(dataset, 2, shuffle = True)

for latent, img in dataloader:
    print(latent.shape, img.shape)
    sdsd

lr_gen = 0.0001
lr_disc = 0.0001
batch_size = 4
epoch = 4
optim_gen = optim.Adam(model_gen.parameters(), lr = lr_gen)
optim_disc = optim.Adam(model_disc.parameters(), lr = lr_disc)
generator = model_gen.to(device)
discriminator = model_disc.to(device)
loss_gen = nn.L1Loss()
loss_disc = nn.BCELoss()
step = 0

for epoch in range(epoch):
    for latent, img in dataloader:
        latent = latent.to(device)
        img = img.to(device)

        y_pred = generator(latent)

        with torch.no_grad():
            y_pred_fake = discriminator(y_pred)
            y_pred_real = discriminator(img)

            loss_disc_fake = loss_disc(
                y_pred_fake, torch.zeros_like(y_pred_fake)
            )
            loss_disc_real = loss_disc(
                y_pred_real, torch.ones_like(y_pred_fake)
            )
            loss_disc = loss_disc_fake + loss_disc_real

            total_gen_loss = loss_gen + loss_disc

            optim_gen.zero_grad()
            total_gen_loss.backward()
            optim_gen.step()

        with torch.no_grad():
            y_pred = generator(latent)
        y_pred_fake = discriminator(y_pred)
        y_pred_real = discriminator(img)

        loss_disc_fake = loss_disc(
            y_pred_fake, torch.zeros_like(y_pred_fake)
        )
        loss_disc_real = loss_disc(
            y_pred_real, torch.ones_like(y_pred_fake)
        )
        loss_disc = loss_disc_fake + loss_disc_real

        total_gen_loss = loss_gen + loss_disc

        optim_gen.zero_grad()
        total_gen_loss.backward()
        optim_gen.step()

        step += 1
