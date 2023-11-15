import torch
from torch import nn
import torchvision.datasets as dset
from torchvision import transforms 
from torchvision.utils import save_image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import argparse
import numpy as np

BATCH_SZ = 32
LATENT_DIMS = 32
G_FEATURES = 64
D_FEATURES = 32

EPOCHS_PER_TEST = 1
BATCHES_PER_PRINT = 10

GRAYSCALE = False
IMAGE_SZ = 64
#MEAN = 0.45
#STDDEV = 0.28
MEAN = 0.515
STDDEV = 0.30

if GRAYSCALE:
    CHANNELS=1
else:
    CHANNELS=3

if torch.cuda.is_available():
    print('Using CUDA')
    torch.set_default_device('cuda:0')
    device = torch.device('cuda:0')
else:
    print('Using CPU')
    device = torch.device('cpu')

def tensor2img(tensor):
    return tensor.permute(1, 2, 0)

def denormalize(imgs):
    return (imgs * STDDEV + MEAN).cpu().clamp(-1.0, 1.0)

def show_imgs(imgs):
    imgs = denormalize(imgs)
    if args.save:
        save_image(imgs, 'img.png', normalize=True, value_range=(-1.0, 1.0), nrows=5)
    else:
        figure = plt.figure(figsize=(12, 7))
        cols, rows = 5, 4
        for i in range(1, cols * rows + 1):
            figure.add_subplot(rows, cols, i)
            plt.axis("off")
            if GRAYSCALE:
                plt.imshow(tensor2img(imgs[i-1]), cmap='gray', vmin=-1.0, vmax=1.0)
            else:
                plt.imshow(tensor2img(imgs[i-1]), vmin=-1.0, vmax=1.0)
        plt.show()

#Create model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIMS, G_FEATURES * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(G_FEATURES * 8),
            nn.Dropout(0.3),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_FEATURES * 8, G_FEATURES * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_FEATURES * 4),
            nn.Dropout(0.3),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_FEATURES * 4, G_FEATURES * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_FEATURES * 2),
            nn.Dropout(0.3),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_FEATURES * 2, G_FEATURES, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_FEATURES),
            nn.Dropout(0.3),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_FEATURES, CHANNELS, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(CHANNELS, D_FEATURES, 4, stride=2, padding=1, bias=False),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_FEATURES, D_FEATURES * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_FEATURES * 2),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_FEATURES * 2, D_FEATURES * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_FEATURES * 4),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_FEATURES * 4, D_FEATURES * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_FEATURES * 8),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_FEATURES * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

def init_argparse():
    parser = argparse.ArgumentParser(usage="%(prog)s [-t] [-e epochs] [-V] [-s] [model]")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--epochs", action="store", type=int, default=20)
    parser.add_argument("-V", "--visual", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-S", "--noshuffle", action="store_true")
    parser.add_argument("-d", "--datadir", action="store", type=str, default='data/celeba')
    parser.add_argument("-l", "--lr", action="store", type=float, default=0.0002)
    parser.add_argument("model_file", action="store", nargs='?')
    return parser

def compute_stats(loader):
    data_mean = [] # Mean of the dataset
    data_std = [] # std with ddof = 1
    for image, _ in loader:
        numpy_image = image.numpy()

        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std = np.std(numpy_image, axis=(0,2,3), ddof=1)

        data_mean.append(batch_mean)
        data_std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    data_mean = np.array(data_mean).mean(axis=0)[0]
    data_std = np.array(data_std).mean(axis=0)[0]
    print(data_mean, data_std)

#Handle arguments
parser = init_argparse()
args = parser.parse_args()

#Initialize model
d = Discriminator()
g = Generator()
d_optim = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.999))
g_optim = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

model_file_exists = False
if args.model_file:
    try:
        checkpoint = torch.load(args.model_file, map_location=device)
        d.load_state_dict(checkpoint['discriminator'])
        g.load_state_dict(checkpoint['generator'])
        print(f"Loaded {args.model_file}")
        model_file_exists = True
    except (FileNotFoundError):
        pass
    except () as err:
        print(f"{args.prog}: {file}: {err.strerror}", file=sys.stderr)

if not (args.train or model_file_exists):
    raise ValueError("Must either train (-t) or supply a model.")

fixed_noise = torch.randn((20, LATENT_DIMS, 1, 1))

#Train
if args.train:
    print("Loading training data...")
    if GRAYSCALE:
        train_data = dset.ImageFolder(root=args.datadir,
           transform=transforms.Compose([
               transforms.Grayscale(),
               transforms.Resize(IMAGE_SZ),
               transforms.CenterCrop(IMAGE_SZ),
               transforms.ToTensor(),
               transforms.Normalize(MEAN, STDDEV),
        ]))
    else:
        train_data = dset.ImageFolder(root=args.datadir,
           transform=transforms.Compose([
               transforms.Resize(IMAGE_SZ),
               transforms.CenterCrop(IMAGE_SZ),
               transforms.ToTensor(),
               transforms.Normalize((MEAN, MEAN, MEAN), (STDDEV, STDDEV, STDDEV))
        ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SZ, drop_last=True, shuffle=(not args.noshuffle))
    #compute_stats(train_loader)
    real_imgs, _ = next(iter(train_loader))
    show_imgs(real_imgs)

    print("Training...")
    for epoch in range(args.epochs):
        d.train()
        g.train()
        for i, (real_samples, _) in enumerate(train_loader):
            #Train D on real
            real_samples = real_samples.to(device)
            d.zero_grad()
            real_labels = torch.ones(BATCH_SZ)
            d_out_real = d(real_samples).view(-1)
            d_loss = criterion(d_out_real, real_labels)
            d_loss.backward()
            #Train D on fake
            latent_samples = torch.randn(BATCH_SZ, LATENT_DIMS, 1, 1)
            fake_samples = g(latent_samples)
            fake_labels = torch.zeros(BATCH_SZ)
            d_out_fake = d(fake_samples.detach()).view(-1)
            d_loss = criterion(d_out_fake, fake_labels)
            d_loss.backward()
            d_optim.step()
            #Train G
            g.zero_grad()
            d_out_fake = d(fake_samples).view(-1)
            g_loss = criterion(d_out_fake, real_labels)
            g_loss.backward()
            g_optim.step()

            if i % BATCHES_PER_PRINT == BATCHES_PER_PRINT - 1:
                print(f'[{epoch+1}/{args.epochs} : {i+1}] dloss {d_loss:.4f} gloss {g_loss:.4f} fake mean {d_out_fake.mean().item():.3f} real mean {d_out_real.mean().item():.3f}')

        if args.visual:
            #Show progress
            if epoch % EPOCHS_PER_TEST == 0:
                g.eval()
                with torch.no_grad():
                    out = g(fixed_noise)
                show_imgs(out)

    #Save model
    if args.model_file:
        try:
            torch.save({
                'discriminator': d.state_dict(),
                'generator': g.state_dict()
                }, args.model_file)
            print(f"Saved {args.model_file}")
        except (FileNotFoundError, IsADirectoryError) as err:
            print(f"{sys.argv[0]}: {file}: {err.strerror}", file=sys.stderr)

if args.visual or not args.train:
    print("Testing...")
    g.eval()
    with torch.no_grad():
        out = g(fixed_noise)
        show_imgs(out)
