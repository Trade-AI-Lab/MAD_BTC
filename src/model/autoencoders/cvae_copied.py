import torch
from torch import nn, optim
from torch.nn import functional as F

batch_size = 64
latent_size = 20

device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}

def one_hot(labels, class_size):
    """
    Instead use :  torch.nn.functional.one_hot(torch.tensor([4,2,3,1,0]),num_classes=5)
    Args:
        labels (int):
        class_size ():

    Returns:

    """
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_size = latent_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def sample_normal(self, mu, logvar):
        self.normal_dist = torch.distributions.normal.Normal(mu,torch.exp(0.5*logvar))
        return self.normal_dist.sample()

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28), c)
        # z = self.reparameterize(mu, logvar)
        z = self.sample_normal(mu, logvar)
        return self.decode(z, c), mu, logvar

    def loss_function_dist(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        prior_normal = torch.distributions.normal.Normal(torch.zeros_like(self.latent_size), torch.ones_like(self.latent_size))
        KL = torch.sum(torch.distributions.kl.kl_divergence(prior_normal,self.normal_dist),axis=-1)
        return KL + BCE

# create a CVAE model
model = CVAE(28*28, latent_size, 10).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


"""
In [22]: b = torch.distributions.Normal(torch.tensor([0.5,1,0.3]),torch.tensor([1.0,4,8]))

In [23]: b.sample()
Out[23]: tensor([0.7969, 2.1985, 5.9910])

In [24]: a = torch.distributions.Normal(torch.tensor([0.5,0.2,0.3]),torch.tensor([1.0,3,2]))

In [25]: torch.distributions.kl.kl_divergence(a,b)
Out[25]: tensor([0.0000, 0.0889, 0.9175])

In [26]: torch.sum(torch.distributions.kl.kl_divergence(a,b),axis=-1)
Out[26]: tensor(1.0065)

"""