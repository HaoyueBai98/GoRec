import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input, out=False):
        output = self.linear(input)
        return output


class Encoder(nn.Module):
    def __init__(self, latent_dim, layer, z_size, si_dim):
        super(Encoder, self).__init__()
        self.size = latent_dim
        # layers = []
        # for i in range(layer):
        #     layers.append(EncoderBlock(input_dim=self.size, output_dim=64))
        #     self.size = latent_dim
        # self.inference = nn.Sequential(*layers)

        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + si_dim), out_features=(latent_dim), bias=False),
                                nn.BatchNorm1d(num_features=latent_dim),
                                nn.Tanh())
        self.l_mu = nn.Linear(in_features= self.size, out_features=z_size)
        self.l_var = nn.Linear(in_features= self.size, out_features=z_size)

        self.l_mu_zgc = nn.Linear(in_features= si_dim, out_features=z_size)
        self.l_var_zgc = nn.Linear(in_features= si_dim, out_features=z_size)

    def forward(self, warm, side_information):
        # warm = self.inference(warm)
        
        mu_zgc = self.l_mu_zgc(side_information)
        logvar_zgc = self.l_var_zgc(side_information)

        warm = torch.cat((side_information, warm), 1)
        warm = self.fc(warm)
        mu = self.l_mu(warm)
        logvar = self.l_var(warm)
        return mu, logvar, mu_zgc, logvar_zgc


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderBlock, self).__init__()


        self.linear = nn.Linear(input_dim, output_dim)
        # self.bn = nn.BatchNorm1d(num_features=dim_out, momentum=0.01, eps=0.001)

    def forward(self, input, out=False):
        output = self.linear(input)
        ten_out = output
        # output = self.bn(output)
        # warm = F.relu(warm, False)
        # output = torch.tanh(output)
        # if out=True return intermediate output for reconstruction error
        if out:
            return output, ten_out
        return output


class Decoder(nn.Module):
    def __init__(self, z_size, latent_dim, layer, si_dim):
        super(Decoder, self).__init__()

        # start from B * z_size
        # concatenate one hot encoded class vector
        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + si_dim), out_features=(latent_dim), bias=False),
                                nn.BatchNorm1d(num_features=latent_dim),
                                nn.Tanh())
        self.size = latent_dim
        layers = []
        for i in range(layer):
            layers.append(EncoderBlock(input_dim=self.size, output_dim=64))
            self.size = latent_dim

        self.geneator = nn.Sequential(*layers)

    def forward(self, z, side_information):
        z_cat = torch.cat((side_information, z), 1)
        rec_warm = self.fc(z_cat)
        rec_warm = self.geneator(rec_warm)
        return rec_warm

class GoRec(nn.Module):
    def __init__(self, env, latent_dim, z_size, si_dim, training=True, encoder_layer=2, decoder_layer=2):
        super(GoRec, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(latent_dim=latent_dim , layer=encoder_layer, z_size=self.z_size, si_dim=si_dim)
        # self.de_dim_layer = nn.Linear(in_features=si_dim, out_features=si_dim//2)
        # si_dim = si_dim//2
        self.decoder = Decoder(z_size=self.z_size, latent_dim=latent_dim, layer=decoder_layer, si_dim=si_dim)
        self.env = env
        self.latent = latent_dim
        self.training = training
        self.to(env.device)
        self.dropout = nn.Dropout(p=env.args.dropout)

    def forward(self, warm, side_information, gen_size=10):
        # side_information = self.de_dim_layer(side_information)
        # side_information = torch.tanh(side_information)
        if self.training:
            original = warm

            # encode
            mu, log_variances, mu_zgc, log_variances_zgc = self.encoder(warm, side_information)

            # we need true variance not log
            variances = torch.exp(log_variances * 0.5)
            variances_zgc = torch.exp(log_variances_zgc * 0.5)

            # sample from gaussian
            sample_from_normal = Variable(torch.randn(len(warm), self.z_size).to(self.env.device), requires_grad=True)
            sample_from_normal_zgc = Variable(torch.randn(len(warm), self.z_size).to(self.env.device), requires_grad=True)

            # shift and scale using mean and variances
            z = sample_from_normal * variances + mu
            zgc = sample_from_normal_zgc * variances_zgc + mu_zgc

            # decode tensor
            side_information = self.dropout(side_information)
            rec_warm = self.decoder(z, side_information)

            return rec_warm, mu, log_variances, z, zgc
        else:
            if warm is None:
                # just sample and decode
                z = Variable(torch.randn(gen_size, self.z_size).to(self.env.device), requires_grad=False)
            
            else:
                mu, log_variances, _, _ = self.encoder(warm, side_information)
                # torch.save(mu, os.path.join(self.env.DATA_PATH, f'z_u{self.env.args.uni_coeff}_{self.env.args.dataset}.pt'))

                # _, _, mu, log_variances = self.encoder(warm, side_information)
                # we need true variance not log
                variances = torch.exp(log_variances * 0.5)

                # sample from gaussian
                sample_from_normal = Variable(torch.randn(len(warm), self.z_size).to(self.env.device), requires_grad=True)

                # shift and scale using mean and variances
                # z = sample_from_normal * variances + mu
                z =  mu


            # decode tensor
            rec_warm = self.decoder(z, side_information)
            return rec_warm
