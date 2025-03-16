from DCHMM.HMM import *


class Model(nn.Module):
    def __init__(self, args):  # i_dim, h_dim, k, m, h0, cls,device):
        super(Model, self).__init__()
        i_dim = args.input_dim
        h_dim = args.hidden_dim
        k = args.k
        m = args.m
        device = args.device
        cls_dim = args.class_num
        h0 = torch.zeros(args.batch_size, h_dim, device=device)

        self.prior = prior_pyramid_HMM(i_dim, h_dim, k, m, h0, device).to(device)

        self.post = post_pyramid_HMM(i_dim, h_dim, k, m, h0, device).to(device)

        self.vae_decoder = MLP(h_dim, i_dim, activate=False).to(device)
        self.predictor = MLP(h_dim * m + h_dim, i_dim, activate=False).to(device)
        self.classifier = MLP(h_dim * m + h_dim, cls_dim, activate=False).to(device)

    def reparameterize(self, mu, logvar, test):
        if not test:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x_t, test=False):
        prior_mu, prior_logvar = self.prior(x_t)
        post_mu, post_logvar, vae_mu, vae_logvar = self.post(x_t)

        post_h = self.reparameterize(vae_mu, vae_logvar, test)

        vae_x = self.vae_decoder(post_h)

        return vae_x, post_mu, post_logvar, prior_mu, prior_logvar

    def predict_one_step(self):
        return self.predictor(self.post.get_output())

    def classify_one_step(self):
        return self.classifier(self.post.get_output())

    def reset(self, h0):
        self.prior.reset(h0)
        self.post.reset(h0)

    def init_parameters(self):
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    pass
            elif isinstance(m, nn.GRUCell):
                nn.init.xavier_normal_(m.weight_hh.data)
                nn.init.xavier_normal_(m.weight_ih.data)
                nn.init.constant_(m.bias_ih.data, 1)
                nn.init.constant_(m.bias_hh.data, 1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='our')  # i_dim, h_dim, k, m, h0, cls,device):
    parser.add_argument('--input_dim', type=int, default=10, help='None')
    parser.add_argument('--hidden_dim', type=int, default=128, help='None')
    parser.add_argument('--k', type=int, default=3, help='None')

    parser.add_argument('--m', type=int, default=3, help='None')
    parser.add_argument('--batch_size', type=int, default=4, help='None')
    parser.add_argument('--class_num', type=int, default=4, help='None')
    parser.add_argument('--device', type=str, default='cuda:0', help='None')
    args = parser.parse_args()
    device = args.device
    init = torch.zeros(4, 128, device=device)
    model = model(args)
    model.init_papameters()
    model.to(device)
    data = torch.randn(100, 200, 10, device=device)
    import time

    t = time.time()
    for i in range(25):
        for j in range(200):
            a = model(data[4 * i:4 * i + 4, j, :], False)
            logit = model.classify()
            pre = model.predict_one_step()
    print(time.time() - t)
