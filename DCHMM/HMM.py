import torch
from torch import nn


class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (l b d) / (l h b d)
        """
        position_ids = self.position_ids[:x.size(0)]
        return x + self.embeddings(position_ids)[None, :, :].permute(1, 0, 2)


class DataCache(nn.Module):
    def __init__(self, i_dim, k, init_state: torch.Tensor):
        super(DataCache, self).__init__()
        if i_dim != init_state.size(1):
            init_state = torch.zeros(init_state.size(0), i_dim)
        self.k = k
        self.init_state = init_state
        self.cached_data = init_state.unsqueeze(0)
        for i in range(k - 1):
            self.cached_data = torch.cat((self.cached_data, init_state.unsqueeze(0)), 0)
        assert self.cached_data.size(0) == k
        self.count = 0

    def forward(self, dat):
        """
        forward once and cache a data into the cached list in dim 0
        """
        self.count += 1
        if self.count % self.k == 1 or self.k == 1:
            self.cached_data = dat.unsqueeze(0)
        else:
            self.cached_data = torch.cat((self.cached_data, dat.unsqueeze(0)), 0)

        return self.get_cache()

    def get_cache(self):
        """
        return the cached data if the cached list is full or return None
        """
        if self.count % self.k == 0:
            return self.cached_data

    def reset(self, h0):
        self.init_state = h0
        self.cached_data = self.init_state.unsqueeze(0)
        for i in range(self.k - 1):
            self.cached_data = torch.cat((self.cached_data, self.init_state.unsqueeze(0)), 0)
        assert self.cached_data.size(0) == self.k
        self.count = 0


class Attention(nn.Module):
    def __init__(self, enc_dim, hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.a = nn.Softmax(dim=1)

    def forward(self, i_t, h_last):
        # h_last = [batch_size, dec_hid_dim]
        # i_t = [src_len, batch_size, enc_hid_dim]

        src_len = i_t.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        h_last = h_last.unsqueeze(1).repeat(1, src_len, 1)
        i_t = i_t.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((h_last, i_t), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        # a [batch, len]
        a = self.a(attention)
        c = torch.bmm(a.unsqueeze(1), i_t).transpose(0, 1)

        return c, a


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activate=True):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.GELU(),
                                 nn.Linear(out_dim * 4, out_dim),
                                 nn.Tanh() if activate else nn.Identity())

    def forward(self, x):
        print(f"MLP input shape: {x.shape}")  # 添加这一行来打印输入形状
        return self.mlp(x)


class prior_chain(nn.Module):
    def __init__(self, i_dim, h_dim, k):
        super(prior_chain, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.k = k

        # 修改点 确保 GRUCell 的输入和隐藏状态维度一致
        self.gru = nn.GRUCell(self.i_dim, self.h_dim)
        # self.gru = nn.GRUCell(self.i_dim, self.h_dim)
        if self.k > 1:
            self.attention = Attention(self.i_dim, self.h_dim)  # x: k, batch, dim
            self.position_encoder = LearnableAbsolutePositionEmbedding(self.k, i_dim)  # x:batch, k, dim
        # 修改点 添加一个全连接层来调整维度
        self.fc = nn.Linear(self.i_dim, self.h_dim)

    def forward(self, x_t, h_last: torch.Tensor):
        if isinstance(x_t, torch.Tensor):
            if self.k > 1:
                x_t = self.position_encoder(x_t)
                x_t, _ = self.attention(x_t, h_last)

            # 修改点 使用全连接层调整维度
            x_t = self.fc(x_t)

            print("x_t size: ", x_t.size())
            print("h_last size: ", h_last.size())
            h_t = self.gru(x_t.squeeze(0), h_last)
            return h_t


class prior_stacked_chain(nn.Module):
    def __init__(self, i_dim, h_dim, k, init_state):
        super(prior_stacked_chain, self).__init__()
        self.cache = DataCache(i_dim, k, init_state)
        self.chain = prior_chain(i_dim, h_dim, k)
        self.h0 = init_state
        self.h_t = init_state

        # 修改点：确保 init_state 的维度与 h_dim 一致
        if self.h0.size(1) != h_dim:
            self.h0 = torch.zeros(self.h0.size(0), h_dim, device=self.h0.device)
            self.h_t = self.h0

    def forward(self, x_t):
        if isinstance(x_t, torch.Tensor):
            x_t = self.cache(x_t)
            o = self.chain(x_t, self.h_t)
            if o is not None:
                self.h_t = o
            return o

    def get_h_t(self):
        return self.h_t

    def reset(self, h0):
        self.h0 = h0
        self.h_t = self.h0
        self.cache.reset(h0)


class post_chain(nn.Module):
    def __init__(self, i_dim, h_dim, k):
        super(post_chain, self).__init__()
        self.h_dim = h_dim
        self.k = k
        self.factor = nn.Parameter(torch.ones(k) / self.k)
        self.post_fc = MLP(i_dim + h_dim, h_dim)

    def forward(self, x, h_last):
        if isinstance(x, torch.Tensor):
            x_in = x[0] * self.factor[0]
            for i in range(self.k - 1):
                x_in += x[i + 1] * self.factor[i + 1]

            return self.post_fc(torch.cat((x_in, h_last), 1))


class post_stacked_chain(nn.Module):
    def __init__(self, i_dim, h_dim, k, init_state):
        super(post_stacked_chain, self).__init__()
        self.cache = DataCache(i_dim, k, init_state)
        self.chain = post_chain(i_dim, h_dim, k)
        self.h0 = init_state
        self.h_t = init_state
        assert init_state.size(1) == h_dim

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            i_t = self.cache(x)
            o = self.chain(i_t, self.h_t)
            if o is not None:
                self.h_t = o
            return o

    def get_h_t(self):
        return self.h_t

    def reset(self, h0):
        self.h0 = h0
        self.h_t = self.h0
        self.cache.reset(h0)


class prior_pyramid_HMM(nn.Module):
    def __init__(self, i_dim, h_dim, k, m, init_state, device):
        super(prior_pyramid_HMM, self).__init__()
        self.m = m
        self.h0 = init_state
        self.Encoder = MLP(i_dim, h_dim, activate=False).to(device)
        self.chain_list = [prior_stacked_chain(h_dim, h_dim, 1, init_state).to(device)]
        for _ in range(m):
            self.chain_list.append(prior_stacked_chain(h_dim, h_dim, k, init_state).to(device))
        self.h = self.h0
        for _ in range(m):
            self.h = torch.cat((self.h, init_state), 1)

        self.o = None
        self.out_fc1 = nn.Linear(m * h_dim + h_dim, m * h_dim + h_dim).to(device)
        self.out_fc2 = nn.Linear(m * h_dim + h_dim, m * h_dim + h_dim).to(device)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # 添加这一行来打印输入形状
        self.h = None
        for i in range(self.m + 1):
            if self.h is None:
                x = self.Encoder(x)
                print(f"After Encoder shape: {x.shape}")  # 添加这一行来打印输入形状
                h = self.chain_list[i](x)
                self.h = h
            else:
                h = self.chain_list[i](h)
                self.h = torch.cat((self.h, self.chain_list[i].get_h_t()), 1)
        self.o = self.out_fc1(self.h)
        return self.o, self.out_fc2(self.h)

    def get_output(self):
        return self.o

    def reset(self, h0):
        self.h0 = h0
        self.h = self.h0
        for _ in range(self.m):
            self.h = torch.cat((self.h, self.h0), 1)
        for i in range(len(self.chain_list)):
            self.chain_list[i].reset(h0)

    def _init_papameters(self):
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


class post_pyramid_HMM(nn.Module):
    def __init__(self, i_dim, h_dim, k, m, init_state, device):
        super(post_pyramid_HMM, self).__init__()
        self.m = m
        self.h0 = init_state
        self.Encoder = MLP(i_dim, h_dim, activate=False).to(device)
        self.chain_list = [post_stacked_chain(h_dim, h_dim, 1, init_state).to(device)]
        for _ in range(m):
            self.chain_list.append(post_stacked_chain(h_dim, h_dim, k, init_state).to(device))
        self.h = self.h0
        for _ in range(m):
            self.h = torch.cat((self.h, init_state), 1)

        self.o = None

        self.out_fc1 = nn.Linear(m * h_dim + h_dim, m * h_dim + h_dim).to(device)
        self.out_fc2 = nn.Linear(m * h_dim + h_dim, m * h_dim + h_dim).to(device)
        self.vae_fc1 = nn.Linear(h_dim, h_dim).to(device)
        self.vae_fc2 = nn.Linear(h_dim, h_dim).to(device)

    def reset(self, h0):
        self.h0 = h0
        self.h = self.h0
        for _ in range(self.m):
            self.h = torch.cat((self.h, self.h0), 1)

        for i in range(len(self.chain_list)):
            self.chain_list[i].reset(h0)

    def forward(self, x):
        self.h = None

        for i in range(self.m + 1):
            if self.h is None:
                x = self.Encoder(x)
                h = self.chain_list[i](x)
                vae_mu, vae_logvar = self.vae_fc1(h), self.vae_fc2(h)
                self.h = h
            else:
                h = self.chain_list[i](h)
                self.h = torch.cat((self.h, self.chain_list[i].get_h_t()), 1)

        self.o = self.out_fc1(self.h)
        return self.o, self.out_fc2(self.h), vae_mu, vae_logvar

    def get_output(self):
        return self.o

    def _init_papameters(self):
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
    device = torch.device('cuda:0')
    init = torch.zeros(4, 128, device=device)
    model = prior_pyramid_HMM(10, 128, 3, 3, init, device)
    model._init_papameters()
    model.to(device)
    data = torch.randn(100, 200, 10, device=device)
    import time

    t = time.time()
    for i in range(25):
        for j in range(200):
            x = model(data[4 * i:4 * i + 4, j, :])

    print(time.time() - t)
