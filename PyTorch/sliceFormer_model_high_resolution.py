import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_width = 1
print('kernel_width:', kernel_width)
if kernel_width == 1:
    patch_size = (512, 1)
    final_input_channels = 64

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=(256, 16), embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        # self.positional_encodings = nn.Parameter(torch.rand(32, embedding_dim), requires_grad=True)
        num_patches = (512 // patch_size[0]) * (1024 // patch_size[1])
        self.positional_encodings = nn.Parameter(torch.rand(num_patches, embedding_dim), requires_grad=True)

    def forward(self, x):
        # x --> [bs, 64, 128, 256]
        embeddings = self.embedding_convPxP(x).flatten(2) # [bs, 128, 16]  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)

class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=(256, 16), dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x) # x --> [2, 128, 128, 256]

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        # self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

        self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features // 32)

        self.conv3 = nn.Conv2d(features // 32, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d5)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class SliceFormer(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(SliceFormer, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(64, n_query_channels=64, patch_size=patch_size,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(num_classes=64)
        self.conv_out = nn.Sequential(nn.Conv2d(final_input_channels, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs) # shape --> [2, 64, 512, 1024]
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=False)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


# if __name__ == '__main__':
#     model = SliceFormer.build(100)
#     x = torch.rand(2, 3, 512, 1024)
#     bins, pred = model(x)
#     print(bins.shape, pred.shape)

