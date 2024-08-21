import torch
import torch.nn as nn
from torchvision.models import resnet101



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoded_image_size = config.encoded_image_size

        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]      # Pooling 및 FC Layer 제거
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))     # 입력을 가변적인 크기의 이미지를 받을 수 있도록 고정 된 크기로 조정

        self.fine_tune()

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        # 초반(low level)의 5개 layer를 제외한 나머지 부분(high-level) 학습
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def forward(self, images):
        output = self.resnet(images)    # (B, 2048, H/32, W/32)
        output = self.adaptive_pool(output) # (B, 2048, encoded_size, encoded_size)
        output = output.permute(0, 2, 3, 1) # (B, encoded_size, encoded_size, 2048)

        return output


class Attention(nn.Module):
    def __init__(self, config, encoder_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, config.attention_dim)
        self.decoder_att = nn.Linear(config.decoder_dim, config.attention_dim)
        self.full_att = nn.Linear(config.attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_output, hidden_state):
        """
        encoder_output: (batch size, num pixels, encoder dimension)
        hidden_state: (batch size, decoder dimension)
        """
        att1 = self.encoder_att(encoder_output)
        att2 = self.decoder_att(hidden_state)

        att = self.full_att(self.relu((att1 + att2.unsqueeze(1)))).squeeze(2) # (batch size, num pixels, attention_dim) -> (batch size, num pixels)
        att_score = self.softmax(att)
        attention_weighted_encoding = (encoder_output * att_score.unsqueeze(2)).sum(dim=1) # (batch size, encoder dimension)

        return attention_weighted_encoding, att_score


class Decoder(nn.Module):
    def __init__(self, config, encoder_dim=2048):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.init_h = nn.Linear(encoder_dim, config.decoder_dim)
        self.init_c = nn.Linear(encoder_dim, config.decoder_dim)
        self.lstm = nn.LSTMCell(config.embedding_dim + encoder_dim, config.decoder_dim, bias=True)
        self.attention = Attention(config, encoder_dim)
        self.beta_layer = nn.Linear(config.decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(config.decoder_dim, config.vocab_size)
        self.dropout = nn.Dropout(p=config.dropout)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)
        h = self.init_h(mean_encoder_output)
        c = self.init_c(mean_encoder_output)
        return h, c

    def forward(self, encoder_output, captions):
        batch_size = encoder_output.size(0)
        encoder_dim = encoder_output.size(-1)
        captions_length = captions.size(1)

        encoder_output = encoder_output.view(batch_size, -1, encoder_dim)   # (B, num_pixels, encoder_dim)
        h, c = self.init_hidden_state(encoder_output)   # (B, decoder_dim)
        num_pixels = encoder_output.size(1)

        embeddings = self.embedding(captions)    # (B, caption_lens, embedding dim)

        predictions = torch.zeros(batch_size, captions_length, self.config.vocab_size).to(embeddings.device)
        all_attention_scores = torch.zeros(batch_size, captions_length, num_pixels).to(embeddings.device)
        for t in range(captions_length):
            # attention
            attention_weighted_encoding, attention_score = self.attention(encoder_output, h)

            gate = self.sigmoid(self.beta_layer(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # concat & lstm
            h, c = self.lstm(torch.cat([embeddings[:, t], attention_weighted_encoding], dim=1), (h, c))  # (batch, decoder dim)
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            all_attention_scores[:, t, :] = attention_score

        return predictions, captions, all_attention_scores