import torch
import torch.nn as nn


class KimCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_classes: int = 2,
        filter_sizes=(3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings=None,
        static: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if static:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)         # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2)         # [batch, embed_dim, seq_len]

        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))   # [batch, num_filters, L]
            p = torch.max(c, dim=2).values
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)