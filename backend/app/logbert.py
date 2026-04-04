import torch
import torch.nn as nn


class LogBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 514,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, hidden_size) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=128,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mask_lm = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = x.size()
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        encoded = self.encoder(x)
        logits = self.mask_lm(encoded)
        cls_emb = encoded[:, 0, :]
        return logits, cls_emb
