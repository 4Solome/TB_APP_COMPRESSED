import torch
import torch.nn as nn


class TTVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=16,
        d_model=64,
        nhead=4,
        n_layers=2,
        n_cont=0,
        n_bin=0,
        cat_sizes=None,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_cont = n_cont
        self.n_bin = n_bin
        self.cat_sizes = cat_sizes or []

        self.value_proj = nn.Linear(1, d_model)
        self.positional = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

        self.decoder_trunk = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self.dec_cont = nn.Linear(d_model, n_cont) if n_cont > 0 else None
        self.dec_bin = nn.Linear(d_model, n_bin) if n_bin > 0 else None
        self.dec_cat = nn.ModuleList(
            [nn.Linear(d_model, size) for size in self.cat_sizes]
        )

    def encode(self, x):
        x_seq = x.unsqueeze(-1)                      # [B, D] -> [B, D, 1]
        x_emb = self.value_proj(x_seq) + self.positional
        h = self.encoder(x_emb).mean(dim=1)         # pooled transformer output
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=-8.0, max=8.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_heads(self, z):
        h = self.decoder_trunk(z)
        cont_hat = self.dec_cont(h) if self.dec_cont is not None else None
        bin_logits = self.dec_bin(h) if self.dec_bin is not None else None
        cat_logits = [head(h) for head in self.dec_cat]
        return cont_hat, bin_logits, cat_logits

    def decode(self, z):
        """
        Returns a single reconstructed tensor aligned to transformed feature space:
        continuous values + sigmoid(binary) + one-hot categorical argmax
        This is useful for Streamlit inference and reconstruction error.
        """
        cont_hat, bin_logits, cat_logits = self.decode_heads(z)
        parts = []

        if cont_hat is not None:
            parts.append(torch.clamp(cont_hat, 0.0, 1.0))

        if bin_logits is not None:
            parts.append(torch.sigmoid(bin_logits))

        for logits in cat_logits:
            probs = torch.softmax(logits, dim=1)
            onehot = torch.zeros_like(probs)
            onehot.scatter_(1, torch.argmax(probs, dim=1, keepdim=True), 1.0)
            parts.append(onehot)

        if len(parts) == 0:
            raise ValueError("Decoder produced no outputs. Check n_cont, n_bin, cat_sizes.")

        return torch.cat(parts, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        cont_hat, bin_logits, cat_logits = self.decode_heads(z)
        return cont_hat, bin_logits, cat_logits, mu, logvar
