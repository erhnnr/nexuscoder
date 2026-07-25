
"""
Kod-LLM v1 mimarisi.

v1 hedefi: doğru CALISAN bir pipeline. Buyukluk degil, dogruluk onceligi.
- Ogrenilen pozisyon embedding (RoPE yok, KV-cache yok - v1'de basit tutuyoruz)
- Causal self-attention (mask ZORUNLU - onceki koddaki ana bug buydu)
- Pre-norm + SwiGLU FFN (modern, kararli egitim icin)
- ~20M parametre civari (varsayilan ayarlarla)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalSelfAttention(nn.Module):
    """
    Onemli: causal mask burada F.scaled_dot_product_attention'a
    is_causal=True olarak veriliyor. Egitimde seq_len > 1 oldugu icin
    bu dogru sonuc verir (v1'de KV-cache/tek-token decode YOK, o yuzden
    onceki koddaki is_causal + kv_cache uyumsuzlugu burada soz konusu degil).
    """

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # <-- v1'in kritik duzeltmesi
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class KodLLM_v1(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        dim=384,
        num_layers=6,
        num_heads=6,
        max_seq_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        hidden_dim = int(dim * 8 / 3)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"seq_len {T} > max_seq_len {self.max_seq_len}"

        pos = torch.arange(T, device=input_ids.device)
        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(pos))

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # <pad> token id
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    model = KodLLM_v1(vocab_size=1000, dim=128, num_layers=4, num_heads=4, max_seq_len=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre: {total_params:,} ({total_params/1e6:.2f}M)")

    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))
    logits, loss = model(x, y)
    print(f"Logits: {logits.shape}, Loss: {loss.item():.4f}")

    # Sanity check: causal mask gercekten calisiyor mu?
    # Ayni prefix, farkli devam -> ilk pozisyonlarin logit'i AYNI olmali
    model.eval()
    x1 = torch.randint(0, 1000, (1, 10))
    x2 = x1.clone()
    x2[0, 5:] = torch.randint(0, 1000, (5,))  # 5. pozisyondan sonrasini degistir
    logits1, _ = model(x1)
    logits2, _ = model(x2)
    same = torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5)
    print(f"Causal mask sanity check (ilk 5 pozisyon degismemeli): {'GECTI' if same else 'BASARISIZ'}")
