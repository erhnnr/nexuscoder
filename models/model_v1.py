"""
Kod-LLM v5 mimarisi (model_v2.py).

ADR-0014: RoPE (Rotary Position Embedding) eklendi, KV-cache ile
BIRLIKTE calisacak sekilde tasarlandi (v1'deki ADR-0002 kisitini
kapatiyor). Bu, ogrenilen pozisyon embedding yerine, pozisyon
bilgisini query/key vektorlerine matematiksel bir rotasyonla
gomuyor - daha iyi uzun-baglam genellemesi sagliyor.

v1'den (model_v1.py) farkli, AYRI bir dosya - eski checkpoint'lerle
(v3, v4) UYUMSUZ (farkli parametre sekli/sayisi), bu kasitli: v5
SIFIRDAN egitilecek.
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


def precompute_rope_cache(head_dim, max_seq_len, theta=10000.0):
    """RoPE icin cos/sin tablolarini onceden hesaplar."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """
    cos/sin: (T, head_dim) - ilgili pozisyon araligina onceden dilimlenmis.
    q, k: (B, num_heads, T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """
    ADR-0014: RoPE + KV-cache birlikte. onemli detay: RoPE, cache'deki
    GECMIS key'lere DEGIL, sadece YENI hesaplanan q/k'ya uygulanir -
    gecmis key'ler zaten kendi pozisyonlarina gore rotate edilmis
    halde cache'de duruyor (dogru pozisyon kaymasi icin past_len
    offset'i cos/sin dilimlemede kullanilir).
    """

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)
        use_causal = kv_cache is None

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), present_kv


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

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, present_kv = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, present_kv


class KodLLM_v2(nn.Module):
    def __init__(
        self,
        vocab_size=40000,
        dim=640,
        num_layers=10,
        num_heads=10,
        max_seq_len=768,
        dropout=0.1,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        cos, sin = precompute_rope_cache(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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

    def forward(self, input_ids, targets=None, kv_caches=None):
        B, T = input_ids.shape

        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].shape[2]

        assert past_len + T <= self.max_seq_len, \
            f"seq_len {past_len + T} > max_seq_len {self.max_seq_len}"

        x = self.dropout(self.token_emb(input_ids))

        cos = self.rope_cos[past_len:past_len + T].to(x.device)
        sin = self.rope_sin[past_len:past_len + T].to(x.device)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, present_kv = layer(x, cos, sin, layer_cache)
            new_kv_caches.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        self.eval()

        logits, _, kv_caches = self(input_ids)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            logits, _, kv_caches = self(next_token, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    model = KodLLM_v2(vocab_size=1000, dim=128, num_layers=4, num_heads=4, max_seq_len=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre: {total_params:,} ({total_params/1e6:.2f}M)")

    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))
    logits, loss, _ = model(x, y)
    print(f"Logits: {logits.shape}, Loss: {loss.item():.4f}")

    model.eval()
    x1 = torch.randint(0, 1000, (1, 10))
    x2 = x1.clone()
    x2[0, 5:] = torch.randint(0, 1000, (5,))
    logits1, _, _ = model(x1)
    logits2, _, _ = model(x2)
    same = torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5)
    print(f"Causal mask sanity check: {'GECTI' if same else 'BASARISIZ'}")

    prompt = torch.randint(0, 1000, (1, 5))
    full_seq = torch.randint(0, 1000, (1, 8))
    full_seq[0, :5] = prompt[0]

    logits_full, _, _ = model(full_seq)

    logits_prefill, _, cache = model(prompt)
    step1_token = full_seq[:, 5:6]
    logits_step1, _, cache = model(step1_token, kv_caches=cache)

    kv_matches = torch.allclose(logits_full[0, 5], logits_step1[0, -1], atol=1e-4)
    print(f"RoPE + KV-cache tutarlilik testi: {'GECTI' if kv_matches else 'BASARISIZ'}")
