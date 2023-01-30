from dataclasses import dataclass, field
import torch
from einops import rearrange
import torch.nn as nn
from typing import List, OrderedDict, Union, Optional, Dict, Any
from transformers.generation_utils import GenerationMixin
from retrieval import Retriever, DummyRetriever
import torch.nn.functional as F
import math
from transformers.activations import NewGELUActivation


@dataclass
class RetroConfig:

    num_embeddings: int = 28_996 
    pad_token_idx: int = 0
    chunk_size: int = 64
    dropout_p: float = 0.1

    # Encoder
    enc_share_decoder_embs: bool = False
    enc_hidden_dim: int = 768
    enc_num_layers: int = 2
    enc_ffn_dim: int = enc_hidden_dim * 4
    enc_num_heads: int = 4
    enc_qkv_dim: int = 768
    enc_ca_layers: List[int] = field(default_factory=lambda: [0, 1])
    enc_sa_pos_bias_num_buckets: int = 128+128
    enc_ca_pos_bias_num_buckets: int = 128+64

    # Decoder
    dec_hidden_dim: int = 768
    dec_num_layers: int = 2
    dec_ffn_dim: int = dec_hidden_dim * 4
    dec_num_heads: int = 4
    dec_qkv_dim: int = 768
    dec_cca_layers: List[int] = field(default_factory=lambda: [1])
    dec_sa_pos_bias_num_buckets: int = 2048
    dec_cca_pos_bias_num_buckets: int = 64+128 

    # Generation args
    is_encoder_decoder: bool = False
    bos_token_id = None
    forced_bos_token_id = None
    forced_eos_token_id = None
    exponential_decay_length_penalty = None
    bad_words_ids = None
    typical_p = None


@dataclass
class RetroModelOutput(OrderedDict):
    hidden_states: Union[torch.Tensor, List[torch.Tensor]] = None,
    neighbour_hidden_states: Union[torch.Tensor, List[torch.Tensor]] = None

@dataclass
class RetroModelLMHeadOutput(RetroModelOutput):
    logits: torch.Tensor = None
    loss: torch.Tensor = None


def _prepare_attention_mask_for_broadcast(attention_mask):
    attention_mask = rearrange(attention_mask, "... l -> ... 1 1 l")  # Add broadcast dimensions for heads and queries
    attention_mask = (1.0 - attention_mask) * -10000.0  # Add -10000 on masked positions in score matrix
    return attention_mask


# Credit: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
class RMSNorm(nn.Module):

    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
        Root Mean Square Layer Normalization
        
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class RetroFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout_rate: float):
        super().__init__()
        self.layer_norm = RMSNorm(hidden_dim)
        self.wi_0 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.wi_1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.wo = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.gelu_act = NewGELUActivation()

    def forward(self, hidden_states):
        x = self.layer_norm(hidden_states)
        hidden_gelu = self.gelu_act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.dropout_1(x)
        x = self.wo(x)
        return hidden_states + self.dropout_2(x)


class RelativePositionBias(nn.Module):

    def __init__(self, bidirectional: bool, num_buckets: int, num_heads: int, loglinear: bool, loglinear_max_distance: Optional[int]=None):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.loglinear = loglinear
        self.loglinear_max_distance = loglinear_max_distance
        if loglinear:
            assert loglinear_max_distance is not None, "If loglinear is True, loglinear_max_distance must be set"

        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, 
            self.num_heads,
            _weight=torch.zeros((self.num_buckets, self.num_heads))  # Initialize relative pos biases as zeros
        )

    @staticmethod
    def _relative_position_bucket_linear(relative_position, bidirectional, num_buckets):
        assert num_buckets > 0
        if bidirectional:
            # ex: num_buckets=4  => min=-1 max=2
            # ex: num_buckets=5  => min=-2 max=2
            min = -(num_buckets // 2) - (num_buckets % 2 -1)
            max = num_buckets // 2
        else:
            min = -num_buckets +1
            max = 0

        buckets = torch.clamp(relative_position, min=min, max=max) - min
        return buckets

    @staticmethod
    def _relative_position_bucket_loglinear(relative_position, bidirectional, num_buckets, max_distance):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if self.loglinear:
            rp_bucket = self._relative_position_bucket_loglinear(
                relative_position,  # shape (qlen, klen)
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets,
                max_distance=self.loglinear_max_distance
            )
        else:
            rp_bucket = self._relative_position_bucket_linear(
                relative_position,
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets
            )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1])  # shape (num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen):
        return self.compute_bias(qlen, klen)  # shape (num_heads, qlen, klen)


class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        qkv_size: int,
        hidden_size: int, 
        dropout_p: float, 
        pos_bias_bidirectional: bool,
        pos_bias_num_buckets: int,
        pos_bias_num_heads: int,
        pos_bias_loglinear: bool,
        pos_bias_loglinear_max_distance: Optional[int]=None
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout_p)
        self.resid_dropout = nn.Dropout(dropout_p)
        self.final_proj = nn.Linear(in_features=qkv_size, out_features=hidden_size, bias=False)
        self.positional_bias = RelativePositionBias(
            bidirectional=pos_bias_bidirectional,
            num_buckets=pos_bias_num_buckets,
            num_heads=pos_bias_num_heads,
            loglinear=pos_bias_loglinear,
            loglinear_max_distance=pos_bias_loglinear_max_distance
        )

    def forward(self, q, k, v, attention_mask):
        # q - [..., heads, q_len, qk_dim]
        # k - [..., heads, kv_len, qk_dim]
        # v - [..., heads, kv_len, v_dim]

        attention_scores = q @ k.swapdims(-1, -2)  # - [..., heads, q_len, kv_len]

        # Apply relative positional bias
        relative_position_bias = self.positional_bias(
            qlen=attention_scores.size(-2), 
            klen=attention_scores.size(-1)
        )
        attention_scores = attention_scores + relative_position_bias

        # Apply the attention mask (for padding etc.)
        attention_scores = attention_scores + attention_mask

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.attn_dropout(attention_scores)
        output = attention_scores @ v

        # Merge heads
        output = rearrange(output, "... h l d -> ... l (h d)")

        # Final linear projection
        output = self.final_proj(output)

        # Final dropout
        output = self.resid_dropout(output)

        return output, attention_scores


class RetroCrossAttention(nn.Module):

    def __init__(self, hidden_size: int, ca_hidden_size: int, qkv_size: int, num_heads: int, dropout_p: float, pos_bias_num_buckets: int):
        super().__init__()
        self.num_heads = num_heads

        self.ln = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(in_features=hidden_size, out_features=qkv_size, bias=False)
        self.kv_proj = nn.Linear(in_features=ca_hidden_size, out_features=qkv_size * 2, bias=False)

        self.multi_head_attention = MultiHeadAttention(
            qkv_size=qkv_size,
            hidden_size=hidden_size, 
            dropout_p=dropout_p, 
            pos_bias_bidirectional=True,
            pos_bias_num_buckets=pos_bias_num_buckets,
            pos_bias_num_heads=num_heads,
            pos_bias_loglinear=False
        )
    
    def forward(self, x, ca_hidden_states, ca_attention_mask):
        # x (neighbors) - [batch, num chunks, num neighbours, neighbour len, hidden size] 
        # ca_hidden_states - [batch, num chunks, chunk len, ca hidden size]
        # ca_attention_mask - [batch, num chunks, chunk len]

        residual = x
        x = self.ln(x)

        # Calculate q, k and v
        q = self.q_proj(x)
        k, v = rearrange(self.kv_proj(ca_hidden_states), "b l m (i d) -> i b l m d", i=2)

        # Split into heads
        q = rearrange(q, "b l k r (h d) -> b l k h r d", h=self.num_heads)
        k = rearrange(k, "b l m (h d) -> b l 1 h m d", h=self.num_heads)  # Add broadcast dim for num neighbours
        v = rearrange(v, "b l m (h d) -> b l 1 h m d", h=self.num_heads)

        # Multi-head attention
        ca_attention_mask = rearrange(ca_attention_mask, "b l m -> b l 1 1 1 m")  # [batch, num chunks, num neighbours, heads, neighbour len, chunk len]
        ca_attention_mask = (1.0 - ca_attention_mask) * -10000.0

        x, attention_scores = self.multi_head_attention(q, k, v, ca_attention_mask)
        output = x + residual

        return output, attention_scores


class RetroChunkedCrossAttention(nn.Module):

    def __init__(self, hidden_size: int, ca_hidden_size: int, chunk_size: int, qkv_size:int, num_heads: int, dropout_p: float, pos_bias_num_buckets: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_heads = num_heads

        self.ln = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(in_features=hidden_size, out_features=qkv_size, bias=False)
        self.kv_proj = nn.Linear(in_features=ca_hidden_size, out_features=qkv_size * 2, bias=False)

        self.multi_head_attention = MultiHeadAttention(
            qkv_size=qkv_size,
            hidden_size=hidden_size, 
            dropout_p=dropout_p,
            pos_bias_bidirectional=True,
            pos_bias_num_buckets=pos_bias_num_buckets,
            pos_bias_num_heads=num_heads,
            pos_bias_loglinear=False
        )

    def forward(self, x, cca_hidden_states, cca_attention_mask):
        # x - [batch, seq, hidden_dim]
        # cca_hidden_states - [batch, num attending chunks, num neighbors, neighbour len, hidden size]
        # attention_mask - [batch, num attending chunks, num neighbors, neighbour len]

        if x.shape[1] < self.chunk_size:
            # If no attending chunks, we can skip CCA altogether
            return x, None

        residual = x
        x = self.ln(x)

        # Split up input into attending chunks
        pad_len = self.chunk_size-1
        pad_len += self.chunk_size - (x.shape[1] % self.chunk_size) if x.shape[1] % self.chunk_size != 0 else 0
        attending_chunks = F.pad(x[:, self.chunk_size-1:, :], (0, 0, 0, pad_len), "constant", 0)
        attending_chunks = rearrange(attending_chunks, "b (l m) d -> b l m d", m=self.chunk_size)

        # Calculate q, k, v, and split into heads
        q = rearrange(self.q_proj(attending_chunks), "b l m (h d) -> b l h m d", h=self.num_heads)
        k, v = rearrange(self.kv_proj(cca_hidden_states), "b l k r (i h d) -> i b l h (k r) d", i=2, h=self.num_heads)

        # Multi-head attention
        completely_masked_chunks = torch.all(rearrange(torch.logical_not(cca_attention_mask), "b l k r -> b l (k r)"), dim=-1)
        cca_attention_mask = rearrange(cca_attention_mask, "b l k r -> b l 1 1 (k r)")
        cca_attention_mask = (1.0 - cca_attention_mask) * -10000.0
        
        chunked_output, attention_scores = self.multi_head_attention(q, k, v, cca_attention_mask)
        chunked_output[completely_masked_chunks] = 0.  # Set chunked_output to zero for chunks which has all neighbours masked
        output = rearrange(chunked_output, "b l m d -> b (l m) d")
        output = F.pad(output[:, :-pad_len, :], (0, 0, self.chunk_size-1, 0), "constant", 0)

        output = output + residual

        return output, attention_scores


class RetroDecoderSelfAttention(nn.Module):

    def __init__(self, hidden_size: int, qkv_size: int, num_heads: int, dropout_p: float, pos_bias_num_buckets: int):
        super().__init__()
        self.num_heads = num_heads

        self.ln = RMSNorm(hidden_size)
        self.qkv_proj = nn.Linear(in_features=hidden_size, out_features=qkv_size * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            qkv_size=qkv_size,
            hidden_size=hidden_size, 
            dropout_p=dropout_p,
            pos_bias_bidirectional=False,
            pos_bias_num_buckets=pos_bias_num_buckets,
            pos_bias_num_heads=num_heads,
            pos_bias_loglinear=False
        )

    def forward(self, x, attention_mask):
        # x - [batch, seq len, hidden size]
        # attention_mask - [batch, seq len]

        residual = x
        x = self.ln(x)

        # Calculate k, q and v
        q, k, v = rearrange(self.qkv_proj(x), "b l (i d) -> i b l d", i=3)

        # Split into heads
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        # Prepare attention mask
        attention_mask = _prepare_attention_mask_for_broadcast(attention_mask)

        # Apply causal mask
        length = q.size(-2)
        causal_mask = (1.0 - torch.tril(torch.ones((length, length), dtype=torch.uint8, device=attention_mask.device)).view(1, 1, length, length)) * -10000.0
        attention_mask = attention_mask + causal_mask

        # Multi-head attention
        x, attention_scores = self.multi_head_attention(q, k, v, attention_mask)
        output = x + residual

        return output, attention_scores


class RetroEncoderSelfAttention(nn.Module):

    def __init__(self, hidden_size: int, qkv_size: int, num_heads: int, dropout_p: float, pos_bias_num_buckets: int):
        super().__init__()
        self.num_heads = num_heads

        self.ln = RMSNorm(hidden_size)
        self.qkv_proj = nn.Linear(in_features=hidden_size, out_features=qkv_size * 3, bias=False)
        self.multi_head_attention = MultiHeadAttention(
            qkv_size=qkv_size,
            hidden_size=hidden_size, 
            dropout_p=dropout_p, 
            pos_bias_bidirectional=True,
            pos_bias_num_buckets=pos_bias_num_buckets,
            pos_bias_num_heads=num_heads,
            pos_bias_loglinear=False
        )

    def forward(self, x, attention_mask):
        # x (neighbors) - [batch, num chunks, num neighbours, neighbour len, hidden size] 
        # attention_mask - [batch, num chunks, num neighbours, neighbour len]
        
        residual = x
        x = self.ln(x)

        # Calculate q, k and v
        q, k, v = rearrange(self.qkv_proj(x), "b l k r (i d) -> i b l k r d", i=3)

        # Split into heads
        q = rearrange(q, "b l k r (h d) -> b l k h r d", h=self.num_heads)
        k = rearrange(k, "b l k r (h d) -> b l k h r d", h=self.num_heads)
        v = rearrange(v, "b l k r (h d) -> b l k h r d", h=self.num_heads)

        # Multi-head attention
        attention_mask = _prepare_attention_mask_for_broadcast(attention_mask)
        x, attention_scores = self.multi_head_attention(q, k, v, attention_mask)
        output = x + residual

        return output, attention_scores


class RetroEncoder(nn.Module):

    def __init__(
        self, 
        config: RetroConfig, 
        embeddings: nn.Embedding=None
    ):
        super().__init__()
        assert embeddings is None or embeddings.embedding_dim == config.enc_hidden_dim, \
            "When sharing embeddings, the enc_hidden_dim must match the shared embedding's dim"
        self.enc_embs = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.enc_hidden_dim) \
            if embeddings is None else embeddings
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RetroEncoderSelfAttention(
                    hidden_size=config.enc_hidden_dim, 
                    qkv_size=config.enc_qkv_dim,
                    num_heads=config.enc_num_heads, 
                    dropout_p=config.dropout_p, 
                    pos_bias_num_buckets=config.enc_sa_pos_bias_num_buckets),
                RetroCrossAttention(
                    hidden_size=config.enc_hidden_dim, 
                    ca_hidden_size=config.dec_hidden_dim, 
                    qkv_size=config.enc_qkv_dim,
                    num_heads=config.enc_num_heads, 
                    dropout_p=config.dropout_p, 
                    pos_bias_num_buckets=config.enc_ca_pos_bias_num_buckets) if i in config.enc_ca_layers else None,
                RetroFeedForward(
                    hidden_dim=config.enc_hidden_dim,
                    ffn_dim=config.enc_ffn_dim,
                    dropout_rate=config.dropout_p
                )
            ])
            for i in range(config.enc_num_layers)
        ])

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        ca_hidden_states, 
        ca_attention_mask,
        return_all_hidden_states=False
    ):
        # input_ids - [batch, num chunks, num neighbours, neighbour length]
        # ca_hidden_states - [batch, num chunks, chunk length, hidden size]
        
        x = self.enc_embs(input_ids)

        all_hidden_states = [x] if return_all_hidden_states else None

        for sa, ca, ffn in self.layers:
            # Self-attention
            x, _ = sa(x, attention_mask)

            # Optional cross-attention
            if ca is not None:
                x, _ = ca(x, ca_hidden_states, ca_attention_mask)

            # Feed-forward
            x = ffn(x)

            if return_all_hidden_states:
                all_hidden_states.append(x)

        return x, all_hidden_states


class RetroModel(nn.Module):
    
    def __init__(self, config: RetroConfig):
        super().__init__()
        self.config = config

        self.dec_embs = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.dec_hidden_dim)
        self.encoder = RetroEncoder(
            config, 
            self.dec_embs if config.enc_share_decoder_embs else None,
        ) if len(config.dec_cca_layers) > 0 else None
        self.decoder = nn.ModuleList([
            nn.ModuleList([
                RetroDecoderSelfAttention(
                    hidden_size=config.dec_hidden_dim, 
                    qkv_size=config.dec_qkv_dim,
                    num_heads=config.dec_num_heads, 
                    dropout_p=config.dropout_p, 
                    pos_bias_num_buckets=config.dec_sa_pos_bias_num_buckets),
                RetroChunkedCrossAttention(
                    hidden_size=config.dec_hidden_dim, 
                    qkv_size=config.dec_qkv_dim,
                    ca_hidden_size=config.enc_hidden_dim, 
                    chunk_size=config.chunk_size, 
                    num_heads=config.dec_num_heads, 
                    dropout_p=config.dropout_p,
                    pos_bias_num_buckets=config.dec_cca_pos_bias_num_buckets) if i in config.dec_cca_layers else None,
                RetroFeedForward(
                    hidden_dim=config.dec_hidden_dim,
                    ffn_dim=config.dec_ffn_dim,
                    dropout_rate=config.dropout_p
                )
            ])
            for i in range(config.dec_num_layers)
        ])
        self.final_layer_norm = RMSNorm(config.dec_hidden_dim)

    def forward(
        self,
        input_ids,
        neighbour_ids,
        input_attention_mask=None,
        neighbour_attention_mask=None,
        return_all_hidden_states=False,
        return_all_neighbour_hidden_states=False
    ):
        # input_ids:       [batch, seq len]
        # neighbour_ids:   [batch, num chunks, num neighbours, neighbour len]

        if input_attention_mask is None:
            input_attention_mask = (input_ids != self.config.pad_token_idx).type(torch.float32).to(input_ids.device)

        if neighbour_attention_mask is None:
            neighbour_attention_mask = (neighbour_ids != self.config.pad_token_idx).type(torch.float32).to(neighbour_ids.device)

        assert math.ceil(input_ids.size(-1) / self.config.chunk_size) == neighbour_ids.size(1), \
            "input_ids must contain the same number of chunks as in neighbour_ids"
    
        # Input embeddings
        x = self.dec_embs(input_ids)

        all_hidden_states = [x] if return_all_hidden_states else None

        # Decoder
        neighbour_hidden_states = None
        for dec_layer_idx, (sa, cca, ffn) in enumerate(self.decoder):
            x, _ = sa(x, input_attention_mask)

            # If we should encode chunk neighbours
            if dec_layer_idx == min(self.config.dec_cca_layers or [-1]):
                # Split up x in chunks that the neighbour encoder can attend to
                ca_hidden_states = x
                ca_attention_mask = input_attention_mask
                if x.shape[1] % self.config.chunk_size != 0:  # Pad to full chunk length, to be able to reshape
                    ca_hidden_states = F.pad(ca_hidden_states, (0, 0, 0, self.config.chunk_size - x.shape[1] % self.config.chunk_size), mode="constant", value=0.0)
                    ca_attention_mask = F.pad(ca_attention_mask, (0, self.config.chunk_size - x.shape[1] % self.config.chunk_size), mode="constant", value=0)
                ca_hidden_states = rearrange(ca_hidden_states, "b (l m) d -> b l m d", m=self.config.chunk_size)
                ca_attention_mask = rearrange(ca_attention_mask, "b (l m) -> b l m", m=self.config.chunk_size)
                neighbour_hidden_states, all_neighbour_hidden_states = self.encoder(
                    input_ids=neighbour_ids, 
                    attention_mask=neighbour_attention_mask, 
                    ca_hidden_states=ca_hidden_states, 
                    ca_attention_mask=ca_attention_mask,
                    return_all_hidden_states=return_all_neighbour_hidden_states
                )
            
            if cca is not None:
                x, _ = cca(x, neighbour_hidden_states, neighbour_attention_mask)

            x = ffn(x)

            if return_all_hidden_states:
                all_hidden_states.append(x)

        x = self.final_layer_norm(x)

        output = RetroModelOutput(
            hidden_states=all_hidden_states if return_all_hidden_states else x,
            neighbour_hidden_states=all_neighbour_hidden_states if return_all_neighbour_hidden_states else neighbour_hidden_states,
        )

        return output


class RetroModelLMHead(nn.Module, GenerationMixin):
    main_input_name = "input_ids"

    def __init__(self, config: RetroConfig, retriever: Optional[Retriever]=None):
        super().__init__()
        self.config = config
        self.retriever = retriever

        self.base = RetroModel(config)
        self.lm_head = nn.Linear(in_features=config.dec_hidden_dim, out_features=config.num_embeddings, bias=False)
        
        self.base.dec_embs.weight.data.normal_(mean=0.0, std=0.02)

    def tie_lm_head_embeddings(self):
        # Tie embeddings to output lm_head
        self.lm_head.weight = self.base.dec_embs.weight

    def forward(
        self,
        input_ids,
        neighbour_ids,
        input_attention_mask=None,
        neighbour_attention_mask=None,
        labels=None,
        return_all_hidden_states=False,
        return_all_neighbour_hidden_states=False,
        loss_reduction="mean",
        **kwargs
    ):
        
        base_output = self.base(
            input_ids,
            neighbour_ids,
            input_attention_mask,
            neighbour_attention_mask,
            return_all_hidden_states=return_all_hidden_states,
            return_all_neighbour_hidden_states=return_all_neighbour_hidden_states
        )

        hidden_states = base_output.hidden_states[-1] if return_all_hidden_states else base_output.hidden_states
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(reduction=loss_reduction)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            if loss_reduction == "none":
                loss = loss.view(labels.shape)
        else:
            loss = None

        output = RetroModelLMHeadOutput(
            logits=logits,
            loss=loss,
            **base_output.__dict__
        )

        return output

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        assert self.retriever is not None, "No retriever is specified, can't retrieve neighbours"

        # Retrieve neighbours for all input chunks that lacks them
        num_chunks = math.ceil(input_ids.shape[1] / self.config.chunk_size)
        num_complete_chunks = input_ids.shape[1] // self.config.chunk_size
        
        if "neighbour_ids" in kwargs:
            neighbour_ids = kwargs["neighbour_ids"]
        else:
            neighbour_ids = torch.zeros((input_ids.shape[0], 0, self.retriever.num_neighbours, self.retriever.neighbour_len), dtype=torch.int64)
        
        if neighbour_ids.shape[1] < num_chunks:
            chunks_to_retrieve = range(neighbour_ids.shape[1], num_complete_chunks)
            neighbour_ids = F.pad(neighbour_ids, (0, 0, 0, 0, 0, num_chunks - neighbour_ids.shape[1]), value=self.config.pad_token_idx)
            for chunk_idx in chunks_to_retrieve:
                neighbour_ids[:, chunk_idx, :, :] = self.retriever.retrieve_neighbours(input_ids[:, chunk_idx*self.config.chunk_size : (chunk_idx+1)*self.config.chunk_size])

        return {
            "input_ids": input_ids,
            "neighbour_ids": neighbour_ids
        }


if __name__ == "__main__":
    config = RetroConfig()

    input_ids = torch.randint(0, config.num_embeddings, (2, config.chunk_size * 3))
    neighbour_ids = torch.randint(0, config.num_embeddings, (2, math.ceil(input_ids.shape[1] / config.chunk_size), 4, config.chunk_size * 2))    # neighbour tokens - (batch, num chunks, num neighbors, neighbour length)

    model = RetroModelLMHead(config)
    model(
        input_ids=input_ids, 
        neighbour_ids=neighbour_ids,
    )
