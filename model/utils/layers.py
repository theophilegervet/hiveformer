import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_custom_attention import MultiheadCustomAttention


class ParallelAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8, pre_norm=False,
                 self_attention1=True, self_attention2=True,
                 cross_attention1=True, cross_attention2=True,
                 apply_ffn=True,
                 slot_attention12=False, slot_attention21=False,
                 rotary_pe=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention1 = self_attention1
        self.self_attention2 = self_attention2
        self.cross_attention1 = cross_attention1
        self.cross_attention2 = cross_attention2
        self.apply_ffn = apply_ffn
        self.rotary_pe = rotary_pe

        # Self-attention for seq1
        if self.self_attention1:
            self.sa1 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_1 = nn.Dropout(dropout)
            self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        if self.self_attention2:
            self.sa2 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_2 = nn.Dropout(dropout)
            self.norm_2 = nn.LayerNorm(d_model)

        # Cross attention from seq1 to seq2
        self.norm_12 = None
        if cross_attention1:
            self.cross_12 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout,
                slot_competition=slot_attention12
            )
            self.dropout_12 = nn.Dropout(dropout)
            self.norm_12 = nn.LayerNorm(d_model)

        # Cross attention from seq2 to seq1
        self.norm_21 = None
        if cross_attention2:
            self.cross_21 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout,
                slot_competition=slot_attention21
            )
            self.dropout_21 = nn.Dropout(dropout)
            self.norm_21 = nn.LayerNorm(d_model)

        # FFN-1
        if self_attention1 or cross_attention1:
            self.ffn_12 = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, d_model),
                nn.Dropout(dropout)
            )
            self.norm_122 = nn.LayerNorm(d_model)

        # FFN-2
        if self_attention2 or cross_attention2:
            self.ffn_21 = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, d_model),
                nn.Dropout(dropout)
            )
            self.norm_212 = nn.LayerNorm(d_model)

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, seq1, seq1_key_padding_mask, seq2,
                seq2_key_padding_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        rot_args = {}

        # Create key, query, value for seq1, seq2
        q1 = k1 = v1 = self._norm(seq1, self.norm_12, self.pre_norm)
        q2 = k2 = v2 = self._norm(seq2, self.norm_21, self.pre_norm)
        if not self.rotary_pe:
            q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
        q1 = self.with_pos_embed(q1, seq1_sem_pos)
        k1 = self.with_pos_embed(k1, seq1_sem_pos)
        q2 = self.with_pos_embed(q2, seq2_sem_pos)
        k2 = self.with_pos_embed(k2, seq2_sem_pos)

        # Cross-attention from seq1 to seq2
        if self.cross_attention1:
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq1_pos, seq2_pos)
            seq1b = self.cross_12(
                query=q1.transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_12(seq1b)
            seq1 = self._norm(seq1, self.norm_12, not self.pre_norm)

        # Cross-attention from seq2 to seq1
        if self.cross_attention2:
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq2_pos, seq1_pos)
            seq2b = self.cross_21(
                query=q2.transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_21(seq2b)
            seq2 = self._norm(seq2, self.norm_21, not self.pre_norm)

        # Self-attention for seq1
        if self.self_attention1:
            q1 = k1 = v1 = self._norm(seq1, self.norm_1, self.pre_norm)
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq1_pos, seq1_pos)
            else:
                q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q1 = self.with_pos_embed(q1, seq1_sem_pos)
            k1 = self.with_pos_embed(k1, seq1_sem_pos)
            seq1b = self.sa1(
                query=q1.transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_1(seq1b)
            seq1 = self._norm(seq1, self.norm_1, not self.pre_norm)

        # Self-attention for seq2
        if self.self_attention2:
            q2 = k2 = v2 = self._norm(seq2, self.norm_2, self.pre_norm)
            if self.rotary_pe:
                rot_args['rotary_pe'] = (seq2_pos, seq2_pos)
            else:
                q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
            q2 = self.with_pos_embed(q2, seq2_sem_pos)
            k2 = self.with_pos_embed(k2, seq2_sem_pos)
            seq2b = self.sa2(
                query=q2.transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_2(seq2b)
            seq2 = self._norm(seq2, self.norm_2, not self.pre_norm)

        # FFN-1
        if (self.self_attention1 or self.cross_attention1) and self.apply_ffn:
            seq1 = self._norm(seq1, self.norm_122, self.pre_norm)
            seq1 = seq1 + self.ffn_12(seq1)
            seq1 = self._norm(seq1, self.norm_122, not self.pre_norm)

        # FFN-2
        if (self.self_attention2 or self.cross_attention2) and self.apply_ffn:
            seq2 = self._norm(seq2, self.norm_212, self.pre_norm)
            seq2 = seq2 + self.ffn_21(seq2)
            seq2 = self._norm(seq2, self.norm_212, not self.pre_norm)

        return seq1, seq2


class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, value, query_pos=None, value_pos=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query=(query + query_pos) if query_pos is not None else query,
            key=(value + value_pos) if value_pos is not None else value,
            value=value
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights


class RelativeCrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, query_pos=None, value_pos=None, pad_mask=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query=query,
            key=value,
            value=value,
            rotary_pe=(query_pos, value_pos) if query_pos is not None else None,
            key_padding_mask=pad_mask
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)


class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class RelativeCrossAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers):
        super().__init__()

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, query, value, query_pos=None, value_pos=None):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](query, value, query_pos, value_pos)
            query = self.ffw_layers[i](query)
            output.append(query)
        return output


class TaskSpecificRelativeCrossAttentionLayer(nn.Module):
    """Relative cross attention layer with task specific biases."""
    def __init__(self, embedding_dim, num_heads, task_ids, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.task_biases = nn.ParameterDict()
        for task in task_ids:
            self.task_biases[f"{task}_multihead_attn_in_proj_bias"] = nn.Parameter(
                torch.zeros_like(self.multihead_attn.in_proj_bias))
            self.task_biases[f"{task}_multihead_attn_out_proj_bias"] = nn.Parameter(
                torch.zeros_like(self.multihead_attn.out_proj.bias))
            self.task_biases[f"{task}_norm_bias"] = nn.Parameter(
                torch.zeros_like(self.norm.bias))

    def forward(self, task_id, query, value, query_pos=None, value_pos=None):
        output = torch.zeros_like(query)

        for t in torch.unique(task_id):
            self.multihead_attn.in_proj_bias = self.task_biases[f"{t}_multihead_attn_in_proj_bias"]
            self.multihead_attn.out_proj.bias = self.task_biases[f"{t}_multihead_attn_out_proj_bias"]
            self.norm.bias = self.task_biases[f"{t}_norm_bias"]

            query_task = query[:, task_id == t]
            value_task = value[:, task_id == t]
            if query_pos is not None:
                query_pos_task = query_pos[task_id == t]
                value_pos_task = value_pos[task_id == t]
            attn_output_task, attn_output_weights = self.multihead_attn(
                query=query_task,
                key=value_task,
                value=value_task,
                rotary_pe=(query_pos_task, value_pos_task) if query_pos is not None else None
            )
            output_task = query_task + self.dropout(attn_output_task)
            output_task = self.norm(output_task)
            output[:, task_id == t] = output_task

        return output, None


class TaskSpecificFeedforwardLayer(nn.Module):
    """Feedforward layer with task specific biases."""
    def __init__(self, embedding_dim, hidden_dim, task_ids, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

        self.task_biases = nn.ParameterDict()
        for task in task_ids:
            self.task_biases[f"{task}_linear1_bias"] = nn.Parameter(
                torch.zeros_like(self.linear1.bias))
            self.task_biases[f"{task}_linear2_bias"] = nn.Parameter(
                torch.zeros_like(self.linear2.bias))
            self.task_biases[f"{task}_norm_bias"] = nn.Parameter(
                torch.zeros_like(self.norm.bias))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, task_id, x):
        output = torch.zeros_like(x)

        for t in torch.unique(task_id):
            self.linear1.bias = self.task_biases[f"{t}_linear1_bias"]
            self.linear2.bias = self.task_biases[f"{t}_linear2_bias"]
            self.norm.bias = self.task_biases[f"{t}_norm_bias"]

            x_task = x[:, task_id == t]
            output_task = self.linear2(self.dropout(self.activation(self.linear1(x_task))))
            output_task = x_task + self.dropout(output_task)
            output_task = self.norm(output_task)
            output[:, task_id == t] = output_task

        return output


class TaskSpecificRelativeCrossAttentionModule(nn.Module):
    """Relative cross attention module with task specific biases."""
    def __init__(self, embedding_dim, num_attn_heads, num_layers, task_ids):
        super().__init__()

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(TaskSpecificRelativeCrossAttentionLayer(embedding_dim, num_attn_heads, task_ids))
            self.ffw_layers.append(TaskSpecificFeedforwardLayer(embedding_dim, embedding_dim, task_ids))

    def forward(self, task_id, query, value, query_pos=None, value_pos=None):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](task_id, query, value, query_pos, value_pos)
            query = self.ffw_layers[i](task_id, query)
            output.append(query)
        return output
