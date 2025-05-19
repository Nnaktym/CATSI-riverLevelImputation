import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class MLPFeatureImputation(nn.Module):  # static feature context
    def __init__(self, input_size: int, hidden_size: int = 32):
        super(MLPFeatureImputation, self).__init__()

        self.W = Parameter(torch.Tensor(input_size, hidden_size, input_size))
        self.b = Parameter(torch.Tensor(input_size, hidden_size))
        self.nonlinear_regression = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        m = torch.ones(input_size, hidden_size, input_size)
        stdv = 1.0 / math.sqrt(input_size)
        for i in range(input_size):
            m[i, :, i] = 0
        self.register_buffer("m", m)
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:  # initializing
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor | Any) -> torch.Tensor | Any:
        hidden = torch.cat(
            tuple(
                F.linear(x, self.W[i] * Variable(self.m[i]), self.b[i]).unsqueeze(2)
                for i in range(len(self.W))
            ),
            dim=2,
        )
        z_h = self.nonlinear_regression(hidden)
        return z_h.squeeze(-1)


class InputTemporalDecay(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)
        return None

    def forward(self, d: Any) -> torch.Tensor | Any:
        gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        return torch.exp(-gamma)


class RNNContext(nn.Module):  # time series context
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, input: torch.Tensor, seq_lengths: int) -> torch.Tensor | Any:
        T_max = input.shape[1]  # batch x time x dims

        h = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
        hn = torch.zeros(input.shape[0], self.hidden_size).to(input.device)

        for t in range(T_max):
            h = self.rnn_cell(input[:, t, :], h)
            padding_mask = ((t + 1) <= seq_lengths).float().unsqueeze(1).to(input.device)
            hn = padding_mask * h + (1 - padding_mask) * hn

        return hn


class CATSI(nn.Module):  # catsi model class
    def __init__(
        self,
        num_vars: int,
        length: int,
        hidden_size: int = 64,
        context_hidden: int = 32,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.length = length
        self.hidden_size = hidden_size

        self.context_mlp = nn.Sequential(
            nn.Linear(3 * self.num_vars + 1, 2 * context_hidden),
            nn.ReLU(),
            nn.Linear(2 * context_hidden, context_hidden),
        )

        self.context_rnn = RNNContext(3 * self.num_vars, context_hidden)

        self.initial_hidden = nn.Linear(2 * context_hidden, 2 * hidden_size)
        # Linear(in_features=64, out_features=128, bias=True)

        self.initial_cell_state = nn.Tanh()
        # Tanh()

        self.rnn_cell_forward = nn.LSTMCell(2 * num_vars + 2 * context_hidden, hidden_size)
        # LSTMCell(66, 64)

        self.rnn_cell_backward = nn.LSTMCell(2 * num_vars + 2 * context_hidden, hidden_size)
        # LSTMCell(66, 64)

        self.recurrent_impute = nn.Linear(2 * hidden_size, num_vars)
        # Linear(in_features=128, out_features=1, bias=True)

        self.feature_impute = MLPFeatureImputation(num_vars)
        # MLPFeatureImputation(
        #   (nonlinear_regression): Sequential(
        #     (0): ReLU()
        #     (1): Linear(in_features=32, out_features=32, bias=True)
        #     (2): ReLU()
        #     (3): Linear(in_features=32, out_features=1, bias=True)
        #   )
        # )

        self.fuse_imputations = nn.Linear(2 * num_vars, num_vars)
        # Linear(in_features=2, out_features=1, bias=True)

    def forward(
        self,
        data: dict[str, torch.Tensor],  # data for a single batch
    ) -> dict[str, torch.Tensor]:
        seq_lengths = data["lengths"]
        values = data["values"]  # pts x time_stamps x vars = torch.Size([26, 10, 1])
        masks = data["masks"]  # observation masks
        rain_forward = data["rain_forward"]  # rain features
        rain_backward = data["rain_backward"]  # rain features

        # compute context vector, h0 and c0
        T_max = values.shape[1]  # time_stamp length
        padding_masks = torch.cat(
            tuple(
                ((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                for t in range(T_max)
            ),
            dim=1,
        )
        padding_masks = padding_masks.unsqueeze(2)  # pts x time_stamps x 1
        # should be all 1s

        # Calculate data statistics with proper handling of zero denominators

        padding_masks = torch.cat(
            tuple(
                ((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                for t in range(T_max)
            ),
            dim=1,
        )
        padding_masks = padding_masks.unsqueeze(2).repeat(
            1, 1, values.shape[2]
        )  # pts x time_stamps x vars

        data_means = values.sum(dim=1) / masks.sum(dim=1)  # pts x vars
        data_variance = ((values - data_means.unsqueeze(1)) ** 2).sum(dim=1) / (
            masks.sum(dim=1) - 1
        )
        data_stdev = data_variance**0.5
        data_missing_rate = 1 - masks.sum(dim=1) / padding_masks.sum(dim=1)
        data_stats = torch.cat(
            (seq_lengths.unsqueeze(1).float(), data_means, data_stdev, data_missing_rate), dim=1
        )

        # sum of observation masks
        mask_sum = masks.sum(dim=1)
        # should be handled in time series creation
        # Ensure we have at least one observation for each variable
        mask_sum = torch.max(mask_sum, torch.ones_like(mask_sum))
        # print(f"mask_sum shape: {mask_sum.shape}")
        # print(f"mask_sum: {mask_sum}")

        # Calculate means only for variables with observations
        data_means = torch.zeros_like(values.sum(dim=1))
        valid_mask = mask_sum > 0
        if valid_mask.any():  # 10の観察値の合計を7で割っている。これでいいのか？ -> fixed
            data_means[valid_mask] = (masks * values).sum(dim=1)[valid_mask] / mask_sum[valid_mask]

        # Calculate variance only for variables with sufficient observations
        data_variance = torch.zeros_like(data_means)  # torch.Size([44, 1])
        valid_var_mask = mask_sum > 1
        if valid_var_mask.any():
            diff_squared = (values - data_means.unsqueeze(1)) ** 2
            data_variance[valid_var_mask] = diff_squared.sum(dim=1)[valid_var_mask] / (
                mask_sum[valid_var_mask] - 1
            )

        data_stdev = torch.sqrt(data_variance)

        # Calculate missing rate with proper handling
        padding_sum = padding_masks.sum(dim=1)
        padding_sum = torch.max(padding_sum, torch.ones_like(padding_sum))
        data_missing_rate = 1 - masks.sum(dim=1) / padding_sum

        data_stats = torch.cat(
            (seq_lengths.unsqueeze(1).float(), data_means, data_stdev, data_missing_rate), dim=1
        )
        # replace all values of data_stats with 0
        data_stats = torch.where(data_stats != 0, data_stats, torch.zeros_like(data_stats))
        # sequence length, mean(*features), std(*features), missing rate(*features) are aligned in the same order
        # これが、3 * num_vars + 1の意味

        if self.training:  # evals: ground truth values for evaluation
            evals = data["evals"]  # (data["evals"] - data["min_vals"]) / min_max_norm

        # これがなんのために必要なのかわからない⇨遅延特徴量：ragged features, 自己相関？？
        x_prime = torch.zeros_like(values)
        x_prime[:, 0, :] = values[:, 0, :]
        for t in range(1, T_max):
            x_prime[:, t, :] = values[:, t - 1, :]  # replacing with the previous value
        # 値を１タイムスタンプ後ろにずらしたものを入れている

        # deleting gamma consideration ---
        gamma = torch.ones_like(x_prime)  # replace with all 1s
        # gamma is all 1s
        # In [93]: gamma.shape
        # Out[93]: torch.Size([44, 10, 1])
        # In [95]: np.unique(gamma.detach().numpy())
        # Out[95]: array([1.], dtype=float32)
        # x_decay = gamma * x_prime + (1 - gamma) * normalized_means.unsqueeze(1)
        # print("gamma", gamma)
        # decayはもともと「減衰」や「衰退」を意味します。時間的な文脈では、
        # decayは「時間が経つにつれて影響が減少すること」を指します。
        # 例えば、物理学では放射性崩壊や信号の減衰、機械学習では学習率の減衰（learning rate
        # decay）などで使われます。
        # x_prime is the values of the previous time stamp, gamma is all 1s, so x_decay is
        # previous values (= x_prime)
        x_complement = (masks * values + (1 - masks) * x_prime) * padding_masks
        # replacing non-ovserved values with x_decay, with keeping the observed values
        # x_decay is the imputed values
        # print(f"x_complement: {x_complement}")

        context_mlp = self.context_mlp(data_stats)  # multi-layer perceptron (statistical contexts)
        context_rnn = self.context_rnn(
            torch.cat((x_complement, rain_forward, rain_backward), dim=-1),
            seq_lengths,  # test rain here
        )  # temporal contexts
        context_vec = torch.cat((context_mlp, context_rnn), dim=1)
        h = self.initial_hidden(context_vec)  #  torch.Size([2, 128])
        c = self.initial_cell_state(h)  #  torch.Size([2, 128])

        inputs = torch.cat(
            [x_complement, masks, context_vec.unsqueeze(1).repeat(1, T_max, 1)], dim=-1
        )  # context_vecは皆同じみたい torch.Size([3, 10, 68])

        h_forward, c_forward = h[:, : self.hidden_size], c[:, : self.hidden_size]
        h_backward, c_backward = h[:, self.hidden_size :], c[:, self.hidden_size :]
        hiddens_forward = h[:, : self.hidden_size].unsqueeze(1)
        hiddens_backward = h[:, self.hidden_size :].unsqueeze(1)
        for t in range(T_max - 1):
            h_forward, c_forward = self.rnn_cell_forward(inputs[:, t, :], (h_forward, c_forward))
            h_backward, c_backward = self.rnn_cell_backward(
                inputs[:, T_max - 1 - t, :], (h_backward, c_backward)
            )
            hiddens_forward = torch.cat((hiddens_forward, h_forward.unsqueeze(1)), dim=1)
            # torch.Size([2, 10, 64])
            hiddens_backward = torch.cat((h_backward.unsqueeze(1), hiddens_backward), dim=1)
            # torch.Size([2, 10, 64])

        rnn_imp = self.recurrent_impute(  # recurrent imputations
            torch.cat((hiddens_forward, hiddens_backward), dim=2)
        )  #  torch.Size([2, 10, 1])
        # feature-based imputation
        feat_imp = self.feature_impute(x_complement)  # .squeeze(-1)

        # imputation fusion
        # tentatively remove masks (5/10) !!! the masks can be used to define synthetic NA???
        beta = torch.sigmoid(self.fuse_imputations(torch.cat((gamma, masks), dim=-1)))
        imp_fusion = beta * feat_imp + (1 - beta) * rnn_imp
        final_imp = masks * values + (1 - masks) * imp_fusion

        # rnn_loss = F.mse_loss(
        #     rnn_imp * masks, normalized_values * masks, reduction="sum"
        # )  # is masks needed?
        # feat_loss = F.mse_loss(feat_imp * masks, normalized_values * masks, reduction="sum")
        # fusion_loss = F.mse_loss(imp_fusion * masks, normalized_values * masks, reduction="sum")
        # total_loss = rnn_loss + feat_loss + fusion_loss
        rnn_loss = F.mse_loss(rnn_imp, values, reduction="sum")  # is masks needed?
        feat_loss = F.mse_loss(feat_imp, values, reduction="sum")
        fusion_loss = F.mse_loss(imp_fusion, values, reduction="sum")
        total_loss = rnn_loss + feat_loss + fusion_loss

        if self.training:  # data["eval_masks"] is defined only for missing values
            rnn_loss_eval = F.mse_loss(
                rnn_imp * data["eval_masks"], evals * data["eval_masks"], reduction="sum"
            )
            feat_loss_eval = F.mse_loss(
                feat_imp * data["eval_masks"],
                evals * data["eval_masks"],
                reduction="sum",
            )
            fusion_loss_eval = F.mse_loss(
                imp_fusion * data["eval_masks"],
                evals * data["eval_masks"],
                reduction="sum",
            )
            total_loss_eval = rnn_loss_eval + feat_loss_eval + fusion_loss_eval

        def rescale(x: torch.Tensor) -> torch.Tensor:
            """Rescale the imputed values to the original range."""
            return torch.where(
                # padding_masks == 1, x * min_max_norm + data["min_vals"], padding_masks
                padding_masks == 1,
                x,
                padding_masks,
            )

        feat_imp = rescale(feat_imp)
        rnn_imp = rescale(rnn_imp)
        final_imp = rescale(final_imp)

        out_dict = {
            "loss": total_loss / masks.sum(),
            "verbose_loss": [
                ("rnn_loss", rnn_loss / masks.sum(), masks.sum()),
                ("feat_loss", feat_loss / masks.sum(), masks.sum()),
                ("fusion_loss", fusion_loss / masks.sum(), masks.sum()),
            ],
            "loss_count": masks.sum(),
            "imputations": final_imp,
            "feat_imp": feat_imp,
            "hist_imp": rnn_imp,
        }
        if self.training:
            out_dict["loss_eval"] = total_loss_eval / data["eval_masks"].sum()
            out_dict["loss_eval_count"] = data["eval_masks"].sum()
            out_dict["verbose_loss"] += [
                (
                    "rnn_loss_eval",
                    rnn_loss_eval / data["eval_masks"].sum(),
                    data["eval_masks"].sum(),
                ),
                (
                    "feat_loss_eval",
                    feat_loss_eval / data["eval_masks"].sum(),
                    data["eval_masks"].sum(),
                ),
                (
                    "fusion_loss_eval",
                    fusion_loss_eval / data["eval_masks"].sum(),
                    data["eval_masks"].sum(),
                ),
            ]

        return out_dict
