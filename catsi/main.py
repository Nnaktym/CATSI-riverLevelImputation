"""Implementation of the CATSI (Context-Aware Time Series Imputation) model."""

import logging
import os
from datetime import date
from pathlib import Path
from socket import gethostname
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from torch.nn.utils import clip_grad_norm_

from catsi.model import CATSI
from catsi.utils import AverageMeter, TimeSeriesDataSet, build_data_loader

logger = logging.getLogger("CATSI")
logger.setLevel(logging.INFO)


class ContAwareTimeSeriesImp(object):
    """Continuous-Aware Time Series Imputation"""

    def __init__(
        self,
        var_names: list[str],
        train_data: Any,
        val_data: Any,
        out_path: str,
        window_size: int,
        device: torch.device | None = None,
    ):
        # set params
        self.out_path = (
            Path(out_path) / f"{date.today():%Y%m%d}-CATSI-{gethostname()}-{os.getpid()}"
        )
        if not self.out_path.is_dir():
            self.out_path.mkdir()
        self.var_names = var_names
        self.var_names_dict = {i: item for i, item in enumerate(var_names)}
        self.num_vars = len(var_names)
        self.window_size = window_size

        # load data
        self.train_set = TimeSeriesDataSet(train_data)
        self.valid_set = TimeSeriesDataSet(val_data)

        # create model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CATSI(len(var_names), window_size).to(self.device)

    def fit(
        self,
        epochs: int = 300,
        batch_size: int = 64,
        eval_batch_size: int = 64,
        eval_epoch: int = 1,
        learning_rate: float = 1e-3,
        early_stop: bool = False,
        patience: int = 4,  # Number of epochs to wait for improvement
        shuffle: bool = False,
    ) -> None:
        # construct optimizer
        context_rnn_params = {
            "params": self.model.context_rnn.parameters(),
            "lr": learning_rate,
            "weight_decay": 5e-3,
        }
        imp_rnn_params = {
            "params": [
                p[1] for p in self.model.named_parameters() if p[0].split(".")[0] != "context_rnn"
            ],
            "lr": learning_rate,
            "weight_decay": 5e-5,
        }
        optimizer = optim.Adam([context_rnn_params, imp_rnn_params])

        train_iter = build_data_loader(self.train_set, self.device, batch_size, shuffle=shuffle)
        valid_iter = build_data_loader(
            self.valid_set, self.device, eval_batch_size, shuffle=shuffle
        )
        self.eval_batch_size = eval_batch_size

        best_val_loss = float("inf")
        patience_counter = 0

        # Training phase
        for epoch in range(epochs):
            self.model.train()

            pbar_desc = f"Epoch {epoch+1}: "
            pbar = tqdm.tqdm(total=len(train_iter), desc=pbar_desc)

            total_loss = AverageMeter()
            total_loss_eval = AverageMeter()
            for _, data in enumerate(train_iter):
                optimizer.zero_grad()
                ret = self.model(data)
                clip_grad_norm_(self.model.parameters(), 1)
                ret["loss"].backward()
                optimizer.step()

                total_loss.update(ret["loss"].item(), ret["loss_count"].item())
                total_loss_eval.update(ret["loss_eval"].item(), ret["loss_eval_count"].item())

                pbar.set_description(pbar_desc + f"Training loss={total_loss.avg:.3e}")
                pbar.update()
            pbar_desc = f"Epoch {epoch + 1} done, Training loss={total_loss.avg:.3e}"
            pbar.set_description(pbar_desc)
            pbar.close()

            if (epoch + 1) % eval_epoch == 0:
                self.evaluate(valid_iter)

            # Validation phase
            self.model.eval()
            val_loss, _, _, _ = self.evaluate(valid_iter, print_scores=True)
            print(f"Validation loss: {val_loss:.4f}")

            # Early stopping check
            if early_stop:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Training Loss: {total_loss.avg:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print("-" * 50)

        print("Training is done.")

        loss_valid, mae, mre, nrmsd = self.evaluate(valid_iter)
        with open(self.out_path / "final_eval.csv", "w") as txtfile:
            txtfile.write(f"Metrics, " + (", ".join(self.var_names)) + "\n")
            txtfile.write(f"MAE, " + (", ".join([f"{x:.3f}" for x in mae])) + "\n")
            txtfile.write(f"MRE, " + (", ".join([f"{x:.3f}" for x in mre])) + "\n")
            txtfile.write(f"nRMSD, " + (", ".join([f"{x:.3f}" for x in nrmsd])) + "\n")

        print(f"Best validation loss: {best_val_loss:.4f}")

    def evaluate(
        self,
        data_iter: Any,
        print_scores: bool = True,
    ) -> tuple[float, list[float], list[float], list[float]]:
        """Evaluate the model on the validation set."""
        self.model.eval()

        mae = [AverageMeter() for _ in range(self.num_vars)]
        mre = [AverageMeter() for _ in range(self.num_vars)]
        nrmsd = [AverageMeter() for _ in range(self.num_vars)]
        loss_valid = AverageMeter()

        for idx, data in enumerate(data_iter):
            eval_masks = data["eval_masks"]
            eval_ = data["evals"]
            eval_ = torch.FloatTensor(np.nan_to_num(eval_, nan=0)).to(self.device)

            ret = self.model(data)
            imputation = ret["imputations"]
            loss_valid.update(ret["loss"], ret["loss_count"])

            abs_err = (eval_masks * (eval_ - imputation).abs()).sum(dim=[0, 1]) / eval_masks.sum(
                dim=[0, 1]
            )
            rel_err = (eval_masks * (eval_ - imputation).abs() / eval_.clamp(min=1e-5)).sum(
                dim=[0, 1]
            ) / eval_masks.sum(dim=[0, 1])
            [mae[i].update(abs_err[i], eval_.shape[0]) for i in range(self.num_vars)]
            [mre[i].update(rel_err[i], eval_.shape[0]) for i in range(self.num_vars)]

            range_norm = 1  # max_vals - min_vals
            nsd = eval_masks * (eval_ - imputation).abs() / range_norm
            for i, (nsd_val, nsd_num) in enumerate(
                zip((nsd.norm(dim=[0, 1]) ** 2).tolist(), eval_masks.sum(dim=[0, 1]).tolist()),
            ):
                if nsd_num > 0:
                    nrmsd[i].update(nsd_val / nsd_num, nsd_num)
                else:
                    nrmsd[i].update(0.0, 1)

        mae = [x.avg for x in mae]
        mre = [x.avg for x in mre]
        nrmsd = [x.avg**0.5 for x in nrmsd]

        if print_scores:
            print("   MAE = " + ("\t".join([f"{x:.3f}" for x in mae])))
            print("   MRE = " + ("\t".join([f"{x:.3f}" for x in mre])))
            print(" nRMSD = " + ("\t".join([f"{x:.3f}" for x in nrmsd])))

        return loss_valid.avg, mae, mre, nrmsd

    def retrieve_imputation(
        self,
        data_iter: torch.utils.data.DataLoader,
        epoch: int,
        colname: str = "imp",
    ) -> pd.DataFrame:
        """Retrieve imputation results from the model."""
        self.model.eval()
        imp_dfs = []
        for _, data in enumerate(data_iter):
            eval_masks = data["eval_masks"]
            eval_ = data["evals"]

            ret = self.model(data)
            imputation = ret["imputations"]

            sids = data["sids"]
            imp_df = pd.DataFrame(
                eval_masks.nonzero().data.cpu().numpy(), columns=["sid", "tid", "colid"]
            )
            imp_df["sid"] = imp_df["sid"].map({i: sid for i, sid in enumerate(sids)})
            imp_df["epoch"] = epoch
            imp_df["analyte"] = imp_df["colid"].map(self.var_names_dict)
            imp_df[colname] = imputation[eval_masks == 1].data.cpu().numpy()
            imp_df[colname + "_feat"] = ret["feat_imp"][eval_masks == 1].data.cpu().numpy()
            imp_df[colname + "_hist"] = ret["hist_imp"][eval_masks == 1].data.cpu().numpy()

            imp_df["ground_truth"] = eval_[eval_masks == 1].data.cpu().numpy()
            imp_dfs.append(imp_df)
        if not imp_dfs:
            raise ValueError(
                "No valid data frames. Check the input data or conditions in the loop."
            )
        imp_dfs = pd.concat(imp_dfs, axis=0).set_index(["sid", "tid", "analyte", "ground_truth"])
        return imp_dfs

    def impute_test_set(
        self,
        data_set: Any,
        batch_size: int | None = None,
        ground_truth: bool = True,
    ) -> list[pd.DataFrame]:
        """Impute the test set using the trained model."""
        batch_size = batch_size or self.eval_batch_size
        if ground_truth:
            data_iter = build_data_loader(data_set, self.device, batch_size, False, testing=False)
        else:
            data_iter = build_data_loader(data_set, self.device, batch_size, False, testing=True)
        self.model.eval()

        out_dir = self.out_path / "imputations_test_set"
        out_dir.mkdir(exist_ok=True)

        imp_dfs = []
        pbar = tqdm.tqdm(desc="Generating imputation", total=len(data_iter))
        for i, data in enumerate(data_iter):
            missing_masks = 1 - data["masks"]
            ret = self.model(data)
            imputation = ret["imputations"]
            data_set[i]["imputation"] = imputation

            sids = data["sids"]
            imp_df = pd.DataFrame(
                missing_masks.nonzero().data.cpu().numpy(), columns=["sid", "tid", "colid"]
            )
            imp_df["sid"] = imp_df["sid"].map({i: sid for i, sid in enumerate(sids)})
            imp_df["analyte"] = imp_df["colid"].map(self.var_names_dict)
            imp_df["imputation"] = imputation[missing_masks == 1].data.cpu().numpy()
            if ground_truth:
                imp_df["ground_truth"] = data["evals"][missing_masks == 1].numpy()
            imp_dfs.append(imp_df)

            for p in range(len(sids)):
                seq_len = data["lengths"][p]
                time_stamps = data["time_stamps"][p, :seq_len].unsqueeze(1)
                imp = imputation[p, :seq_len, :]
                df = pd.DataFrame(
                    torch.cat([time_stamps, imp], dim=1).data.cpu().numpy(),
                    columns=["CHARTTIME"] + self.var_names,
                )
                df["CHARTTIME"] = df["CHARTTIME"].apply(int)
                df.to_csv(out_dir / f"{sids[p]}.csv", index=False)
            pbar.update()
        pbar.close()
        print(f"Done, results saved in:\n {out_dir.resolve()}")
        return imp_dfs
