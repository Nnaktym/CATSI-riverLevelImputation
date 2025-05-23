"""Script to train and evaluate the model."""

import argparse
import configparser
import datetime
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from catsi.catsi import ContAwareTimeSeriesImp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatsiDataset(object):
    """Dataset for CATSI"""

    def __init__(
        self,
        rain_file_name: str,
        river_file_name: str,
        flood_duration_file_name: str,
        missing_rate: list[float | Any],
        phase: str,  # "training" or "testing"
        n_repeat: int = 1,
        var_names: list[str] | None = None,
        diff: bool = True,
        rain_smoothing: int = 12,
        window_size: int = 20,
        random_state: int = 0,
        dataset_dir: str = "data",
        result_dir: str = "result",
    ):
        self.rain_file_name = rain_file_name
        self.river_file_name = river_file_name
        self.flood_duration_file_name = flood_duration_file_name
        self.diff = diff
        self.var_names = var_names
        self.rain_smoothing = rain_smoothing
        self.window_size = window_size
        self.random_state = random_state

        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.saving_dir = Path(self.result_dir) / current_time
        self.saving_dir.mkdir(parents=True, exist_ok=True)

        self.rain, self.river_org, self.flood_duration, self.var_names, self.timestamps = (
            self.import_raw_data()
        )
        if self.diff:
            self.river = self.river_org.copy().diff()
        else:
            self.river = self.river_org.copy()
        rain_acc = moving_average(self.rain, rain_smoothing)

        self.river_scalers = fit_scalers(self.river)
        self.rain_scalers = fit_scalers(self.rain)
        self.river_scaler = fit_scaler(self.river)
        self.rain_scaler = fit_scaler(self.rain)
        self.rain_ma_scaler = fit_scaler(rain_acc)

        self.river_data = self.river_scaler.transform(self.river)
        self.rain_data = self.rain_scaler.transform(self.rain)
        self.rain_acc_data = self.rain_ma_scaler.transform(rain_acc)

        if phase == "training":
            self.samples = {
                "train": [{"missing_rate": x, "n_repeat": n_repeat} for x in missing_rate],
                "val": [{"missing_rate": x, "n_repeat": n_repeat} for x in missing_rate],
            }
        elif phase == "testing":
            assert len(missing_rate) == 1, "For testing, only one missing rate is allowed."
            self.samples = {
                "test": [{"missing_rate": x, "n_repeat": 1} for x in missing_rate],
            }
        else:
            raise ValueError("Invalid phase. Use 'training' or 'testing'.")
        self.dataset = self.generate_catsi_dataset(phase)

    def __len__(self) -> dict[str, int]:
        """Return the length of the dataset."""
        return {k: len(v) for k, v in self.dataset.items()}

    def __getitem__(self, stage: str, idx: int) -> Any:
        """Return the item at the given index."""
        return self.dataset[stage][idx]

    def import_raw_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], list[datetime.datetime]]:
        """Import raw rain, river, and flood duration data."""
        rain_df = pd.read_csv(
            Path(self.dataset_dir) / self.rain_file_name,
            index_col="timestamps",
            parse_dates=True,
        )
        river_df = pd.read_csv(
            Path(self.dataset_dir) / self.river_file_name,
            index_col="timestamps",
            parse_dates=True,
        )
        assert rain_df.index.equals(
            river_df.index
        ), "Rain and river data must have the same timestamps"
        timestamps = river_df.index.to_list()
        with open(
            Path(self.dataset_dir) / self.flood_duration_file_name,
            "r",
        ) as f:
            flood_duration = json.load(f)
        if self.var_names is None:
            var_names = river_df.columns.to_list()
        else:
            var_names = self.var_names
        assert list(flood_duration.keys()) == [
            "train",
            "val",
            "test",
        ], "Flood duration keys incorrect"
        return rain_df[var_names], river_df[var_names], flood_duration, var_names, timestamps

    def generate_catsi_dataset(
        self,
        phase: str,
    ) -> dict[str, Any]:
        """Generate sequencial data for CATSI"""
        sequences: dict[str, Any] = {}
        if phase == "training":
            stages = ["train", "val"]
        elif phase == "testing":
            stages = ["test"]
        self.river_flood: dict[str, list[Any]] = {k: [] for k in stages}
        river_mean = np.nanmean(self.rain_data, axis=0)
        river_std = np.nanstd(self.rain_data, axis=0)
        for stage in stages:
            idx = 0
            logger.info(f"Generating {stage} dataset")
            river_flood_data = slice_data_for_flood(
                pd.DataFrame(self.river_data, index=self.timestamps, columns=self.var_names),
                self.flood_duration[stage],
                self.var_names,
            )
            rain_flood_data = slice_data_for_flood(
                pd.DataFrame(self.rain_data, index=self.timestamps, columns=self.var_names),
                self.flood_duration[stage],
                self.var_names,
            )
            rain_flood_acc_data = slice_data_for_flood(
                pd.DataFrame(self.rain_acc_data, index=self.timestamps, columns=self.var_names),
                self.flood_duration[stage],
                self.var_names,
            )
            assert len(river_flood_data) == len(rain_flood_acc_data)
            assert len(rain_flood_data) == len(rain_flood_acc_data)
            self.river_flood[stage] = river_flood_data
            for sample_info in self.samples[stage]:
                logger.info(f"-------------- {sample_info}")
                missing_rate, n_repeat = sample_info["missing_rate"], sample_info["n_repeat"]
                seeds = np.arange(n_repeat) + self.random_state
                all_sequence = []
                for seed in seeds:
                    np.random.seed(seed)
                    for river_item, rain_item, rain_acc_item in zip(
                        river_flood_data,
                        rain_flood_data,
                        rain_flood_acc_data,
                    ):
                        if len(river_item) < self.window_size:
                            continue
                        # 1: nan in the original data
                        org_nan_mask = river_item.isna().astype(int).values
                        # 1: synthetically added nan
                        synth_nan_mask = (
                            np.random.rand(len(river_item), len(self.var_names)) < missing_rate
                        ).astype(int)
                        synth_nan_mask = synth_nan_mask * (1 - org_nan_mask)
                        # 1: non-nan (no need to impute)
                        observed_mask = 1 - (org_nan_mask + synth_nan_mask)
                        # apply synthetic nan
                        river_item_with_nan = river_item.copy()
                        river_item_with_nan[synth_nan_mask == 1] = np.nan
                        # linear interpolation as initial value
                        river_item_interp = pd.DataFrame(river_item).interpolate(
                            method="linear", limit_direction="both"
                        )
                        for i in range(0, len(river_item) - self.window_size + 1):
                            seq = river_item[i : i + self.window_size]
                            if len(seq) < self.window_size:
                                continue
                            sequence_data = {
                                "pt_with_na": river_item_interp[i : i + self.window_size].values,
                                "pt_with_na_org": river_item_with_nan[
                                    i : i + self.window_size
                                ].values,
                                "pt_ground_truth": seq.values,
                                "rain": rain_item[i : i + self.window_size].values,
                                "rain_accumulation": rain_acc_item[i : i + self.window_size].values,
                                "timestamps_dt": seq.index.to_list(),
                                "observed_mask": observed_mask[i : i + self.window_size],
                                "eval_mask": synth_nan_mask[i : i + self.window_size],
                                "length": self.window_size,
                                "sid": idx,
                                "vars": self.var_names,
                                "time_stamps": np.arange(len(seq)),
                                "river_mean": river_mean,
                                "river_std": river_std,
                            }
                            all_sequence.append(sequence_data)
                            idx += 1
            sequences[stage] = all_sequence
            print(f"{stage} dataset size: {len(all_sequence)}")
        return sequences


def load_config(config_path: str) -> configparser.ConfigParser:
    """Load configuration from a specified config.ini file."""
    config = configparser.ConfigParser()
    config.read(config_path)  # Use the provided file path
    return config


def slice_data_for_flood(
    data: pd.DataFrame,
    flood_duration: list[datetime.datetime, datetime.datetime],
    var_names: list[str] | None,
) -> list[pd.DataFrame]:
    """Slice data for flood duration."""
    if var_names is None:
        var_names = data.columns.to_list()
    flood_duration_data = []
    for start_time, end_time in flood_duration:
        mask = (data.index >= start_time) & (data.index <= end_time)
        flood_duration_data.append(data[mask][var_names])
    return flood_duration_data


def fit_scalers(
    data: pd.DataFrame,
) -> dict[str, MinMaxScaler]:
    """Fit scalers for each column in the data."""
    scalers: dict[str, MinMaxScaler] = {}
    for col in data.columns:
        scaler = MinMaxScaler()
        scaler.fit(data[[col]])
        scalers[col] = scaler
    return scalers


def fit_scaler(
    data: pd.DataFrame,
) -> MinMaxScaler:
    """Fit scalers for the entire data."""
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler


def moving_average(
    data: pd.DataFrame,
    window_size: int,
    type: str = "forward",
) -> pd.DataFrame:
    """Generate moving average for data."""
    if type == "forward":
        return data.rolling(window=window_size, min_periods=1).mean()
    elif type == "backward":
        return data[::-1].rolling(window=window_size, min_periods=1).mean()[::-1]
    else:
        raise ValueError("Invalid type. Use 'forward' or 'backward'.")


def create_comparison_table(
    imputation: pd.DataFrame,
    ground_truth: pd.DataFrame,
    var: str,
) -> pd.DataFrame:
    """Create comparison table for imputed and ground truth values."""
    logger.info(f"Create comparison table for var: {var}")
    actual = ground_truth[[var]].rename(columns={var: "actual"})
    pred = imputation[imputation["var_name"] == var]
    comparison_table = actual.copy()
    for _, row in pred.iterrows():
        idx = row["idx"]
        assert var == row["var_name"]
        imputed_value = row["imputation"]
        comparison_table.loc[comparison_table.index[idx], ["pred"]] = imputed_value
    comparison_table["observed_mask"] = 0
    comparison_table.loc[comparison_table["pred"].isna(), ["observed_mask"]] = 1
    comparison_table["val_mask"] = 0
    comparison_table.loc[
        (comparison_table["actual"].notna()) & (comparison_table["pred"].notna()),
        ["val_mask"],
    ] = 1
    return comparison_table


def generate_imputation(
    model: ContAwareTimeSeriesImp,
    test_set: list[dict[str, Any]],
) -> pd.DataFrame:
    """Get imputed result. Note that the the value is scaled to [0, 1]."""
    imp_list = model.impute_test_set(test_set, batch_size=1, ground_truth=True)
    imp = pd.concat(imp_list)
    imp["idx"] = imp["sid"] + imp["tid"]
    imp = imp[["idx", "var_name", "imputation"]].groupby(["idx", "var_name"], as_index=False).mean()
    return imp


def unscale(
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Unscale the data using the scaler."""
    if cols is None:
        cols = data.columns
    other_cols = [col for col in data.columns if col not in cols]
    scaled_data = pd.DataFrame(
        scaler.inverse_transform(data[cols]),
        index=data.index,
        columns=cols,
    )
    scaled_data = pd.concat([scaled_data, data[other_cols]], axis=1)
    return scaled_data


def antiderivative(
    act_original: pd.Series,
    comparison_table: pd.DataFrame,
) -> pd.DataFrame:
    """Antiderivative for the original data."""
    pred_original = []
    for i in range(len(comparison_table)):
        row = comparison_table.iloc[i]
        if i == 0:
            prev_value = act_original.iloc[0]
            pred_original.append(np.nan)
            continue
        imputed_value = row["pred"]
        if pd.isna(imputed_value):
            prev_value = act_original.iloc[i]
            # pred_original.append(np.nan)
            pred_original.append(prev_value)  # for visualization
        else:
            prev_value += imputed_value
            pred_original.append(prev_value)
    new_table = comparison_table.copy()
    new_table.loc[:, ["pred"]] = pred_original
    new_table.loc[:, ["actual"]] = act_original.values
    new_table = new_table.iloc[1:]  # because the first value is nan
    return new_table


def add_linear_interpolation(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add linear interpolation for comparison"""
    assert "actual" in data.columns
    assert "pred" in data.columns
    data.loc[:, ["linear_interp"]] = data["actual"].values
    data.loc[data["observed_mask"] == 0, ["linear_interp"]] = np.nan
    data.loc[:, ["linear_interp"]] = data["linear_interp"].interpolate(
        method="linear", limit_direction="both"
    )
    return data


def calculate_scores(
    comparison_table: pd.DataFrame,
) -> tuple[float, float]:
    """Calculate RMSE for imputed and linear interpolation values."""
    data = comparison_table[comparison_table["val_mask"] == 1]
    catsi_rmse = mean_squared_error(data["actual"], data["pred"], squared=True)
    linear_rmse = mean_squared_error(data["actual"], data["linear_interp"], squared=True)
    return catsi_rmse, linear_rmse


def create_plot(
    data: pd.DataFrame,
    var: str,
    missing_rate: float,
    saving_dir: str | Path,
) -> str:
    """Create plot for imputed and ground truth values."""
    plt.figure(figsize=(8, 6))
    plt.plot(data["pred"], label="CATSI imputation", color="red")
    plt.plot(data["linear_interp"], label="linear interpolation", color="green")
    plt.plot(data["actual"], label="ground truth", color="blue")
    plt.title(f"missing rate: {missing_rate}")
    plt.fill_between(
        data.index,
        data[["actual", "pred", "linear_interp"]].min(axis=1).min(),
        data[["actual", "pred", "linear_interp"]].max(axis=1).max() * 1.1,
        where=data["observed_mask"] == 0,
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="missing period",
    )
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    saving_path = Path(saving_dir) / f"{var}_{missing_rate}.png"
    plt.savefig(saving_path)
    plt.close()
    return str(saving_path)


def main() -> None:
    """Main function to train and evaluate the model."""
    parser = argparse.ArgumentParser(description="Train and evaluate the CATSI model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = config.get("PATH", "dataset_dir")
    result_dir = config.get("PATH", "result_dir")
    rain_file_name = config.get("PATH", "rain_file_name")
    river_file_name = config.get("PATH", "river_file_name")
    flood_duration_file_name = config.get("PATH", "flood_duration_file_name")
    diff = config.getboolean("FEATURE_GENERATION", "diff")
    rain_smoothing = config.getint("FEATURE_GENERATION", "rain_smoothing")
    window_size = config.getint("FEATURE_GENERATION", "window_size")
    n_repeat = config.getint("SAMPLE_GENERATION", "n_repeat")
    random_state = config.getint("SAMPLE_GENERATION", "random_state")
    train_missing_rate = config.get("SAMPLE_GENERATION", "train_missing_rate")
    train_missing_rate = [float(x) for x in train_missing_rate.split(",")]
    test_missing_rate = config.get("SAMPLE_GENERATION", "test_missing_rate")
    test_missing_rate = [float(x) for x in test_missing_rate.split(",")]

    # Create dataset
    catsi_dataset = CatsiDataset(
        rain_file_name=rain_file_name,
        river_file_name=river_file_name,
        flood_duration_file_name=flood_duration_file_name,
        diff=diff,
        rain_smoothing=rain_smoothing,
        window_size=window_size,
        random_state=random_state,
        dataset_dir=dataset_dir,
        result_dir=result_dir,
        phase="training",
        n_repeat=n_repeat,
        missing_rate=train_missing_rate,
    )
    with open(catsi_dataset.saving_dir / "catsi_dataset.pkl", "wb") as f:
        pickle.dump(catsi_dataset, f)

    # Training
    model = ContAwareTimeSeriesImp(
        var_names=catsi_dataset.var_names,
        train_data=catsi_dataset.dataset["train"],
        val_data=catsi_dataset.dataset["val"],
        window_size=catsi_dataset.window_size,
        out_path=catsi_dataset.saving_dir,
        device=DEVICE,
    )
    model.fit(
        epochs=1000,
        batch_size=32,
        eval_batch_size=32,
        learning_rate=1e-3,
        early_stop=True,
        shuffle=False,
    )
    model_save_path = Path(catsi_dataset.saving_dir) / "model.pth"
    torch.save(model.model.state_dict(), model_save_path)
    logger.info(f"Trained model saved to {model_save_path}")

    # Evaluation
    scores = pd.DataFrame()
    for missing_rate in test_missing_rate:
        logger.info(f"Testing with missing rate: {missing_rate}")
        test_catsi_dataset = CatsiDataset(
            rain_file_name=rain_file_name,
            river_file_name=river_file_name,
            flood_duration_file_name=flood_duration_file_name,
            diff=diff,
            rain_smoothing=rain_smoothing,
            window_size=window_size,
            random_state=random_state,
            dataset_dir=dataset_dir,
            result_dir=result_dir,
            phase="testing",
            missing_rate=[missing_rate],
        )
        imputed = generate_imputation(model, test_catsi_dataset.dataset["test"])
        for var in test_catsi_dataset.var_names:
            logger.info(f"Testing variable: {var}")
            comparison_table = create_comparison_table(
                imputation=imputed,
                ground_truth=test_catsi_dataset.river_flood["test"][0],
                var=var,
            )
            comparison_table = unscale(
                data=comparison_table,
                scaler=test_catsi_dataset.river_scalers[var],
                cols=["pred", "actual"],
            )
            if diff:
                comparison_table = antiderivative(
                    act_original=test_catsi_dataset.river_org.loc[
                        comparison_table.index, var
                    ].rename(
                        "act_original",
                    ),
                    comparison_table=comparison_table,
                )
            comparison_table = add_linear_interpolation(
                data=comparison_table,
            )
            catsi_rmse, linear_rmse = calculate_scores(comparison_table)
            logger.info(f"catsi_rmse: {catsi_rmse}, linear_rmse: {linear_rmse}")
            score = pd.DataFrame(
                {
                    "var_name": [var],
                    "missing_rate": [missing_rate],
                    "catsi_rmse": [catsi_rmse],
                    "linear_rmse": [linear_rmse],
                }
            )
            scores = pd.concat([scores, score], axis=0)
            fig_path = create_plot(
                data=comparison_table,
                var=var,
                missing_rate=missing_rate,
                saving_dir=catsi_dataset.saving_dir,
            )
            logger.info(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
