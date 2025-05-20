"""Script to train and evaluate the model."""

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
from sklearn.preprocessing import MinMaxScaler

from catsi.main import ContAwareTimeSeriesImp

logger = logging.getLogger(__name__)

dataset_dir = "data"
result_dir = "result"

rain_file_name = "rain_intensity.csv"
river_file_name = "river_level.csv"
flood_duration_file_name = "flood_duration.json"
diff = True
var_names = None
rain_smoothing = 12
window_size = 20
random_state = 0
samples = {
    "train": [
        {"missing_rate": 0.1, "n_repeat": 100},
        {"missing_rate": 0.2, "n_repeat": 10},
    ],
    "val": [
        {"missing_rate": 0.1, "n_repeat": 10},
        {"missing_rate": 0.2, "n_repeat": 1},
    ],
    "test": [
        {"missing_rate": 0.2, "n_repeat": 1},
    ],
}


class CatsiDataset(object):
    """Dataset for CATSI"""

    def __init__(
        self,
        rain_file_name: str,
        river_file_name: str,
        flood_duration_file_name: str,
        diff: bool = False,
        var_names: list[str] | None = None,
        rain_smoothing: int = 12,
        window_size: int = 20,
        random_state: int = 0,
        samples: dict[str, Any] = {
            "train": [
                {"missing_rate": 0.1, "n_repeat": 100},
                {"missing_rate": 0.2, "n_repeat": 10},
            ],
            "val": [
                {"missing_rate": 0.1, "n_repeat": 10},
                {"missing_rate": 0.2, "n_repeat": 1},
            ],
            "test": [
                {"missing_rate": 0.2, "n_repeat": 1},
            ],
        },
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
        self.samples = samples

        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.saving_dir = Path(self.result_dir) / current_time
        self.saving_dir.mkdir(parents=True, exist_ok=True)

        rain, river, self.flood_duration, self.var_names, self.timestamps = self.import_raw_data()
        rain_forward = moving_average(rain, rain_smoothing, "forward")
        rain_backward = moving_average(rain, rain_smoothing, "backward")

        self.river_scalers = fit_scalers(river)
        self.rain_scalers = fit_scalers(rain)
        self.river_scaler = fit_scaler(river)
        self.rain_scaler = fit_scaler(rain)
        self.rain_ma_scaler = fit_scaler(pd.concat([rain_forward, rain_backward], axis=0))

        # [TODO] diff の実装
        # river_level_df_diff = river_level_df.diff()  # 多分結束部分がうまくdiffできてない

        self.river_data = self.river_scaler.transform(river)
        self.rain_forward_data = self.rain_ma_scaler.transform(rain_forward)
        self.rain_backward_data = self.rain_ma_scaler.transform(rain_backward)

        self.dataset = self.generate_catsi_dataset()

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
    ) -> dict[str, Any]:
        """Generate sequencial data for CATSI"""
        sequences: dict[str, Any] = {"train": [], "val": [], "test": []}
        for stage in sequences.keys():
            logger.info(f"Generating {stage} dataset")
            river_flood_data = slice_data_for_flood(
                pd.DataFrame(self.river_data, index=self.timestamps, columns=self.var_names),
                self.flood_duration[stage],
                self.var_names,
            )
            rain_flood_forward_data = slice_data_for_flood(
                pd.DataFrame(self.rain_forward_data, index=self.timestamps, columns=self.var_names),
                self.flood_duration[stage],
                self.var_names,
            )
            rain_flood_backward_data = slice_data_for_flood(
                pd.DataFrame(
                    self.rain_backward_data, index=self.timestamps, columns=self.var_names
                ),
                self.flood_duration[stage],
                self.var_names,
            )
            assert len(river_flood_data) == len(rain_flood_forward_data)
            assert len(rain_flood_forward_data) == len(rain_flood_backward_data)
            for sample_info in self.samples[stage]:
                logger.info(f"-------------- {sample_info}")
                missing_rate, n_repeat = sample_info["missing_rate"], sample_info["n_repeat"]
                seeds = np.arange(n_repeat) + self.random_state
                idx = 0
                all_pts = []
                for seed in seeds:
                    np.random.seed(seed)
                    for river_item, rain_forward_item, rain_backward_item in zip(
                        river_flood_data, rain_flood_forward_data, rain_flood_backward_data
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

                        for i in range(1, len(river_item) - self.window_size + 1):
                            seq = river_item[i : i + self.window_size]
                            if len(seq) < self.window_size:
                                continue
                            time_series_data = {
                                "pt_with_na": river_item_interp[i : i + self.window_size].values,
                                "pt_with_na_org": river_item_with_nan[
                                    i : i + self.window_size
                                ].values,
                                "pt_ground_truth": seq.values,
                                "rain_accumulation_forward": rain_forward_item[
                                    i : i + self.window_size
                                ].values,
                                "rain_accumulation_backward": rain_backward_item[
                                    i : i + self.window_size
                                ].values,
                                "timestamps_dt": seq.index.to_list(),
                                "observed_mask": observed_mask[i : i + self.window_size],
                                "eval_mask": synth_nan_mask[i : i + self.window_size],
                                "length": self.window_size,
                                "pid": idx,
                                "vars": self.var_names,
                                "time_stamps": np.arange(len(seq)),
                            }
                            all_pts.append(time_series_data)
                            idx += 1
            sequences[stage] = all_pts
            print(f"{stage} dataset size: {len(all_pts)}")
        return sequences


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


if __name__ == "__main__":
    catsi_dataset = CatsiDataset(
        rain_file_name=rain_file_name,
        river_file_name=river_file_name,
        flood_duration_file_name=flood_duration_file_name,
        diff=diff,
        var_names=var_names,
        rain_smoothing=rain_smoothing,
        window_size=window_size,
        random_state=random_state,
        samples=samples,
        dataset_dir=dataset_dir,
        result_dir=result_dir,
    )

    # Save to pickle file
    with open(catsi_dataset.saving_dir / "catsi_dataset.pkl", "wb") as f:
        pickle.dump(catsi_dataset, f)

    # start context aware imputation ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = catsi_dataset.saving_dir
    output = catsi_dataset.saving_dir
    reload = False
    # for debugging - - -
    out_path = output
    force_reload_raw = reload
    train_data_path = input
    epochs: int = 5
    batch_size: int = 64
    eval_batch_size: int = 64
    eval_epoch: int = 1
    record_imp_epoch: int = 50
    early_stop: bool = False
    hidden_size: int = 64
    context_hidden: int = 32
    num_vars = 6
    learning_rate = 1e-3
    # MAE (Mean Absolute Error)
    # MRE (Mean Relative Error)
    # nRMSD (normalized Root Mean Square Deviation)

    model = ContAwareTimeSeriesImp(
        var_names=catsi_dataset.var_names,
        train_data=catsi_dataset.dataset["train"],
        val_data=catsi_dataset.dataset["val"],
        window_size=catsi_dataset.window_size,
        out_path=output,
        device=device,
    )
    model.fit(
        epochs=1000,
        batch_size=32,
        eval_batch_size=1000,
        learning_rate=1e-2,
        early_stop=True,
        shuffle=True,
    )

    # model.fit(
    #     epochs=1000,
    #     batch_size=32,
    #     eval_batch_size=32,
    #     learning_rate=1e-2,
    #     early_stop=True,
    # )
    # model.fit(epochs=1000, batch_size=1, eval_batch_size=1)
    # model.fit(epochs=10, batch_size=32, eval_batch_size=32)

    # Save the trained model
    model_save_path = Path(output) / "trained_model.pth"
    torch.save(model.model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # # Load the original data
    # test_set = catsi_dataset["train"]
    # test_set = catsi_dataset["val"]
    test_set = catsi_dataset.dataset["test"]

    imputation_results = model.impute_test_set(test_set, batch_size=1, ground_truth=True)
    print(imputation_results)

    # create imputation result
    # ground_truth = test_river_level[0]
    for col_name in var_names:
        col_id = var_names.index(col_name)
        compare_all = pd.DataFrame()
        for pid, imputation_result in enumerate(imputation_results):
            print(f"pid: {pid}")
            # print(imputation_result)
            timestamp_dt = test_set[pid]["timestamps_dt"]
            observed_mask = test_set[pid]["observed_mask"][:, col_id]
            original_river_level = scaler.inverse_transform(test_set[pid]["pt_with_na_org"])[
                :, col_id
            ]
            eval_mask = test_set[pid]["eval_mask"][:, col_id]
            ground_truth = scaler.inverse_transform(test_set[pid]["pt_ground_truth"])[:, col_id]
            compare = pd.DataFrame(
                {
                    "timestamps": timestamp_dt,
                    "ground_truth": ground_truth,
                    "ground_truth_with_synthetic_nan": original_river_level,
                    "imputation": np.nan,
                },
                index=timestamp_dt,
            )
            col_imputation_result = imputation_result[imputation_result["analyte"] == col_name]
            for _, row in col_imputation_result.iterrows():
                tid = row["tid"]
                assert col_name == row["analyte"]
                imputed_value = row["imputation"]
                compare.loc[compare.index == timestamp_dt[tid], "imputation"] = imputed_value
            imputation = np.zeros_like(test_set[pid]["pt_ground_truth"])
            imputation[:, col_id] = compare["imputation"].values
            compare["imputation"] = scaler.inverse_transform(imputation)[:, col_id]
            compare["imputation"] = np.where(
                compare["imputation"].isna(), compare["ground_truth"], compare["imputation"]
            )

            # compare_all["initial_value"] = river_level_df.loc[initial_timestamps, col_name].values
            # compare_all["ground_truth"] = compare_all["initial_value"] + compare_all["ground_truth"]
            if len(compare_all) == 0:
                compare_all = compare
            else:
                compare = compare[~compare["timestamps"].isin(compare_all["timestamps"])]
                compare_all = pd.concat([compare_all, compare], axis=0)

        initial_timestamps = compare_all["timestamps"].values - pd.Timedelta(minutes=30)
        converted_imputation = []
        for k in range(len(compare_all)):
            imputed = compare_all["imputation"].iloc[k]
            # print(f"imputed: {imputed}")
            initial = river_level_df.loc[initial_timestamps, col_name].iloc[k]
            # print(f"initial: {initial}")
            if pd.isna(initial):
                print("initial is NaN")
                converted = converted + imputed
            else:
                converted = initial + imputed
            converted_imputation.append(converted)

        compare_all["imputation"] = converted_imputation

        converted_ground_truth = []
        for k in range(len(compare_all)):
            gt = compare_all["ground_truth"].iloc[k]
            initial = river_level_df.loc[initial_timestamps, col_name].iloc[k]
            # print(f"initial: {initial}")
            if pd.isna(initial):
                print("initial is NaN")
                converted = converted + gt
            else:
                converted = initial + gt
            converted_ground_truth.append(converted)
        compare_all["ground_truth"] = converted_ground_truth

        compare_all["ground_truth_with_synthetic_nan"] = np.where(
            compare_all["ground_truth_with_synthetic_nan"].isna(),
            np.nan,
            compare_all["ground_truth"],
        )

        plt.figure(figsize=(15, 6))
        plt.plot(
            compare_all["imputation"],
            label="Imputation",
            color="red",
            # linestyle="--",
            # marker="x",
        )
        plt.plot(
            compare_all["ground_truth"],
            label="Ground Truth",
            color="blue",
            # linestyle="--",
            # marker="o",
        )
        plt.plot(
            compare_all["ground_truth_with_synthetic_nan"],
            label="Ground Truth with Synthetic NaN",
            color="green",
            # linestyle="--",
            marker="x",
        )
        plt.title("Imputation vs Ground Truth")
        plt.xlabel("Time Stamps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        plt.savefig(f"{col_name}_imputation_vs_ground_truth.png")
        plt.close()
        print(f"Saved plot for {col_name} imputation vs ground truth.")

    # checking peak performance
    modified_test_river_level = test_river_level[0].copy()

    # centre = modified_test_river_level.mean(axis=1).argmax()
    # width = 2
    # start = max(0, centre - width)
    # end = min(modified_test_river_level.shape[0], centre + width)

    # modified_test_river_level.iloc[start:end] = np.nan
    test_river_level2 = [modified_test_river_level]

    catsi_dataset2: dict[str, Any] = {"train": [], "val": [], "test": []}

    for key, river_data in zip(["test"], [test_river_level2]):
        seeds_to_run = [seeds[0]] if key == "test" else seeds
        idx = 0
        all_pts = []
        for seed in seeds_to_run:
            np.random.seed(seed)
            for item in river_data:
                if len(item) < window_size:
                    continue

                dat = item[cols_to_include]
                peak_nan_mask = np.zeros_like(dat, dtype=int)
                for col in cols_to_include:
                    dat_col = dat[col]
                    peak_pos = dat_col.argmax()
                    width = 2
                    start = max(0, peak_pos - width) + 5
                    end = min(dat.shape[0], peak_pos + width) + 5
                    peak_nan_mask[start:end, cols_to_include.index(col)] = 1

                original_nan_mask = dat.isna().astype(int).values

                synthetic_nan_mask = (
                    np.random.rand(len(dat), len(cols_to_include)) < missing_ratio
                ).astype(int)
                synthetic_nan_mask = np.maximum(synthetic_nan_mask.copy(), peak_nan_mask).astype(
                    int
                )
                synthetic_nan_mask = synthetic_nan_mask.copy() * (1 - original_nan_mask)

                dat_with_nan = dat.copy()
                dat_with_nan[synthetic_nan_mask == 1] = np.nan
                dat_interp = pd.DataFrame(dat).interpolate(method="linear", limit_direction="both")

                rain_forward_dat = rain_forward_acc_df.loc[dat.index]
                rain_backward_dat = rain_backward_acc_df.loc[dat.index]

                for i in range(1, len(dat) - window_size + 1):
                    seq = dat[i : i + window_size]
                    seq_interp = dat_interp[i : i + window_size]
                    seq_with_synthetic_nan = dat_with_nan[i : i + window_size]
                    scaled_seq = scaler.transform(seq)
                    scaled_seq_interp = scaler.transform(seq_interp)
                    scaled_seq_with_synthetic_nan = scaler.transform(seq_with_synthetic_nan)

                    rain_forward_seq = rain_forward_dat[i : i + window_size]
                    rain_backward_seq = rain_backward_dat[i : i + window_size]
                    scaled_rain_forward_seq = rain_scaler.transform(rain_forward_seq)
                    scaled_rain_backward_seq = rain_scaler.transform(rain_backward_seq)

                    if len(seq) < window_size:
                        continue

                    timestamps = seq.index.to_list()

                    seq_original_nan_mask = original_nan_mask[i : i + window_size]
                    seq_synthetic_nan_mask = synthetic_nan_mask[i : i + window_size]

                    # if key == "test":
                    #     multiplier = 2
                    # else:
                    #     multiplier = 1

                    time_series_data = {
                        "pt_with_na": scaled_seq_interp.copy(),
                        "pt_with_na_org": scaled_seq_with_synthetic_nan.copy(),
                        "pt_ground_truth": scaled_seq.copy(),
                        "rain_accumulation_forward": scaled_rain_forward_seq.copy(),
                        "rain_accumulation_backward": scaled_rain_backward_seq.copy(),
                        "timestamps_dt": timestamps,
                        "observed_mask": 1 - seq_original_nan_mask.copy(),
                        # "observed_mask": 1
                        # - np.maximum(seq_original_nan_mask.copy(), 1 - seq_synthetic_nan_mask.copy()),
                        "eval_mask": seq_synthetic_nan_mask.copy(),
                        "length": window_size,
                        "pid": idx,
                        "vars": cols_to_include,
                        "time_stamps": np.arange(len(seq)),
                        "river_scaler": scaler,
                        "rain_scaler": rain_scaler,
                    }
                    all_pts.append(time_series_data)
                    idx += 1
        catsi_dataset2[key] = all_pts
        print(f"{key} dataset size: {len(all_pts)}")

    test_set = catsi_dataset2["test"]

    imputation_results = model.impute_test_set(test_set, batch_size=1, ground_truth=True)
    print(imputation_results)

    # create imputation result
    # ground_truth = test_river_level[0]
    for col_name in var_names:
        col_id = var_names.index(col_name)
        compare_all = pd.DataFrame()
        for pid, imputation_result in enumerate(imputation_results):
            print(f"pid: {pid}")
            # print(imputation_result)
            timestamp_dt = test_set[pid]["timestamps_dt"]
            observed_mask = test_set[pid]["observed_mask"][:, col_id]
            original_river_level = scaler.inverse_transform(test_set[pid]["pt_with_na_org"])[
                :, col_id
            ]
            eval_mask = test_set[pid]["eval_mask"][:, col_id]
            ground_truth = scaler.inverse_transform(test_set[pid]["pt_ground_truth"])[:, col_id]
            compare = pd.DataFrame(
                {
                    "timestamps": timestamp_dt,
                    "ground_truth": ground_truth,
                    "ground_truth_with_synthetic_nan": original_river_level,
                    "imputation": np.nan,
                },
                index=timestamp_dt,
            )
            col_imputation_result = imputation_result[imputation_result["analyte"] == col_name]
            for _, row in col_imputation_result.iterrows():
                tid = row["tid"]
                assert col_name == row["analyte"]
                imputed_value = row["imputation"]
                compare.loc[compare.index == timestamp_dt[tid], "imputation"] = imputed_value
            imputation = np.zeros_like(test_set[pid]["pt_ground_truth"])
            imputation[:, col_id] = compare["imputation"].values
            compare["imputation"] = scaler.inverse_transform(imputation)[:, col_id]
            compare["imputation"] = np.where(
                compare["imputation"].isna(), compare["ground_truth"], compare["imputation"]
            )

            # compare_all["initial_value"] = river_level_df.loc[initial_timestamps, col_name].values
            # compare_all["ground_truth"] = compare_all["initial_value"] + compare_all["ground_truth"]
            if len(compare_all) == 0:
                compare_all = compare
            else:
                compare = compare[~compare["timestamps"].isin(compare_all["timestamps"])]
                compare_all = pd.concat([compare_all, compare], axis=0)

        initial_timestamps = compare_all["timestamps"].values - pd.Timedelta(minutes=30)
        converted_imputation = []
        for k in range(len(compare_all)):
            imputed = compare_all["imputation"].iloc[k]
            # print(f"imputed: {imputed}")
            initial = river_level_df.loc[initial_timestamps, col_name].iloc[k]
            # print(f"initial: {initial}")
            if pd.isna(initial):
                print("initial is NaN")
                converted = converted + imputed
            else:
                converted = initial + imputed
            converted_imputation.append(converted)

        compare_all["imputation"] = converted_imputation

        converted_ground_truth = []
        for k in range(len(compare_all)):
            gt = compare_all["ground_truth"].iloc[k]
            initial = river_level_df.loc[initial_timestamps, col_name].iloc[k]
            # print(f"initial: {initial}")
            if pd.isna(initial):
                print("initial is NaN")
                converted = converted + gt
            else:
                converted = initial + gt
            converted_ground_truth.append(converted)
        compare_all["ground_truth"] = converted_ground_truth

        compare_all["ground_truth_with_synthetic_nan"] = np.where(
            compare_all["ground_truth_with_synthetic_nan"].isna(),
            np.nan,
            compare_all["ground_truth"],
        )

        plt.figure(figsize=(15, 6))
        plt.plot(
            compare_all["imputation"],
            label="Imputation",
            color="red",
            # linestyle="--",
            # marker="x",
        )
        plt.plot(
            compare_all["ground_truth"],
            label="Ground Truth",
            color="blue",
            # linestyle="--",
            # marker="o",
        )
        plt.plot(
            compare_all["ground_truth_with_synthetic_nan"],
            label="Ground Truth with Synthetic NaN",
            color="green",
            # linestyle="--",
            marker="x",
        )
        plt.title("Imputation vs Ground Truth")
        plt.xlabel("Time Stamps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        plt.savefig(f"{col_name}_imputation_vs_ground_truth.png")
        plt.close()
        print(f"Saved plot for {col_name} imputation vs ground truth.")
