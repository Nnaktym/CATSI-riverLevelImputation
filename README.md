# CATSI-based River Water Level Time Series Imputation

This repository provides a framework for training and evaluating a model for river water level time series imputation using the CATSI (Context-Aware Time Series Imputation) approach. The repository includes data preprocessing, model training, evaluation, and visualization scripts.

## Usage

To use this repository for river water level time series imputation, follow these steps:

1. **Install Dependencies**  
  Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

2. **Prepare Data**  
  Place your input data files (e.g., river level, rainfall intensity) in the `data/` directory. Ensure the file formats match the expected structure.

3. **Run the Main Script**  
  Execute the main script to train and evaluate the model:

    ```bash
    python main.py --config config.ini
    ```

4. **Visualize Results**  
  Use the visualization tools provided in the repository to analyze the imputed time series.

5. **Customize Configuration**  
  Modify `config.ini` to adjust model parameters, data paths, and other settings as needed.
