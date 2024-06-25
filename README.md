# Wearable Abnormal Emotion Detection
This repository contains the code for the paper "Seamless Monitoring of Stress Levels Leveraging a Universal Model for Time Sequences".


## Results

### Comparison of Different Methods

|                | DREAMER | HCI   | WESAD (ECG) | WESAD (BVP) | AVG F1          |
|----------------|---------|-------|-------------|-------------|-----------------|
| LSTM-NDT       | 0.313   | 0.375 | 0.785       | 0.772       | 0.561 ± .218    |
| OmniAnomaly    | 0.599   | 0.547 | 0.767       | 0.650       | 0.641 ± .081    |
| CAE-M          | 0.658   | 0.530 | 0.801       | 0.590       | 0.645 ± .101    |
| HypAD          | 0.650   | 0.643 | 0.815       | 0.567       | 0.669 ± .090    |
| MTAD-GAT       | 0.536   | 0.742 | 0.857       | 0.719       | 0.714 ± .115    |
| TranAD         | 0.617   | 0.623 | 0.837       | _0.804_     | 0.720 ± .101    |
| TadGAN         | 0.590   | 0.691 | _0.864_     | 0.743       | 0.722 ± .099    |
| GDN            | 0.713   | 0.580 | 0.858       | 0.802       | 0.738 ± .105    |
| DAGMM          | _0.743_ | 0.647 | 0.831       | 0.773       | 0.749 ± .067    |
| USAD           | 0.730   | 0.660 | 0.830       | 0.797       | 0.754 ± .065    |
| MAD-GAN        | 0.706   | 0.743 | 0.839       | 0.787       | 0.769 ± .050    |
| MSCRED         | 0.675   | _0.824_ | **0.876**  | 0.775       | _0.788 ± .074_  |
| **UniTS**      | **0.869** | **0.878** | 0.834 | **0.856** | **0.859 ± .019** |

### P-values from Dunn Post-hoc Test

|         | UniTS                 |
|---------|-----------------------|
| MSCRED  | $2.813 \times 10^{-2}$ |
| MAD-GAN | $4.795 \times 10^{-3}$ |
| USAD    | $6.839 \times 10^{-3}$ |
| DAGMM   | $1.136 \times 10^{-3}$ |
| GDN     | $5.146 \times 10^{-3}$ |

## Setup
The following command installs the required dependencies:
```bash
# Install the dependencies
pip install -r requirements.txt
```

## Datasets and Preprocessing
The following datasets are used in this project:
- [WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
- [DREAMER](https://zenodo.org/records/546113)
- [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/)

The datasets are preprocessed using the following steps:
```bash
# Download and extract the datasets
./data/download-datasets.sh

# Preprocess the datasets
python preprocess.py
```


## Training
Ensure to be logged and to set your WANDB_USER in the file `run.py` to log the results to [Weights & Biases](https://wandb.ai/).
The following command launches the training of the models:
```bash
# Train the models
python run.py
```


## Evaluation
The evaluation results are processed from logged runs on [Weights & Biases](https://wandb.ai/).
To retrieve the evaluation results, run the following command:
```bash
# Evaluate the models
python results.py
```

## Acknowledgements
This project use the code from the repositiories of [UniTS](https://github.com/mims-harvard/UniTS), [TranAD](https://github.com/imperial-qore/TranAD) and [HypAD](https://github.com/aleflabo/HypAD/) for the implementation of the models.