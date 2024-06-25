# Wearable Abnormal Emotion Detection
This repository contains the code for the paper "Seamless Monitoring of Stress Levels Leveraging a Universal Model for Time Sequences".

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