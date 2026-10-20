> **Evaluation correction (October 2026):** The previously reported 100% metrics are not trustworthy. The `LIARDataset` label mapping compares the dataset's numeric IDs to string values, potentially collapsing all samples to class 0. The reported single-class confusion matrix is consistent with this problem. **Do not use the published figures as evidence of predictive performance until the label mapping and holdout evaluation are corrected.** No corrected performance numbers have yet been established.

---


**Credits**: This project was created by Mohamed Amine EL MOUSSAOUI.

# LIAR Dataset Classification with BERT and RoBERTa

This project provides a complete pipeline for training and evaluating BERT and RoBERTa models on the LIAR dataset. The project is organized to facilitate data handling, model training, evaluation, logging, and model saving.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Description](#model-description)
- [Logging](#logging)
- [Results](#results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

## Introduction

The LIAR dataset is a large-scale dataset for fake news detection. This project demonstrates how to preprocess the dataset, train BERT and RoBERTa models, evaluate their performance, and save the models and logs.

## Installation

To run this project, you need to have Python installed along with some dependencies. You can install the required dependencies using pip:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

### Training and Evaluating Models

1. Clone the repository:
    \`\`\`bash
    git clone https://github.com/yourusername/project.git
    cd project
    \`\`\`

2. Run the training script:
    \`\`\`bash
    python scripts/run_training.py
    \`\`\`

3. Alternatively, you can use the Jupyter notebook \`run_training.ipynb\` to run the training and evaluation pipeline on Google Colab.

### Configuration

You can configure the training parameters and other settings in the \`config.py\` file.

## Data Description

The LIAR dataset contains labeled statements with various features such as the subject, speaker, job title, state information, party affiliation, context, and the statement itself. The labels indicate whether the statement is true, mostly true, or false. The dataset is split into training, validation, and test sets.

- **train**: 10,269 examples
- **validation**: 1,284 examples
- **test**: 1,283 examples

## Model Description

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed to understand the context of a word in search queries. In this project, we fine-tune the BERT model for sequence classification.

### RoBERTa

RoBERTa (Robustly optimized BERT approach) is an optimized method for pretraining a self-supervised NLP system. We fine-tune the RoBERTa model for sequence classification in this project.

## Logging

Training logs and evaluation results are saved in the specified log directory. The logs include information such as training loss, validation accuracy, and the time taken for each epoch.

Example log snippet:
\`\`\`
======== Epoch 1 / 2 ========
Training...
  Batch 40  of  642.    Elapsed: 0:00:27.
  Batch 80  of  642.    Elapsed: 0:00:53.
  ...
  Average training loss: 0.01
  Training epoch took: 0:07:21
Running Validation...
  Accuracy: 1.00
  Evaluation took: 0:00:20
\`\`\`

## Results — not yet validated

The previously published test output showed a single predicted/reference class with perfect scores. **Those metrics are invalid as model-quality evidence** and have been withdrawn from this README. See the correction notice at the top.

The implementation still needs an explicit mapping of the LIAR dataset's integer labels, checks of class distributions at every split, and a complete reevaluation against majority-class and other baselines. No corrected BERT or RoBERTa accuracy, macro-F1, or per-class metrics are currently asserted.

## Project Structure

\`\`\`

project/
├── data/
│   └── liar_dataset.py
├── models/
│   └── model_utils.py
├── notebooks/
│   └── run_training.ipynb
├── config/
│   └── config.py
├── utils/
│   └── helpers.py
├── logs/                 # Directory to store log files
│   └── (empty for now)
├── saved_models/         # Directory to store saved models
│   └── (empty for now)
├── README.md
└── requirements.txt

\`\`\`

## Configuration

Edit the \`config/config.py\` file to set the training parameters, model types, and other settings.

### Example Configuration (\`config/config.py\`)

\`\`\`python

config = {
    "epochs": 2,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "eps": 1e-8,
    "log_dir": "/content/drive/My Drive/logs",
    "model_save_dir": "/content/drive/My Drive/saved_models"
}

\`\`\`

## Acknowledgements

This project uses the following libraries and datasets:
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [LIAR Dataset](https://github.com/Tariq60/LIAR)

Special thanks to the developers and contributors of these libraries.
