# Mamba3 classifier
This project introduces classification heads for Mamba 3 models, with an implementation compatible with the Hugging Face Transformers training interface.

Additionally, I trained models on the GoEmotions and BAH datasets. Their results are presented below.
- **GoEmotions** is a dataset of 58,000 Reddit comments annotated with 27 emotion categories and a neutral category ([Dataset paper](https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/))
- **BAH** is a dataset designed for recognizing ambivalence and hesitancy. It contains video responses and their corresponding text transcripts. The dataset was introduced at ICLR 2026. ([Dataset paper](https://arxiv.org/pdf/2505.19328))

The GoEmotions model is available on [Hugging Face](https://huggingface.co/FeDuck113/mamba3-go_emotions)

For the BAH experiments, I trained models using two approaches: fine-tuning directly from the original Mamba 3 weights, and fine-tuning from a Mamba 3 model previously trained on GoEmotions. The latter approach was motivated by the hypothesis that emotional features can provide useful representations for detecting confidence and uncertainty.<br>
I previously explored this hypothesis in our research paper [**TOMAR**](https://openreview.net/pdf?id=mdHET3onT2), developed for the [ABAW Workshop](https://affective-behavior-analysis-in-the-wild.github.io/11th/). The paper focused on developing a multimodal model for confidence and uncertainty detection. Within this work, I was also responsible for the text modality, but used Transformer-based architectures. Our team achieved **2nd place at the 11th ABAW Workshop**, providing further evidence of the effectiveness of this approach. A comparison of the results obtained in TOMAR with those from the Mamba 3 experiments is also presented below.

## Table of Contents

- [Project Structure](#project-structure)
- [Results](#results)
  - [GoEmotions test set](#goemotions-test-set)
  - [BAH test set](#bah-test-set)
  - [GoEmotions](#goemotions)
  - [BAH](#bah)
- [Quickstart](#quickstart)
- [Training pipeline](#training-pipeline)
- [Citation](#citation)
- [License](#license)

## Project Structure
```text
|-- datasets/
|-- src/
|   |-- mambaClf.py             # Mamba3 classifier and custom classification heads
|   |-- model_loader.py         # Load pretrained models from hf and local
|   |-- prepare_bah.py          # Preprocess raw BAH data 
|   '-- utils/
|       |-- load_datasets.py    # Datasets loading utils
|       '-- metrics.py          
|-- configs/
|   |-- emo_config.yaml         # Selected configs for training on GoEmotions
|   '-- bah_config.yaml         # --//-- on BAH
|-- results/                    # Evaluation results
|-- models/
|   |-- bah/                    # Here you can store model files in <id>/ folders
|   '-- go_emotions/            # for local loading (e.g. you can download my model from hf)
'-- experiments/
    |-- mamba_optuna.py         # Hyperparameter optimization with Optuna   
    |-- save_model_bah.py
    '-- save_model_emo.py
```

## Results
Tables with all evaluation results are available in the [results/](results/) directory.

#### GoEmotions test set

| Model | Precision | Recall | F1 | Source |
|---|---:|---:|---:|---|
| Mamba3 GoEmotions (this work) | 37.67 | 66.49 | 47.00 | This repository |
| RoBERTa GoEmotions | 57.52 | 39.62 | 45.03 | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
| UltraBERT GoEmotions | 65.67 | 42.98 | 49.67 | [cirimus/ultrabert-380M-go-emotions](https://huggingface.co/cirimus/ultrabert-380M-go-emotions) |

#### BAH test set

| Model | Modality | Macro F1 | Source |
|---|---|---:|---|
| Mamba3 classifier (31 run) | Text | 67.86 | This repository |
| Mamba3 classifier with emo features (71 run) | Text | 66.66 | This repository |
| TOMAR text model | Text | 72.10 | [Paper](https://openreview.net/pdf?id=mdHET3onT2) (11th ABAW 2nd place)|
| TOMAR multimodal model | Text, Audio, Face, Scene | 74.14 | [Paper](https://openreview.net/pdf?id=mdHET3onT2) (11th ABAW 2nd place)|
| AffectVision's model | Text, Audio, Face, Scene | 73.1 | [Paper](https://arxiv.org/pdf/2607.11120) (11th ABAW winner) |
| BAH baseline | Text, Audio, Visual  | 59.34 | [BAH paper](https://arxiv.org/pdf/2505.19328) |

31 run: Fine-tuned directly from the original `mamba3-siso-187m` weights.<br>
71 run: Fine-tuned from a Mamba3 previously fine-tuned on the GoEmotions dataset, but have higher ROC-AUC. More details below.

The winner at the 11th ABAW workshop was determined based on private test sets. As a result, the AffectVision team took first place (F1 - 0.7959), and RAS with the TOMAR model took second (F1 - 0.7824). You can view the workshop's results [here](https://drive.google.com/file/d/1nLprNq1Sxl8ZT8G462hxKYrQa1Eomy27/view)

### GoEmotions

Test-set results for the GoEmotions model (seed 42, threshold 0.5):

| Split | Examples | Subset accuracy | Macro precision | Macro recall | Macro F1 | Micro F1 | Macro average precision | Hamming loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 5,426 | 0.2326 | 0.3935 | 0.6752 | 0.4857 | 0.5297 | 0.5008 | 0.0567 |
| Test | 5,427 | 0.2091 | 0.3767 | 0.6649 | 0.4700 | 0.5225 | 0.4849 | 0.0579 |

Per-class test results:

| Emotion | Support | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|---:|
| admiration | 504 | 0.4619 | 0.8671 | 0.6028 | 0.7260 |
| amusement | 264 | 0.6295 | 0.9205 | 0.7477 | 0.8060 |
| anger | 198 | 0.2762 | 0.6515 | 0.3880 | 0.3955 |
| annoyance | 320 | 0.2032 | 0.6750 | 0.3124 | 0.3202 |
| approval | 351 | 0.2479 | 0.6695 | 0.3618 | 0.4344 |
| caring | 135 | 0.2877 | 0.6222 | 0.3934 | 0.3634 |
| confusion | 153 | 0.2359 | 0.7386 | 0.3576 | 0.4011 |
| curiosity | 284 | 0.4091 | 0.8873 | 0.5600 | 0.5123 |
| desire | 83 | 0.4128 | 0.5422 | 0.4688 | 0.4513 |
| disappointment | 151 | 0.1555 | 0.4768 | 0.2345 | 0.2523 |
| disapproval | 267 | 0.2600 | 0.6816 | 0.3764 | 0.3597 |
| disgust | 123 | 0.3255 | 0.6748 | 0.4392 | 0.4985 |
| embarrassment | 37 | 0.4500 | 0.4865 | 0.4675 | 0.4045 |
| excitement | 103 | 0.2651 | 0.5534 | 0.3585 | 0.4209 |
| fear | 78 | 0.4745 | 0.8333 | 0.6047 | 0.7170 |
| gratitude | 352 | 0.7557 | 0.9403 | 0.8380 | 0.9553 |
| grief | 6 | 0.5000 | 0.5000 | 0.5000 | 0.3611 |
| joy | 161 | 0.3296 | 0.7329 | 0.4547 | 0.5408 |
| love | 238 | 0.6324 | 0.9034 | 0.7439 | 0.8430 |
| nervousness | 23 | 0.1739 | 0.3478 | 0.2319 | 0.2009 |
| optimism | 186 | 0.3664 | 0.6559 | 0.4701 | 0.5719 |
| pride | 16 | 0.4286 | 0.3750 | 0.4000 | 0.2552 |
| realization | 145 | 0.2090 | 0.3517 | 0.2622 | 0.2213 |
| relief | 11 | 0.2000 | 0.3636 | 0.2581 | 0.1376 |
| remorse | 56 | 0.5426 | 0.9107 | 0.6800 | 0.6147 |
| sadness | 156 | 0.3567 | 0.7500 | 0.4835 | 0.5773 |
| surprise | 141 | 0.3731 | 0.6879 | 0.4838 | 0.5332 |
| neutral | 1,787 | 0.5840 | 0.8170 | 0.6811 | 0.7027 |

### BAH

Validation and test results for runs 31 and 71 (seed 42, threshold 0.5):

| Model | Split | Examples | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | Validation | 124 | **0.6694** | **0.6754** | **0.6729** | **0.6754** | **0.6688** | 0.6664 | 0.7467 |
| 71 | Validation | 124 | 0.6452 | 0.6540 | 0.6528 | 0.6540 | 0.6451 | **0.6937** | **0.7754** |

| Model | Split | Examples | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | Test | 525 | **0.6800** | 0.6855 | 0.6813 | 0.6855 | **0.6786** | 0.7385 | **0.7910** |
| 71 | Test | 525 | 0.6667 | 0.6804 | 0.6792 | 0.6804 | 0.6666 | **0.7525** | 0.7828 |

On the test set, run 31 achieves higher macro F1 and accuracy, while run 71 has higher ROC-AUC.

Per-class test results:

| Model | Label | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|---:|
| 31 | 0 | **0.6030** | 0.7220 | 0.6571 | 0.6646 |
| 71 | 0 | 0.5811 | **0.7713** | **0.6628** | **0.7048** |

| Model | Label | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|---:|
| 31 | 1 | 0.7597 | **0.6490** | **0.7000** | **0.7910** |
| 71 | 1 | **0.7773** | 0.5894 | 0.6704 | 0.7828 |

Test confusion matrices (rows are actual labels, columns are predicted labels):

| Model | Actual label | Predicted 0 | Predicted 1 |
|---|---:|---:|---:|
| 31 | 0 | 161 | 62 |
| 31 | 1 | 106 | **196** |
| 71 | 0 | **172** | 51 |
| 71 | 1 | 124 | 178 |

Run 61 has a slightly higher test ROC-AUC and average precision than run 71, but run 71 performs better on the classification metrics:

| Test metric | Run 61 | Run 71 |
|---|---:|---:|
| ROC-AUC | **0.7555** | 0.7525 |
| Average precision | **0.7845** | 0.7828 |
| Accuracy | 0.6571 | **0.6667** |
| Balanced accuracy | 0.6715 | **0.6804** |
| Macro precision | 0.6709 | **0.6792** |
| Macro recall | 0.6715 | **0.6804** |
| Macro F1 | 0.6571 | **0.6666** |
| Label 0 F1 | 0.6552 | **0.6628** |
| Label 1 F1 | 0.6591 | **0.6704** |

Among these runs, run 83 has the highest validation macro F1 (0.7790), but its test macro F1 (0.6490) is lower than those of runs 31, 61 and 71.


## Quickstart
### Hugging Face
The best GoEmotions model is available on [Hugging Face](https://huggingface.co/FeDuck113/mamba3-go_emotions). Clone this repository and install dependencies from `requirements.txt` (`mamba-ssm` installation may be quite hard), then run:

```python
from src.model_loader import load_model

model = load_model("FeDuck113/mamba3-go_emotions")
predictions = model.predict("I am so grateful for your help!")
print(predictions)
```

The loaded model can also be fine-tuned with Hugging Face `Trainer`, for example on GoEmotions:

```python
model = load_model("FeDuck113/mamba3-go_emotions")

training_args = TrainingArguments(...)
trainer = Trainer(
    model=model,
    processing_class=model.tokenizer,
    ... 
)
trainer.train()
```
### Local
To load a downloaded model from a local folder, use `load_model_from_dir`. The folder must contain `model.safetensors`, `config.json` and the tokenizer files:

```python
from src.model_loader import load_model_from_dir

model = load_model_from_dir("models/go_emotions/11")
predictions = model.predict("I am so grateful for your help!")
print(predictions)
```

## Training pipeline
1. Requirements

Install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

2. Data preparation

[GoEmotions dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions) is automatically downloaded from HF by `load_go_emotions` function in `src/utils/load_datasets.py`.

To request access to BAH, visit the [BAH dataset repository](https://github.com/LIVIAETS/bah-dataset).
After placing data into `datasets/BAH/` run `src/prepare_bah.py`.

3. Train and evaluate

Set the experiment parameters in the global variables at the top of `experiments/mamba_optuna.py`, then run it.<br>
The best experiment results were obtained using `state-spaces/mamba3-mimo-187m` as the BASE_MODEL and `meta-llama/Llama-3.1-8B` as the tokenizer (following the configuration used by the state-spaces implementation). The implementation is also compatible with other Mamba 3 models released by state-spaces.<br>
For detailed information about the experiment parameters, see `experiments/mamba_optuna.py`.

Optuna evaluates runs on the validation set and saves the results to the path specified by `EXPERIMENTS_FILE`.<br>You can queue a configuration from `configs/` for faster searching: uncomment `enqueue_trial_from_file(study, <id>)` and provide its run ID.

4. Save your model

Add the selected run configurations to an appropriate `configs/*_config.yaml` file. Set its ID in IDS in `experiments/save_model_*.py` and then run it.<br>
Models are saved to `models/go_emotions/<id>/` or `models/bah/<id>/` in safetensors format.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mamba3_classifier,
  title  = {Mamba3 Classifier},
  author = {Fedor Shchetinin},
  year   = {2026},
  url    = {https://github.com/FeDuck113/mamba3-classifier}
}
```

## License

The source code in this repository is licensed under the [Apache License 2.0](LICENSE).<br>
The BAH dataset is not included and is subject to its own access conditions and license. Please refer to the [BAH dataset repository](https://github.com/LIVIAETS/bah-dataset) for details.
