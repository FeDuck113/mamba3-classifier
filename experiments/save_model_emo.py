import os
import random
import json
import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import save_file
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    precision_recall_fscore_support, average_precision_score, hamming_loss,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)

from mambaClf import MambaClassifier, CLF_HEADS 
from load_datasets import load_go_emotions
from utils.metrics import compute_metrics_multi

logging.set_verbosity_error()
logging.disable_progress_bar()

def _set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
SEED = 42    
_set_seed(SEED)


def save_run_results_and_model(model, tokenizer, params, val_ds, test_ds, label_names, reproduced_val_f1):
    model_id = params["id"]
    results_dir = os.path.join("experiments", "results", "go_emotions")
    model_dir = os.path.join("models", "go_emotions", str(model_id))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=os.path.join("experiments", "temp", f"eval_go_emotions_{model_id}"),
        per_device_eval_batch_size=params["stage2_batch_size"],
        bf16=True,
        report_to="none",
        disable_tqdm=True,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_multi,
    )

    summary_rows = []
    test_class_rows = []
    for split, dataset in (("validation", val_ds), ("test", test_ds)):
        result = trainer.predict(dataset, metric_key_prefix=split)
        logits = np.asarray(result.predictions, dtype=np.float32)
        labels = np.asarray(result.label_ids, dtype=np.int32)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        preds = (probs > 0.5).astype(np.int32)
        summary = {
            "model_id": model_id,
            "split": split,
            "n_examples": len(labels),
            "accuracy_subset": accuracy_score(labels, preds),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "hamming_loss": hamming_loss(labels, preds),
            "average_precision_macro": average_precision_score(labels, probs, average="macro"),
            "threshold": 0.5,
        }
        summary_rows.append(summary)

        if split == "test":
            precision, recall, f1, support = precision_recall_fscore_support(
                labels, preds, average=None, zero_division=0
            )
            for i, name in enumerate(label_names):
                positive_count = int(labels[:, i].sum())
                test_class_rows.append({
                    "label_id": i,
                    "label": name,
                    "support": positive_count,
                    "negative_support": len(labels) - positive_count,
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i],
                    "average_precision": (
                        average_precision_score(labels[:, i], probs[:, i])
                        if positive_count else np.nan
                    ),
                    "predicted_positive": int(preds[:, i].sum()),
                    "threshold": 0.5,
                })

    for row in summary_rows:
        row["seed"] = SEED
        row["tokenizer"] = params["tokenizer"]
        row["base_model"] = params["model"]
        row["max_length"] = int(params["max_length"])
        row["pooling"] = params["pooling"]
        row["classifier_head"] = params["clf_head"]
        row["reproduced_best_validation_f1_macro"] = float(reproduced_val_f1)
        row["selected_run_validation_f1_macro"] = float(params["f1"])
    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.DataFrame(test_class_rows)
    summary_path = os.path.join(results_dir, f"summary_{model_id}.csv")
    class_path = os.path.join(results_dir, f"test_per_class_{model_id}.csv")
    summary_df.to_csv(summary_path, index=False)
    class_df.to_csv(class_path, index=False)
    run_info = {
        "model_id": model_id,
        "seed": SEED,
        "tokenizer": params["tokenizer"],
        "base_model": params["model"],
        "max_length": int(params["max_length"]),
        "selected_run_validation_f1_macro": float(params["f1"]),
        "reproduced_validation_f1_macro": float(reproduced_val_f1),
        "training_params": params,
        "threshold": 0.5,
        "summary_file": os.path.basename(summary_path),
        "test_per_class_file": os.path.basename(class_path),
    }

    state_dict = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state_dict, os.path.join(model_dir, "model.safetensors"))
    tokenizer.save_pretrained(model_dir)
    config = {
        "model_type": "mamba3_text_classifier",
        "architectures": ["MambaClassifier"],
        "base_model_name_or_path": (
            params["model"] if "/" in params["model"]
            else f"state-spaces/{params['model']}"
        ),
        "tokenizer_name": (
            params["tokenizer"] if "/" in params["tokenizer"]
            else ("meta-llama/Llama-3.1-8B" if params["tokenizer"] == "Llama-3.1-8B" else params["tokenizer"])
        ),
        "num_labels": len(label_names),
        "problem_type": "multi_label_classification",
        "id2label": {str(i): name for i, name in enumerate(label_names)},
        "label2id": {name: i for i, name in enumerate(label_names)},
        "max_length": int(params["max_length"]),
        "pooling": params["pooling"],
        "classifier_head": params["clf_head"],
        "dropout": float(params["dropout"]),
        "threshold": 0.5,
        "seed": SEED,
        "training_params": params,
        "reproduced_validation_f1_macro": float(reproduced_val_f1),
        "evaluation": summary_rows,
    }
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(model_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)


def run_stage(
    model: MambaClassifier,
    tokenizer, 
    output_dir: str,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    per_device_batch_size: int,
    warmup_steps: int,
    lr_scheduler_type: str,
    max_grad_norm: float,
    stage_name: str,
    seed = 42,
) -> float:
    global train_ds, val_ds
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,        
        bf16=True,
        max_grad_norm=max_grad_norm,        
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,        
        optim="adamw_torch_fused",
        
        eval_strategy="epoch",
        logging_steps=500,
        report_to="none",
        disable_tqdm=True,
        
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,        
        
        seed=seed,
        data_seed=seed,
    )
        
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_multi
    )
    trainer.train()
    best_f1 = trainer.state.best_metric
    return best_f1


def train_model(params):
    _set_seed(SEED)
    TOKENIZER_MODEL = params['tokenizer']
    MAX_LENGTH = params['max_length']
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B" if TOKENIZER_MODEL=="Llama-3.1-8B" else "aboba")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    global train_ds, val_ds
    train_ds, val_ds, test_ds, base_pos_weight_ratio, label_names = load_go_emotions(
        tokenize_batch, 28, return_label_names=True
    )
    
    BASE_MODEL = "state-spaces/mamba3-siso-187m" if params['model'] == "mamba3-siso-187m" else "aboba"
    mamba_lm = MambaLMHeadModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)
    
    pos_weight_clamp = params['pos_weight_clamp']
    pos_weight = torch.tensor(base_pos_weight_ratio, dtype=torch.float32)
    if not pos_weight_clamp is None:
        pos_weight = pos_weight.clamp(max=pos_weight_clamp)
        
    clf_model = MambaClassifier(
        mamba_lm=mamba_lm,
        num_labels=28,
        problem_type="multi_label_classification",
        classifier_head=CLF_HEADS[params['clf_head']],
        pooling=params['pooling'],
        dropout=params['dropout'],
        pos_weight=pos_weight
    )
    
    trial_dir = os.path.join('experiments', 'temp', 'save_mamba')
    if params['freeze_backbone']:
        stage1_warmup_ratio =params["stage1_warmup_ratio"]
        stage1_steps_per_epoch = len(train_ds) // params["stage1_batch_size"]
        stage1_total_steps = stage1_steps_per_epoch * 10
        stage1_warmup_steps = int(stage1_total_steps * stage1_warmup_ratio)
        
        clf_model.freeze_backbone()
        run_stage(
            clf_model, tokenizer,
            output_dir=os.path.join(trial_dir, "stage1"),
            num_train_epochs=10,
            learning_rate=params['stage1_lr'],
            weight_decay=params['stage1_weight_decay'],
            per_device_batch_size=params['stage1_batch_size'],
            warmup_steps=stage1_warmup_steps,
            lr_scheduler_type=params.get('stage1_lr_scheduler', "cosine"),
            max_grad_norm=params.get('stage1_max_grad_norm', 1.0),
            stage_name="stage1"
        )
        clf_model.unfreeze_backbone() 
    
    stage2_warmup_ratio = params["stage2_warmup_ratio"]
    stage2_steps_per_epoch = len(train_ds) // params["stage2_batch_size"]
    stage2_total_steps = stage2_steps_per_epoch * 20
    stage2_warmup_steps = int(stage2_total_steps * stage2_warmup_ratio)
    
    stage2_f1 = run_stage(
        clf_model, tokenizer,
        output_dir=os.path.join(trial_dir, "stage2"),
        num_train_epochs=20,
        learning_rate=params['stage2_lr'],
        weight_decay=params['stage2_weight_decay'],
        per_device_batch_size=params['stage2_batch_size'],
        warmup_steps=stage2_warmup_steps,
        lr_scheduler_type=params['stage2_lr_scheduler'],
        max_grad_norm=params['stage2_max_grad_norm'],
        stage_name="stage2"
    )
    
    expected_f1 = params["f1"]
    diff = abs(stage2_f1 - expected_f1)

    F1_TOLERANCE = 0.01
    
    if diff > F1_TOLERANCE:
        print('bad f1')
        return
    
    save_run_results_and_model(
        clf_model, tokenizer, params, val_ds, test_ds, label_names, stage2_f1
    )
    


if __name__ == "__main__":  
    IDS = [11]
    
    with open('configs/emo_config.yaml', 'r') as file:
        emo_data = yaml.safe_load(file)
        
    for run_id in IDS:
        params = next((block for block in emo_data if block.get("id") == run_id), None)

        if params is None:
            raise ValueError(f"Run {run_id} is not found")

        print(
            f"\n{'=' * 60}\n"
            f"Training run {run_id}\n"
            f"{'=' * 60}"
        )

        train_model(params)
