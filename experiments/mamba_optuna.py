import os
import random
import gc
import numpy as np
import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from safetensors.torch import load_file
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    logging,
)

from mambaClf import MambaClassifier, CLF_HEADS 
from utils.load_datasets import load_go_emotions, load_bah
from utils.metrics import compute_metrics_single, compute_metrics_multi

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

# ------------- RUN PARAMETERS --------------

# state-spaces/mamba3-siso-187m | state-spaces/mamba3-siso-1.5b | state-spaces/mamba3-mimo-187m
BASE_MODEL = "state-spaces/mamba3-mimo-187m"
TOKENIZER_MODEL = "meta-llama/Llama-3.1-8B"

PRETRAINED = False      # Load pretrained GoEmotions weights if True
MODEL_ID = 11           # Weight file: models/go_emotions/{MODEL_ID}/model.safetensors

BAH = True              # True if you want train model on BAH dataset
AH_METHODOLOGY = "max"  # Method for detecting AH in responses (BAH only)

MAX_LENGTH = 256        # Maximum tokenizer sequence length
FREEZE_BACKBONE = True  # Use a two-stage training strategy
FREEZE_LAYERS = False   # Enable layer freezing and optimize the number of frozen layers with Optuna

N_OPTUNA_TRIALS = 150   # Number of Optuna trials

# ---------------------------

if PRETRAINED and not MODEL_ID:
    raise ValueError("Pretrained model id is incorrect")

MODEL_NAME = BASE_MODEL.split('/')[1]
TOKENIZER_NAME = TOKENIZER_MODEL.split('/')[1]
PROBLEM_TYPE = "multi_label_classification" if not BAH else "single_label_classification"
NUM_LABELS = 28 if not BAH else 2
folder = "mamba_emo" if not BAH else "mamba_bah"
OPTUNA_OUTPUT_DIR = f"./temp/{folder}_optuna"

FILE_NAME = f"{TOKENIZER_NAME}_{MAX_LENGTH}_{MODEL_NAME}_{FREEZE_BACKBONE}"
if BAH:
    FILE_NAME += f"_{AH_METHODOLOGY}"
    if PRETRAINED:
        MODEL_NAME = f"{MODEL_ID}_" + MODEL_NAME
        FILE_NAME = f"{MODEL_ID}_" + FILE_NAME 
if FREEZE_LAYERS:
    FILE_NAME += f"_FREEZE"
EXPERIMENTS_FILE = f"experiments/sweep_results/{folder}/{FILE_NAME}.yaml"

# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

train_ds, val_ds, test_ds, base_pos_weight_ratio = (None, None, None, None)
if not BAH:
    train_ds, val_ds, test_ds, base_pos_weight_ratio = load_go_emotions(tokenize_batch, NUM_LABELS)
else:
    train_ds, val_ds, test_ds, base_pos_weight_ratio = load_bah(tokenize_batch, ah_methodology=AH_METHODOLOGY)

# ---------------------------

def write_results(new_record: dict, save_file_name: str = EXPERIMENTS_FILE):
    try:
        with open(save_file_name, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or []
    except FileNotFoundError:
        yaml_data = []
    new_id = max((int(record.get("id", 0)) for record in yaml_data), default=0) + 1
    new_record["id"] = new_id
    yaml_data.append(new_record)
    with open(save_file_name, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)           

class WriteResultCallback(TrainerCallback):
    def __init__(self, write_func, trial_number: int, stage: str,
                 trial_params: dict, extra_params: dict | None = None, save_file_name: str = EXPERIMENTS_FILE,):
        self.write_func = write_func
        self.trial_number = trial_number
        self.stage = stage
        self.trial_params = trial_params
        self.save_file_name = save_file_name
        self.extra_params = extra_params if not extra_params is None else dict()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        new_record = {
            "tokenizer": TOKENIZER_NAME,
            "model": MODEL_NAME,
            "trial": self.trial_number,
            "stage": self.stage,
            "device": str(torch.cuda.get_device_name(0)),
            "freeze_backbone": FREEZE_BACKBONE,
            **self.trial_params,
            **self.extra_params,
            "max_length": MAX_LENGTH,
            "epoch": metrics.get("epoch"),
            "eval_loss": metrics.get("eval_loss"),
            "f1": metrics.get("eval_f1_macro"),
            "accuracy": metrics.get("eval_accuracy"),
            "seed": SEED
        }
        self.write_func(new_record, self.save_file_name)


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, metric_name="eval_f1_macro"):
        self.trial = trial
        self.metric_name = metric_name
        self.best_metric = float("-inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        score = metrics.get(self.metric_name)
        if score is None:
            return
        
        self.best_metric = max(self.best_metric, score)
        self.trial.report(score, step=state.epoch)
        
        if self.trial.should_prune():
            raise optuna.TrialPruned() 
        
# ----------------------------

def build_model(trial: optuna.Trial) -> MambaClassifier:
    pooling = trial.suggest_categorical("pooling", ["mean", "last", "first", "max"])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    clf_head_name = trial.suggest_categorical("clf_head", list(CLF_HEADS.keys()))
    pos_weight_clamp = trial.suggest_categorical("pos_weight_clamp", [None, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0])
    
    pos_weight = torch.tensor(base_pos_weight_ratio, dtype=torch.float32)
    if not pos_weight_clamp is None:
        pos_weight = pos_weight.clamp(max=pos_weight_clamp)
    
    mamba_lm = MambaLMHeadModel.from_pretrained(BASE_MODEL, device="cuda", dtype=torch.bfloat16)
    model = MambaClassifier(
        mamba_lm=mamba_lm,
        num_labels=NUM_LABELS,
        problem_type=PROBLEM_TYPE,
        classifier_head=CLF_HEADS[clf_head_name],
        pooling=pooling,
        dropout=dropout,
        pos_weight=pos_weight
    )
    
    if PRETRAINED:
        state_dict = load_file(
            os.path.join("models", "go_emotions", str(MODEL_ID), "model.safetensors"),
            device="cuda",
        )
        state_dict.pop("classifier.out_proj.weight", None)
        state_dict.pop("classifier.out_proj.bias", None)
        model.load_state_dict(state_dict, strict=False)
    
    if FREEZE_LAYERS:
        num_layers = len(model.backbone.layers)
        freeze_layers = trial.suggest_int("freeze_layers", 1, num_layers)
        
        for param in model.parameters():
            param.requires_grad = True
        for layer in model.backbone.layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    return model

def run_stage(
    model: MambaClassifier,
    trial: optuna.Trial,
    output_dir: str,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    per_device_batch_size: int,
    warmup_steps: int,
    lr_scheduler_type: str,
    max_grad_norm: float,
    stage_name: str,
    enable_pruning: bool = True,
    extra_params: dict | None = None
) -> dict:
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
        
        save_strategy="no" if enable_pruning else "epoch",
        load_best_model_at_end=not enable_pruning,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1 if not enable_pruning else 0,        
        
        seed=SEED,
        data_seed=SEED,
    )
    write_callback = WriteResultCallback(write_results, trial.number, stage_name, dict(trial.params), extra_params)
    callbacks = [write_callback]         
    if enable_pruning:
        metrics_callback = OptunaPruningCallback(trial)
        callbacks.append(metrics_callback)   
        
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_multi if not BAH else compute_metrics_single,
        callbacks=callbacks
    )
    trainer.train()
    
    if enable_pruning:
        return {"eval_f1_macro": getattr(metrics_callback, "best_metric", None)}   
    return {"eval_f1_macro": trainer.state.best_metric}    
        
def objective(trial: optuna.Trial) -> float:
    _set_seed(SEED)
    trial_dir = os.path.join(OPTUNA_OUTPUT_DIR, f"trial_{trial.number}")    

    stage2_epochs = 20
    stage2_lr = trial.suggest_float("stage2_lr", 1e-6, 5e-5, log=True)
    stage2_weight_decay = trial.suggest_float("stage2_weight_decay", 1e-4, 0.1, log=True)
    stage2_batch_size = trial.suggest_categorical("stage2_batch_size", [16, 32, 64, 128, 256])
    
    stage2_warmup_ratio = trial.suggest_float("stage2_warmup_ratio", 0.0, 0.2)
    stage2_steps_per_epoch = len(train_ds) // stage2_batch_size
    stage2_total_steps = stage2_steps_per_epoch * stage2_epochs
    stage2_warmup_steps = int(stage2_total_steps * stage2_warmup_ratio)
    
    stage2_lr_scheduler = "cosine"
    # stage2_max_grad_norm = 1.0
    # stage2_lr_scheduler = trial.suggest_categorical("stage2_lr_scheduler", ["cosine", "linear"])
    stage2_max_grad_norm = trial.suggest_float("stage2_max_grad_norm", 0.5, 2.0)

    clf_model = build_model(trial)

    try:
        if FREEZE_BACKBONE:
            stage1_epochs = 10
            stage1_lr = trial.suggest_float("stage1_lr", 1e-5, 1e-2, log=True)
            stage1_weight_decay = trial.suggest_float("stage1_weight_decay", 1e-4, 0.1, log=True)
            stage1_batch_size = trial.suggest_categorical("stage1_batch_size", [16, 32, 64, 128, 256])

            stage1_warmup_ratio = trial.suggest_float("stage1_warmup_ratio", 0.0, 0.2)
            stage1_steps_per_epoch = len(train_ds) // stage1_batch_size
            stage1_total_steps = stage1_steps_per_epoch * stage1_epochs
            stage1_warmup_steps = int(stage1_total_steps * stage1_warmup_ratio)

            stage1_lr_scheduler = "cosine"
            stage1_max_grad_norm = 1.0
            # stage1_lr_scheduler = trial.suggest_categorical("stage1_lr_scheduler", ["cosine", "linear"])
            # stage1_max_grad_norm = trial.suggest_float("stage1_max_grad_norm", 0.5, 2.0)            
            
            extra_params = {"freeze_backbone": FREEZE_BACKBONE,
                            "stage1_lr_scheduler": stage1_lr_scheduler,
                            "stage1_max_grad_norm": stage1_max_grad_norm
                           } 
            
            clf_model.freeze_backbone()
            run_stage(
                clf_model,
                trial,
                output_dir=os.path.join(trial_dir, "stage1"),
                num_train_epochs=stage1_epochs,
                learning_rate=stage1_lr,
                weight_decay=stage1_weight_decay,
                per_device_batch_size=stage1_batch_size,
                warmup_steps=stage1_warmup_steps,
                lr_scheduler_type=stage1_lr_scheduler,
                max_grad_norm=stage1_max_grad_norm,
                stage_name="stage1",
                enable_pruning=False,
                extra_params=extra_params
            )
            clf_model.unfreeze_backbone()            
            
        extra_params = {"freeze_backbone": FREEZE_BACKBONE,
                        "stage2_lr_scheduler": stage2_lr_scheduler,
                        # "stage2_max_grad_norm": stage2_max_grad_norm,
                       }
        if FREEZE_BACKBONE:
            extra_params['stage1_lr_scheduler'] = stage1_lr_scheduler
            extra_params['stage1_max_grad_norm'] = stage1_max_grad_norm
        if BAH:
            extra_params['ah_methodology'] = AH_METHODOLOGY
        
        metrics = run_stage(
            clf_model,
            trial,
            output_dir=os.path.join(trial_dir, "stage2"),
            num_train_epochs=stage2_epochs,
            learning_rate=stage2_lr,
            weight_decay=stage2_weight_decay,
            per_device_batch_size=stage2_batch_size,
            warmup_steps=stage2_warmup_steps,
            lr_scheduler_type=stage2_lr_scheduler,
            max_grad_norm=stage2_max_grad_norm,
            stage_name="stage2",
            extra_params=extra_params
        )
        best_f1 = metrics["eval_f1_macro"]
    except torch.cuda.OutOfMemoryError as e:
        print(f"Trial {trial.number} ran out of memory, skipping: {e}")
        raise optuna.TrialPruned() from e
    finally:
        del clf_model
        gc.collect()
        torch.cuda.empty_cache()
    return best_f1        
        

def enqueue_trial_from_file(study, run_id):
    filename = f"configs/{"bah" if BAH else "emo"}_config.yaml"
    with open(filename, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    for run in data:
        if int(run["id"]) == int(run_id):
            unused_keys = {"id", "source_file", "source_id", "tokenizer", "model", "stage", "device", "eval_loss", "accuracy", "timestamp", "freeze_backbone", "ah_methodology", "epoch", "f1", "seed"}
            if "freeze_layers" in run.keys() and not FREEZE_LAYERS:
                unused_keys.add("freeze_layers")
            params = {
                key: value
                for key, value in run.items()
                if key not in unused_keys
            }
            
            if "ah_methodology" in params and AH_METHODOLOGY != params["ah_methodology"]:
                raise ValueError("ah_methodology")

            study.enqueue_trial(params)
            return params 
    
if __name__ == "__main__":    
    study_dir = os.path.join(OPTUNA_OUTPUT_DIR, f"mamba_optuna.db")  
    # optuna.delete_study(study_name=f"{FILE_NAME}_goemotions", storage=f"sqlite:///{study_dir}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=MedianPruner(n_warmup_steps=10),
        
        study_name=f"{FILE_NAME}_goemotions",
        storage=f"sqlite:///{study_dir}",
        load_if_exists=True,
    )
    
    # enqueue_trial_from_file(study, 11)
    
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, catch=(torch.cuda.OutOfMemoryError,))

    print("Best trial:")
    print(f"  value (f1_macro): {study.best_trial.value}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    print("Parameter importances:")
    importances = optuna.importance.get_param_importances(study)
    for name, score in importances.items():
        print(f"  {name}: {score:.4f}")        
