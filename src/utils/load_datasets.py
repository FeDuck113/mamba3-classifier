from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np

def load_go_emotions(tokenize_func, num_labels, return_label_names=False):
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    label_names = dataset["train"].features["labels"].feature.names
    
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    train_ds = train_ds.map(tokenize_func, batched=True, load_from_cache_file=False)
    val_ds = val_ds.map(tokenize_func, batched=True, load_from_cache_file=False)
    test_ds = test_ds.map(tokenize_func, batched=True, load_from_cache_file=False)

    def make_multihot(batch):
        multihot = np.zeros((len(batch["labels"]), num_labels), dtype=np.float32)
        for idx, labels in enumerate(batch["labels"]):
            multihot[idx, labels] = 1.0
        batch["labels"] = multihot
        return batch

    train_ds = train_ds.map(make_multihot, batched=True, load_from_cache_file=False)
    val_ds = val_ds.map(make_multihot, batched=True, load_from_cache_file=False)
    test_ds = test_ds.map(make_multihot, batched=True, load_from_cache_file=False)

    train_labels_np = np.array(train_ds["labels"], dtype=np.float32)
    pos_counts = train_labels_np.sum(axis=0)
    neg_counts = train_labels_np.shape[0] - pos_counts
    base_pos_weight_ratio = neg_counts / np.clip(pos_counts, 1, None)

    train_ds = train_ds.remove_columns(["text", "id"])
    val_ds = val_ds.remove_columns(["text", "id"])
    test_ds = test_ds.remove_columns(["text", "id"])

    train_ds = train_ds.with_format("torch")
    val_ds = val_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    
    result = (train_ds, val_ds, test_ds, base_pos_weight_ratio)
    if return_label_names:
        return (*result, label_names)
    return result

# -------------------------------

def load_bah(tokenize_func, df_dir="bah_data.json", ah_methodology="max"):
    video_df = pd.read_json(df_dir)
    
    df_fulltext = video_df.groupby(["id", "question_id"], as_index=False).agg({"text": lambda s: "".join(s).strip(), "has_ah": ah_methodology, "split_type": "first"})
    train_df = df_fulltext[df_fulltext['split_type'] == 'train'][['text', 'has_ah']]
    val_df = df_fulltext[df_fulltext['split_type'] == 'val'][['text', 'has_ah']]
    test_df = df_fulltext[df_fulltext['split_type'] == 'test'][['text', 'has_ah']]
    
    def _prepare_df(df):
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        ds = ds.rename_column("has_ah", "labels")
        ds = ds.map(tokenize_func, batched=True, load_from_cache_file=False)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds
    
    train_df = _prepare_df(train_df)
    val_df = _prepare_df(val_df)
    test_df = _prepare_df(test_df)
    
    train_labels_np = np.array(train_df["labels"], dtype=np.float32)
    pos_count = train_labels_np.sum()
    neg_count = len(train_labels_np) - pos_count
    base_pos_weight_ratio = neg_count / max(pos_count, 1)
    
    return train_df, val_df, test_df, base_pos_weight_ratio

def prepare_bah(train_df, val_df, tokenizer, max_len):
    def _tokenize(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_len)

    def _prepare(df):
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        ds = ds.rename_column("has_ah", "labels")
        ds = ds.map(_tokenize, batched=True, load_from_cache_file=False)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds
    
    return _prepare(train_df), _prepare(val_df)
    
