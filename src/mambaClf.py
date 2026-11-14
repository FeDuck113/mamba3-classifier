import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class LinearClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))

    
class RobertaClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, num_labels)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:                 
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x 
    
CLF_HEADS = {
    "linear": LinearClassifierHead,
    "roberta": RobertaClassifierHead,
}    

# ---------------------------------------------------

class MambaClassifier(nn.Module):
    def __init__(
        self,
        mamba_lm,        
        num_labels: int,
        problem_type: str = "single_label_classification",        
        pooling: str = "mean",  
        dropout: float = 0.1,
        classifier_head: type[nn.Module] | None = None,
        pos_weight: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.backbone = mamba_lm.backbone
        self.config = mamba_lm.config
        
        self.num_labels = num_labels
        self.pooling = pooling
        self.problem_type = problem_type       
                    
        self.register_buffer("pos_weight", pos_weight, persistent=False)
        self.register_buffer("class_weights", class_weights, persistent=False)   
        
        if classifier_head is not None:
            self.classifier = classifier_head(self.config.d_model, self.num_labels, dropout)
        else:
            self.classifier = LinearClassifierHead(self.config.d_model, self.num_labels, dropout)

    def _pool(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden: (B, L, D)
        if hidden.size(1) == 0:
            return hidden.new_zeros(hidden.size(0), hidden.size(-1))
        
        elif self.pooling == "last":
            if attention_mask is None:
                return hidden[:, -1]

            last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        
        elif self.pooling == "first":
            return hidden[:, 0]
        
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).bool() # !
                hidden = hidden.masked_fill(~mask, float("-inf"))
            return hidden.max(dim=1).values
        
        if self.pooling == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)

            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)   

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):             
        hidden = self.backbone(input_ids)
        pooled = self._pool(hidden, attention_mask)        
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            if self.problem_type == "multi_label_classification":                
                pos_weight = None
                if self.pos_weight is not None:
                    pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), 
                                                          pos_weight=pos_weight.float()
                                                          if pos_weight is not None else None)    
            elif self.problem_type == "single_label_classification":
                class_weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
                loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
            elif self.problem_type == "regression":
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
            else:
                raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def predict(self, text: str | list[str]):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("No tokenizer is attached; load the model with src.model_loader.load_model().")

        single_input = isinstance(text, str)
        texts = [text] if single_input else list(text)
        if not texts:
            return []

        device = next(self.parameters()).device
        max_length = getattr(self, "max_length", 256)
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        self.eval()
        with torch.inference_mode():
            logits = self(**inputs).logits

        id2label = getattr(self, "id2label", {})
        get_label = lambda index: id2label.get(index, str(index))
        if self.problem_type == "multi_label_classification":
            probabilities = torch.sigmoid(logits)
            threshold = getattr(self, "threshold", 0.5)
            predictions = [
                [get_label(i) for i, score in enumerate(row) if score >= threshold]
                for row in probabilities
            ]
        else:
            predictions = [get_label(i) for i in logits.argmax(dim=-1).tolist()]

        return predictions[0] if single_input else predictions

    def tokenize_batch(self, batch):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("No tokenizer is attached; load the model with src.model_loader.load_model().")
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=getattr(self, "max_length", 256),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True
