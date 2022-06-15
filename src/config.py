from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
import torch.optim as optim


class Config:
    train_path = Path(
        'data/microsoft-research-paraphrase-corpus/msr_paraphrase_train.txt')
    validate_path = Path(
        'data/microsoft-research-paraphrase-corpus/msr_paraphrase_test.txt')
    max_length = 64
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    train_bs = 8
    valid_bs = 16
    epoch = 10
    device = 'cuda'
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    base_lr = 1e-6
    max_lr = 5e-5
    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer,
    # 											base_lr=base_lr,
    # 											max_lr=max_lr,
    # 											cycle_momentum=False)
