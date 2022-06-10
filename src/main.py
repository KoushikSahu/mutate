from config import Config
import csv
import pandas as pd
from dataloader import get_dataloader
from dataset import MSRPCDataset
from tqdm import tqdm
from train import train
import torch
from validate import validate


def main():
	train_df = pd.read_csv(Config.train_path, sep='\t',
							quoting=csv.QUOTE_NONE)
	valid_df = pd.read_csv(Config.validate_path, sep='\t',
							quoting=csv.QUOTE_NONE)

	train_ds = MSRPCDataset(train_df,
								'#1 String',
								'#2 String',
								Config.tokenizer,
								Config.max_length)
	valid_ds = MSRPCDataset(valid_df,
								'#1 String',
								'#2 String',
								Config.tokenizer,
								Config.max_length)

	train_dl = get_dataloader(train_ds, Config.train_bs)
	valid_dl = get_dataloader(valid_ds, Config.valid_bs)

	model = Config.model
	model = model.to('cuda')
	optimizer = Config.optimizer
	loss_fn = None

	for epoch in range(Config.epoch):
		train(train_dl, model, optimizer, loss_fn)
		# Config.scheduler.step()
		validate(valid_dl, model)

	validate(valid_dl, model, verbose=True)

if __name__ == '__main__':
	main()
