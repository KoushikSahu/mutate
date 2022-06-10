import torch
from torch.utils.data import Dataset
from config import Config


class MSRPCDataset(Dataset):
	def __init__(self,
					df,
					text_col,
					paraphrase_col,
					tokenizer,
					max_length
				):
		self.df = df
		self.text_col = text_col
		self.paraphrase_col = paraphrase_col
		self.tokenizer = tokenizer
		self.max_length = max_length


	def __len__(self):
		return len(self.df)

	
	def __getitem__(self, idx):
		text = self.df.loc[idx, self.text_col]
		paraphrase = self.df.loc[idx, self.paraphrase_col]

		text_token = self.tokenizer(text,
										padding='max_length',
										truncation=True,
										max_length=Config.max_length
									)
		paraphrase_token = self.tokenizer(paraphrase,
											padding='max_length',
											truncation=True,
											max_length=Config.max_length
										)

		return {
			'text': text,
			'paraphrase': paraphrase,
			'input_ids': torch.tensor(text_token['input_ids']),
			'attention_mask': torch.tensor(text_token['attention_mask']),
			'labels': torch.tensor(paraphrase_token['input_ids'])
		}
		