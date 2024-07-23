import torch
from transformers import BertTokenizer, RobertaTokenizer
from datasets import load_dataset

class LIARDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", model_type="Bert", max_length=256):
        self.split = split
        self.model_type = model_type
        self.max_length = max_length

        self.dataset = load_dataset("ucsbnlp/liar")[self.split]
        self.tokenizer = self._load_tokenizer()
        self.dataset = self._preprocess_and_tokenize()

    def _load_tokenizer(self):
        if self.model_type == "Bert":
            return BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif self.model_type == "Roberta":
            return RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
        else:
            raise ValueError("Invalid model type. Choose 'Bert' or 'Roberta'.")

    def _preprocess_and_tokenize(self):
        self.dataset = self.dataset.map(self._preprocess_function)
        return self.dataset.map(self._tokenize_function, batched=True)

    def _preprocess_function(self, examples):
        examples['label'] = 1 if examples['label'] in ["true", "mostly-true"] else 0 
        examples['sentence'] = (
            examples['subject'] + " " +
            examples['speaker'] + " " +
            examples['job_title'] + " " +
            examples['state_info'] + " " +
            examples['party_affiliation'] + " " +
            examples['context'] + " " +
            examples['statement']
        )
        return examples

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples['sentence'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {
            'label': torch.tensor(self.dataset[idx]['label']).to('cuda'),
            'input_ids': torch.tensor(self.dataset[idx]['input_ids']).to('cuda'),
            'attention_mask': torch.tensor(self.dataset[idx]['attention_mask']).to('cuda'),
            'token_type_ids': torch.tensor(self.dataset[idx]['token_type_ids']).to('cuda')
        }
        return item
