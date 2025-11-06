from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .utils import *

class VLMDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data = []
        self._load_data()
        
    def _load_data(self):
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_data")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def set_dataloader(self,
                        # task engineer
                        task="mc",  
                        with_rationale=False,
                        rationale_in_prompt=True,
                        shuffle_choices=False,
                        unpaired=False,
                        seed=333,
                        # dataloader
                        batch_size=32,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True):
        """
        task: "mc": multiple choices, queried with "choices" field, i.e "car; person; flower; animal"
        task: "mci": multiple choices (indexed with letters), queried with "idx_choices" field, i.e "(A) car\n(B) bike\n(C) train\n(D) bus"
        task: "qa": free generation qa, provided with no "choices", require to return one word or one phrase.
        """
        self.task_engineer = get_taskengineer(task, 
                                              with_rationale=with_rationale, 
                                              rationale_in_prompt=rationale_in_prompt,
                                              shuffle_choices=shuffle_choices,
                                              unpaired=unpaired,
                                              seed=seed)
        for i, ex in enumerate(self.data): # ex is a reference to the dict stored in self.data
            ex['idx'] = i
            self.task_engineer.eng_golds(ex)
            self.task_engineer.eng_prompt(ex)
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=self.image_collate)
        
    def _resize_image(self, img, max_side=800):
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            s = max_side / m
            img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
        return img
    
    def image_collate(self, batch):
        """Collate function that loads images and returns a batch dict.
        Expects items with keys: 'image' (path), 'prompt' (string), 'gold' (dict), 'idx' (int).
        """
        images = [self._resize_image(Image.open(ex["image"]).convert("RGB")) for ex in batch]
        prompts = [ex["prompt"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        idxs = [ex["idx"] for ex in batch]
        return {
            "images": images,
            "prompts": prompts,
            "golds": golds,
            "idxs": idxs,
        }

    def task_generate(self, batch, model):
        """Generate predictions for a single collated batch and write back in place using indices."""
        outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=100)
        for idx, a in zip(batch["idxs"], outs):
            self.task_engineer.eng_preds(self.data[idx], a, model)

class AOKVQADataset(VLMDataset):
    def __init__(self, split: str = "train"):
        super().__init__(split=split)

    def _load_data(self):
        split = self.split if self.split in ("train", "val", "test") else "train"
        split_paths = data_download_parquet_splits(
            repo_id="JJoy333/RationaleVQA",
            path_in_repo="AOKVQA",
        )
        df = data_load_split_df(split_paths.get(split))
        self.data = data_rows_to_examples(df)


class FVQADataset(VLMDataset):
    def __init__(self, split: str = "train"):
        super().__init__(split=split)

    def _load_data(self):
        split = self.split if self.split in ("train", "test") else "train"
        split_paths = data_download_parquet_splits(
            repo_id="JJoy333/RationaleVQA",
            path_in_repo="FVQA",
        )
        df = data_load_split_df(split_paths.get(split))
        self.data = data_rows_to_examples(df)


def get_dataset(config, split="train"):
    dataset_name = str(config.experiment.dataset_name).strip().lower()
    # Prefer explicit dataset_name if provided
    if dataset_name == "aokvqa":
        edit_dataset = AOKVQADataset(split=split)
    elif dataset_name == "fvqa":
        edit_dataset = FVQADataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return edit_dataset
