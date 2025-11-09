from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from .utils import *

class VQADataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        df = self.load_df()
        self.data = self.df2data(df)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_df(self):
        if self.config.experiment.dataset_name == "fvqa":
            path_in_repo = "FVQA"
        elif self.config.experiment.dataset_name == "aokvqa":
            path_in_repo = "AOKVQA"
        else:
            raise ValueError(f"Unknown dataset: {self.config.experiment.dataset_name}")
        split_paths = data_download_parquet_splits(
            repo_id="JJoy333/RationaleVQA",
            path_in_repo=path_in_repo,
        )
        df = data_load_split_df(split_paths.get(self.config.experiment.split))
        return df
    
    def df2data(self, df: pd.DataFrame) -> List[Dict]:
        cols = ["uid", "image_path", "question", "answer", "rationale", "choices", "idx_choices"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")
        if df.empty:
            return []

        records = df[cols].to_dict(orient="records")
        examples: List[Dict] = []
        for r in records:
            ex: Dict[str, object] = {
                "uid": r["uid"],
                "image": r["image_path"],
                "question": r["question"],
                "answer": r["answer"],
                "rationale": r["rationale"],
                "choices": r["choices"],
                "idx_choices": r["idx_choices"],
            }
            examples.append(ex)
        return examples
    
    def set_dataloader(self,
                        with_rationale=False,
                        rationale_in_prompt=True,
                        shuffle_choices=False,
                        unpaired=True):
            
        task = self.config.experiment.task
        batch_size = self.config.batch_size
        seed = self.config.seed

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
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=self.image_collate)
        
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
        # images = [Image.open(ex["image"]).convert("RGB") for ex in batch]
        prompts = [ex["prompt"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        idxs = [ex["idx"] for ex in batch]
        return {
            "images": images,
            "prompts": prompts,
            "golds": golds,
            "idxs": idxs,
        }

    # def task_generate_batch(self, batch, model):
    #     """Generate predictions for a single collated batch and write back in place using indices."""
    #     outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=100)
    #     for idx, a in zip(batch["idxs"], outs):
    #         self.task_engineer.eng_preds(self.data[idx], a, model)

    def task_generate(self, model):
        for batch in self.loader:
            try: # Reset cached rope offsets for Qwen3-VL style models to avoid mask/id length mismatch
                inner = getattr(getattr(model, "model", None), "model", None)
                if inner is not None and hasattr(inner, "rope_deltas"):
                    inner.rope_deltas = None
            except Exception:
                pass
            outs = model.generate(batch["images"], batch["prompts"], max_new_tokens=100, use_cache=False)
            for idx, a in zip(batch["idxs"], outs):
                self.task_engineer.eng_preds(self.data[idx], a, model)
    
    def get_edits(self):
        pred_by = self.config.experiment.pred_by
        for ex in self.data:
            if ex['gold']['label'] != ex['pred'][pred_by]:
                ex['edit'] = True
            else:
                ex['edit'] = False
        edit_ds = self
        edit_ds.data = [ex for ex in edit_ds.data if ex['edit']]
        edit_ds.set_dataloader(shuffle_choices=False)
        return edit_ds

    def snap(self, out_path=None) -> None:
        if out_path is None:
            out_path = os.path.join(self.config.pred_dir, self.config.fname)
        with open(out_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def task_eval(self) -> None:
        task_metrics = self.task_engineer.eval(self)
        out_path = os.path.join(self.config.task_dir, self.config.fname)
        with open(out_path, "w") as f:
            json.dump(task_metrics, f, indent=2)
        print(f"Saved task evaluation metrics to {out_path}")
        print(f"Task evaluation metrics: {task_metrics}", flush=True)