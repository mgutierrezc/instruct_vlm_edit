# ----------------------------------------------------------------------------
# COE Scenario Generation: Generate new scenarios for error chains using GPT-4o
# ----------------------------------------------------------------------------
from pathlib import Path
import json
import math
import os
from typing import List, Dict, Any

from openai import OpenAI
from .e_gen import load_coe


class COEScenarioGenerator:
    def __init__(self, dataset_name: str, model_name: str, openai_key: str = None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.base_dir = Path(f"./data/coe_gen/{model_name}/{dataset_name}")
        self.batch_dir = self.base_dir / "requests"
        self.meta_dir = self.base_dir / "meta"
        self.output_dir = self.base_dir / "outputs"
        
        for d in [self.batch_dir, self.meta_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.client = OpenAI(api_key=openai_key)
        self.n_batches = 20 if dataset_name == "fvqa" else 40
        self.k = 3  # number of scenarios to generate
        print(f"COEScenarioGenerator: {model_name}/{dataset_name}, {self.n_batches} batches")

    def format_prompt(self, error_chain: str) -> str:
        return (
            f'Given these visual facts:\n"{error_chain}"\n\n'
            f"Generate {self.k} different creative scenarios where ALL these facts would be visually true.\n\n"
            "Requirements:\n"
            "- Each scenario should be a distinct visual setting, in 2-3 sentences\n"
            "- Be creative but plausible\n"
            "- Describe what would be visible in the image\n\n"
            "Examples:\n"
            'Visual facts: "A person is standing on a board. There are waves around."\n'
            "1. A surfer rides a wave at a tropical beach during sunset. The ocean is blue with white foam. Palm trees line the shore in the background.\n"
            "2. A wakeboarder is pulled behind a speedboat on a lake. The boat creates a large wake. Mountains are visible in the distance.\n"
            "3. A paddleboarder balances on calm ocean waters near a rocky coastline. Seagulls fly overhead. The sky is overcast.\n\n"
            'Visual facts: "The cake has multiple tiers. There are decorations on top."\n'
            "1. A wedding cake sits on a decorated table at an outdoor garden ceremony. White roses adorn each tier. Guests mingle in the background.\n"
            "2. A birthday cake is displayed in a bakery window. Colorful fondant figures sit on top. The shop interior has warm lighting.\n"
            "3. An elaborate anniversary cake is being served at a rooftop restaurant. City lights sparkle behind. A couple holds champagne glasses nearby.\n\n"
            "Now generate scenarios for the given facts:\n"
            "Respond as a numbered list:\n1. [scenario]\n2. [scenario]\n3. [scenario]"
        )

    def _get_error_chains(self, coe_pred: Dict) -> List[Dict]:
        """Extract all error subsets as chains.
        
        Returns list of {indices: [...], chain: "sentence1 sentence2 ..."}
        """
        sentences = coe_pred.get("sentences", [])
        error_chains = []
        for sub in coe_pred.get("subsets", []):
            if sub["error"] == 1:
                chain = " ".join(sentences[i] for i in sub["indices"])
                error_chains.append({"indices": sub["indices"], "chain": chain})
        return error_chains

    def gen_request(self, coe_results: List[Dict], max_sentences: int = None):
        """Generate batch request from COE prediction results."""
        json_data = []
        for r in coe_results:
            # Optional filter by sentence count
            n_sent = len(r.get("coe_pred", {}).get("sentences", []))
            if max_sentences and n_sent > max_sentences:
                continue
            
            error_chains = self._get_error_chains(r.get("coe_pred", {}))
            for ec in error_chains:
                indices_str = ",".join(map(str, ec["indices"]))
                json_data.append({
                    "custom_id": f"{r['uid']}_[{indices_str}]",  # uid_[0,1,2]
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You generate creative visual scenarios from given facts."},
                            {"role": "user", "content": self.format_prompt(ec["chain"])},
                        ],
                        "max_tokens": 300,
                        "temperature": 1.0
                    }
                })
        
        input_file = self.batch_dir / "batch.jsonl"
        with open(input_file, "w") as f:
            for d in json_data:
                f.write(json.dumps(d) + "\n")
        print(f"Created {len(json_data)} requests in {input_file}")
        return len(json_data)

    def split_file(self, remove_original: bool = True):
        input_file = self.batch_dir / "batch.jsonl"
        with open(input_file, "r") as f:
            lines = f.readlines()
        if not lines:
            return
        per_file = max(1, math.ceil(len(lines) / self.n_batches))
        for i in range(self.n_batches):
            chunk = lines[i * per_file : (i + 1) * per_file]
            if not chunk:
                break
            with open(self.batch_dir / f"batch_{i}.jsonl", "w") as f:
                f.writelines(chunk)
        if remove_original:
            os.remove(input_file)

    def _save_meta(self, b: int, job_id: str, created_at=None):
        meta = {"job_id": job_id, "batch": b, "created_at": created_at}
        with open(self.meta_dir / f"meta_{b}.json", "w") as f:
            json.dump(meta, f)
        print(f"Batch {b}: {job_id}")

    def _load_meta(self, b: int) -> Dict:
        with open(self.meta_dir / f"meta_{b}.json", "r") as f:
            return json.load(f)

    # ----- Runner functions -----
    def run_input(self, coe_results: List[Dict], max_sentences: int = None):
        """Prepare batch input from COE results."""
        n = self.gen_request(coe_results, max_sentences)
        if n > 0:
            self.split_file()

    def run_request(self):
        """Submit all batches to OpenAI."""
        for b in range(self.n_batches):
            self._run_request_batch(b)

    def resubmit_request(self, b_list: List[int]):
        """Cancel and resubmit specific batches."""
        for b in b_list:
            meta_path = self.meta_dir / f"meta_{b}.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                job_id = meta["job_id"]
                try:
                    status = self.client.batches.retrieve(job_id).status
                    if status in {"validating", "queued", "in_progress"}:
                        self.client.batches.cancel(job_id)
                except Exception:
                    pass
                meta_path.unlink()
            output_path = self.output_dir / f"outputs_{b}.jsonl"
            if output_path.exists():
                os.remove(output_path)
            self._run_request_batch(b)

    def _run_request_batch(self, b: int):
        batch_file = self.batch_dir / f"batch_{b}.jsonl"
        meta_file = self.meta_dir / f"meta_{b}.json"
        if not batch_file.exists() or meta_file.exists():
            return
        with open(batch_file, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        meta = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "COE scenario generation", "batch": str(b)}
        )
        self._save_meta(b, meta.id, meta.created_at)

    def get_scenarios(self) -> List[Dict]:
        """Get all scenario results."""
        results = []
        for b in range(self.n_batches):
            try:
                results.extend(self._get_response_batch(b))
            except Exception as e:
                print(f"Batch {b} error: {e}")
        return results

    def _get_response_batch(self, b: int) -> List[Dict]:
        save_path = self.output_dir / f"outputs_{b}.jsonl"
        if save_path.exists():
            with open(save_path, "r") as f:
                return [json.loads(line) for line in f]
        
        meta = self._load_meta(b)
        job = self.client.batches.retrieve(meta["job_id"])
        if not job.output_file_id:
            raise RuntimeError(f"Batch {b} not ready (status: {job.status})")
        
        content = self.client.files.content(job.output_file_id)
        records = []
        for line in content.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            custom_id = payload.get("custom_id", "")
            # Parse uid_[0,1,2] format
            if "_[" in custom_id:
                uid, indices_str = custom_id.rsplit("_[", 1)
                indices = [int(x) for x in indices_str.rstrip("]").split(",")]
            else:
                uid, indices = custom_id, []
            
            text = payload.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            # Parse numbered list
            scenarios = [s.strip() for s in text.split("\n") if s.strip() and s.strip()[0].isdigit()]
            scenarios = [s.split(". ", 1)[1] if ". " in s else s for s in scenarios]
            records.append({"uid": uid, "indices": indices, "scenarios": scenarios, "raw": text})
        
        with open(save_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return records

    # ----- Convenience: use config directly -----
    @classmethod
    def from_config(cls, config: Any, openai_key: str = None) -> "COEScenarioGenerator":
        """Create generator from config object."""
        # Extract model_name from config.pred_postedit_dir path
        # e.g., "results/pred_postedit/baseline/Qwen3-VL-8B-Instruct/fvqa" -> "Qwen3-VL-8B-Instruct"
        parts = config.pred_postedit_dir.split("/")
        model_name = parts[-2]
        dataset_name = parts[-1]
        return cls(dataset_name, model_name, openai_key)

    def run_from_config(self, config: Any, max_sentences: int = None):
        """Load COE results from config and prepare batch input."""
        results, coe_rate = load_coe(config)
        self.run_input(results, max_sentences)
        return results, coe_rate

