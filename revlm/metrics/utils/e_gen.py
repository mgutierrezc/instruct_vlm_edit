"""
Chain-of-Error (COE) detection for VLM reasoning verification.
Standalone job: load prediction snapshot → VLM verify each COT sentence → save COE predictions.
"""
import re
import json
import os
from typing import List, Dict, Tuple, Any
from PIL import Image


def parse_cot_sentences(cot: str) -> List[str]:
    """Split COT into sentences on '.', '!', '?'"""
    if not cot:
        return []
    parts = re.split(r'(?<=[.!?])\s+', cot.strip())
    return [s.strip() for s in parts if s.strip()]


def verify_sentence(model: Any, image_path: str, sentence: str) -> Tuple[int, float, float]:
    """
    Ask VLM: is this sentence correct given the image?
    
    Returns: (error_flag, p_yes, p_no)
        error_flag: 0 if yes wins, 1 if no wins
    """
    prompt = f'Given the image, is the following statement correct? Answer yes or no.\nStatement: "{sentence}"'
    
    img = Image.open(image_path).convert("RGB")
    scores = model.score_choices_single(img, prompt, ["yes", "no"])
    
    p_yes = scores["yes"]["prob"]
    p_no = scores["no"]["prob"]
    error_flag = 0 if p_yes >= p_no else 1
    
    return error_flag, p_yes, p_no


def process_sample(model: Any, ex: Dict) -> Dict:
    """Add coe_pred field to sample."""
    cot = ex.get('cot', '') or ex.get('rationale', '')
    sentences = parse_cot_sentences(cot)
    
    if not sentences:
        ex['coe_pred'] = {
            'sentences': [],
            'errors': [],
            'confidences': [],
            'error_indices': []
        }
        return ex
    
    errors_list = []
    confidences = []
    
    for s in sentences:
        flag, p_yes, p_no = verify_sentence(model, ex['image'], s)
        errors_list.append(flag)
        confidences.append(max(p_yes, p_no))
    
    ex['coe_pred'] = {
        'sentences': sentences,
        'errors': errors_list,
        'confidences': confidences,
        'error_indices': [i for i, e in enumerate(errors_list) if e == 1]
    }
    return ex


def coe_prediction(model: Any, edit_ds: Any, config: Any) -> List[Dict]:
    """
    Main COE prediction function.
    
    Args:
        model: VQAModel instance
        edit_ds: VQADataset with error samples (from find_errors)
        config: Configuration object with pred_postedit_dir
        
    Returns:
        List of samples with coe_pred field added
    """
    n = len(edit_ds.data)
    print(f"Processing {n} error samples for COE prediction...", flush=True)
    
    results = []
    for i, ex in enumerate(edit_ds.data):
        ex_copy = ex.copy()
        ex_copy = process_sample(model, ex_copy)
        results.append(ex_copy)
        
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[{i+1}/{n}] uid={ex_copy['uid']}, errors={ex_copy['coe_pred']['errors']}", flush=True)
    
    # Save results
    out_path = os.path.join(config.pred_postedit_dir, "coe_prediction.json")
    os.makedirs(config.pred_postedit_dir, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} samples to {out_path}", flush=True)
    
    # Count edits with at least one COE error
    edits_with_coe = sum(1 for r in results if r['coe_pred']['error_indices'])
    print(f"Edits with COE: {edits_with_coe}/{len(results)} ({100*edits_with_coe/len(results):.1f}%)", flush=True)
    
    return results

def print_coe_results(results: List[Dict], max_print: int = 10) -> None:
    """Print COE results in a readable format."""
    for r in results[:max_print]:
        print(f"\n=== uid: {r['uid']} ===")
        print(f"Question: {r['question']}")
        print(f"Gold: {r['gold']['label']}, Pred: {r['pred']['label_maxprob']}")
        print(f"COT: {r.get('cot', '')}")
        coe = r['coe_pred']
        for j, (s, e, c) in enumerate(zip(coe['sentences'], coe['errors'], coe['confidences'])):
            mark = 'x' if e == 1 else '√'
            print(f"  [{mark}] {s} (conf={c:.2f})")
        print(f"Error indices: {coe['error_indices']}")


