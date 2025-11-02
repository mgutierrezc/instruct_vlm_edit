# TaskEngineer is a class that engineers task-dependent inputs and outputs for a given example.
# ex is a binding to the dict stored in self.data, mutable in place
# (1) prompt 
# (2) gold
#   - label (answer text)
#   - choices (if applicable)
#   - label_letter (if applicable)
# (3) pred
# for mc
#   - label_text: label from model answer text, compared with choices in mc
#   - label_scores: scores of label from model answer text, compared with choices in mc
#   - label_maxprob: max probability of label from model answer text, compared with choices in mc
# for mci
#   - label_text: label from model answer text, compared with choices in mci and mc
#   - letter_text: letter from model answer text 
#   - label_scores: scores of label from model answer text, compared with choices in mci and mc
#   - letter_scores: scores of letter from model answer text 
#   - label_maxprob: max probability of label from model answer text, compared with choices in mci and mc
#   - letter_maxprob: max probability of letter from model answer text 
# for qa
#   - label_text: label from model answer text, compared with choices in qa
#   - label_scores: scores of label from model answer text, compared with choices in qa

import random
import re

class TaskIOEngineer:
    def __init__(self):
        self.name = "default"
    def eng_prompt(self, ex):
        raise NotImplementedError("Subclasses must implement eng_prompt")
    def eng_golds(self, ex):
        raise NotImplementedError("Subclasses must implement eng_golds")
    def eng_preds(self, ex):
        raise NotImplementedError("Subclasses must implement eng_preds")


class MCTaskEngineer(TaskIOEngineer):
    def __init__(self, 
                with_rationale=False,
                shuffle_choices=False, 
                seed=333,
                **kwargs):
        super().__init__()
        self.name = "mc"
        self.with_rationale = with_rationale
        self.shuffle_choices = shuffle_choices
        self.seed = seed
        self.rng = random.Random(seed)
    

    def _eng_choices(self, s: str):
        # str_choices i.e. "car; person; flower; animal"
        choices_list = s.split(';')
        choices_list = [choice.strip() for choice in choices_list]
        if self.shuffle_choices:
            self.rng.shuffle(choices_list)
        choices_str = '; '.join(choices_list)
        return {"str": choices_str, "ls": choices_list}

    def eng_golds(self, ex):
        ex['gold'] = {}
        ex['gold']['label'] = str(ex['answer']).lower().strip()
        ex['gold']['choices'] = self._eng_choices(ex['choices'])
    
    def eng_prompt(self, ex):
        sys_prompt = "Choose the correct answer from the options."
        base = f"{sys_prompt} {ex['question']} Options: {ex['gold']['choices']['str']}".strip()
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale else base

    def eng_preds(self, ex, a: str, model):
        """ example s: 
        {'cab': {'avg_nll': 3.2672276496887207,
            'sum_nll': 3.2672276496887207,
            'num_tokens': 1,
            'prob': 0.8913333874394938},
        'skateboarder': {'avg_nll': 1.5983691215515137,
            'sum_nll': 6.393476486206055,
            'num_tokens': 4,
            'prob': 0.039113578901274634},...}
        """
        # text-based generation
        ex['pred'] = {}
        ex['pred']['answer'] = a
        # label is the substring in answer(a) that matches any element of ex['gold']['choices']['ls']
        for choice in ex['gold']['choices']['ls']:
            if choice.lower() in a.lower().strip():
                ex['pred']['label_text'] = choice
                break
        # score-based generation
        s = model.score_choices_single(ex['image'], ex['prompt'], ex['gold']['choices']['ls'])
        ex['pred']['label_scores'] = s
        ex['pred']['label_maxprob'] = max(s, key=lambda k: s[k]['prob'])



class MCITaskEngineer(TaskIOEngineer):
    def __init__(self, 
                with_rationale=False,
                shuffle_choices=False,
                unpaired=False,
                seed=333,
                **kwargs):
        super().__init__()
        self.name = "mci"
        self.with_rationale = with_rationale
        self.shuffle_choices = shuffle_choices
        self.unpaired = unpaired
        self.seed = seed
        self.rng = random.Random(seed)

    def extract_choice_pairs(self, s: str):
        pairs = re.findall(r"\(([A-D])\)\s*(.+)", s)
        return [(ltr, txt.strip()) for (ltr, txt) in pairs]

    def get_gold_label_letter(self, label_s:str, idx_choices_s:str):
        pairs = self.extract_choice_pairs(idx_choices_s)
        letter = None
        for ltr, opt in pairs:
            if opt.strip().lower() == label_s.lower():
                letter = ltr
                break
        return letter


    def eng_idx_choices(self, idx_choices_s: str):
        # idx_choises_s : letter indexed choices string like "(A) car\n(B) bike\n(C) train\n(D) bus"
        # return s: 
        # "(D) bus\n\n(B) bike(A) car\n(C) train" if paired shuffle (maintain letter-option association)
        # "(D) car\n(C) bike\n(A) train\n(B) bus" if unpaired shuffle (shuffle both letter and option)
        pairs = self.extract_choice_pairs(idx_choices_s)
        # If shuffle disabled, keep original order
        if not self.shuffle_choices:
            s = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])
            return {"str": s, "ls": pairs}

        if not self.unpaired:  # paired shuffle
            self.rng.shuffle(pairs)
        else:  # unpaired shuffle
            letters = [ltr for ltr, _ in pairs]
            options = [opt for _, opt in pairs]
            self.rng.shuffle(letters)
            self.rng.shuffle(options)
            pairs = list(zip(letters, options))
        s = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])
        return {"str": s, "ls": pairs}
    

    def eng_golds(self, ex):
        ex['gold'] = {}
        ex['gold']['choices'] = self.eng_idx_choices(ex['idx_choices'])
        ex['gold']['label'] = str(ex['answer']).lower().strip()
        ex['gold']['label_letter'] = self.get_gold_label_letter(ex['gold']['label'], ex['gold']['choices']['str'])

    def eng_prompt(self, ex):
        sys_prompt = "Choose A/B/C/D from the options."
        base = f"{sys_prompt} {ex['question']} Options: {ex['gold']['choices']['str']}".strip()
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale else base

    def eng_preds(self, ex, a: str, model):
        # text-based generation
        ex['pred'] = {}
        ex['pred']['answer'] = a
        ex['pred']['letter_text'] = None
        ex['pred']['label_text'] = None
        # label is the substring in answer(a) that matches any element of ex['gold']['choices']['ls']
        for (ltr, _) in ex['gold']['choices']['ls']:
            if ltr.lower() in a.lower().strip():
                ex['pred']['letter_text'] = ltr
                break
        for (_, choice) in ex['gold']['choices']['ls']:
            if choice.lower() in a.lower().strip():
                ex['pred']['label_text'] = choice
                break
        
        # score-based generation
        label_texts = [choice for _, choice in ex['gold']['choices']['ls']]
        label_letters = [ltr for ltr, _ in ex['gold']['choices']['ls']]
        ex['pred']['label_scores'] = model.score_choices_single(ex['image'], ex['prompt'], label_texts)
        ex['pred']['letter_scores'] = model.score_choices_single(ex['image'], ex['prompt'], label_letters)
        ex['pred']['label_maxprob'] = max(ex['pred']['label_scores'], key=lambda k: ex['pred']['label_scores'][k]['prob'])
        ex['pred']['letter_maxprob'] = max(ex['pred']['letter_scores'], key=lambda k: ex['pred']['letter_scores'][k]['prob'])
        



class QATaskEngineer(TaskIOEngineer):
    def __init__(self, with_rationale=False, **kwargs):
        super().__init__()
        self.with_rationale = with_rationale
        self.name = "qa"
        self.seed = 333

    def eng_golds(self, ex): 
        ex['gold'] = {}
        ex['gold']['label'] = str(ex['answer']).lower().strip()
    
    def eng_prompt(self, ex):
        sys_prompt = "Answer the question in one word or phrase."
        base = f"{sys_prompt} {ex['question']}".strip()     
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale else base

    def eng_preds(self, ex, a: str, model):
        ex['pred'] = {}
        ex['pred']['answer'] = a
        ex['pred']['label_text'] = a.lower().strip()
        ex['pred']['label_scores'] = model.score_choices_single(ex['image'], ex['prompt'], [ex['gold']['label']])
        


def get_taskengineer(task: str, **kwargs):
    task = task.lower()
    if task == "mc":
        return MCTaskEngineer(**kwargs)
    elif task == "mci":
        return MCITaskEngineer(**kwargs)
    elif task == "qa":
        return QATaskEngineer(**kwargs)
    else:
        raise ValueError(f"Unknown task engineer: {task}")