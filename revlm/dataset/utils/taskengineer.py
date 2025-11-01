# TaskEngineer is a class that engineers task-dependent inputs and outputs for a given example.
# ex is a binding to the dict stored in self.data, mutable in place
# (1) prompt 
# (2) golds
#   - label (answer text)
#   - choices (if applicable)
#   - label_letter (if applicable)
# (3) preds
#   - label (answer text) from VLM's answer 
#   - label_letter (if applicable) from VLM's answer


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
    

    def _eng_choices(self, s: str):
        # str_choices i.e. "car; person; flower; animal"
        choices_list = s.split(';')
        choices_list = [choice.strip() for choice in choices_list]
        if self.shuffle_choices:
            rng = random.Random(self.seed)
            rng.shuffle(choices_list)
        choices_str = '; '.join(choices_list)
        return {"str": choices_str, "ls": choices_list}

    def eng_golds(self, ex):
        ex['golds'] = {}
        ex['golds']['label'] = str(ex['answer']).lower().strip()
        ex['golds']['choices'] = self._eng_choices(ex['choices'])
    
    def eng_prompt(self, ex):
        sys_prompt = "Choose the correct answer from the options."
        base = f"{sys_prompt} {ex['question']} Options: {ex['golds']['choices']['str']}".strip()
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale else base

    def eng_preds(self, answer: str):
        return answer.lower().strip()


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
        rng = random.Random(self.seed)
        pairs = self.extract_choice_pairs(idx_choices_s)
        if not self.unpaired: # paired shuffle
            rng.shuffle(pairs)
        else: # unpaired shuffle
            letters = [ltr for ltr, _ in pairs]
            options = [opt for _, opt in pairs]
            rng.shuffle(letters)
            rng.shuffle(options)
            pairs = list(zip(letters, options))
        s = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])
        return {"str": s, "ls": pairs}
    

    def eng_golds(self, ex):
        ex['golds'] = {}
        ex['golds']['choices'] = self.eng_idx_choices(ex['idx_choices'])
        ex['golds']['label'] = str(ex['answer']).lower().strip()
        ex['golds']['label_letter'] = self.get_gold_label_letter(ex['golds']['label'], ex['golds']['choices']['str'])

    def eng_prompt(self, ex):
        sys_prompt = "Choose A/B/C/D from the options."
        base = f"{sys_prompt} {ex['question']} Options: {ex['golds']['choices']['str']}".strip()
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale else base

    def eng_preds(self, answer: str):
        return answer.lower().strip()


class QATaskEngineer(TaskIOEngineer):
    def __init__(self, with_rationale=False, **kwargs):
        super().__init__()
        self.with_rationale = with_rationale
        self.name = "qa"
        self.seed = 333

    def eng_golds(self, ex): 
        ex['golds'] = {}
        ex['golds']['label'] = str(ex['answer']).lower().strip()
    
    def eng_prompt(self, ex):
        sys_prompt = "Answer the question in one word or phrase."
        base = f"{sys_prompt} {ex['question']}".strip()     
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale else base

    def eng_preds(self, answer: str):
        return answer.lower().strip()


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