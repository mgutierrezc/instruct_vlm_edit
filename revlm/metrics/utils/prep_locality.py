# """Build unrelated inputs (locality) via simple random sampling."""

# import random
# from typing import Any, Dict, List, MutableMapping, Sequence


# def sample_unrelated_random(
# 	examples: Sequence[MutableMapping[str, Any]],
# 	num_samples: int,
# 	text_key: str = "prompt",
# 	rng: random.Random | None = None,
# ) -> List[Dict[str, Any]]:
# 	"""Randomly sample unrelated items from dataset examples.

# 	Example
# 	-------
# 	>>> exs = [
# 	...     {"image": object(), "prompt": "q1"},
# 	...     {"image": object(), "prompt": "q2"},
# 	... ]
# 	>>> len(sample_unrelated_random(exs, 1))
# 	1
# 	"""
# 	rng = rng or random
# 	pool = list(examples)
# 	if num_samples <= 0:
# 		return []
# 	if num_samples >= len(pool):
# 		return [{"image": ex["image"], "text": ex[text_key]} for ex in pool]
# 	indices = rng.sample(range(len(pool)), k=num_samples)
# 	return [{"image": pool[i]["image"], "text": pool[i][text_key]} for i in indices]


 


 


