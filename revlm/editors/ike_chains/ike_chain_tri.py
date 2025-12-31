import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from .utils import brackets_to_periods, parent_module, Augmenter


class IKE_CHAIN(nn.Module):
    """Tri-encoder chain-of-keys codebook with independent subkey radii.
    
    Key structure: k = (v, l, vl) where each subkey has its own radius
    - v: vision(<img, "">) with radius_v
    - l: lang(<blank, text>) with radius_l  
    - vl: vision(<img, text>) with radius_vl
    
    Entry retrieval: AND-gate - query must be within radius of ALL THREE subkeys.
    Radius: 99th percentile of 10 augmented samples per subkey.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # Hyperparams
        self.top_n = int(getattr(cfg, "top_n", 30))
        self.cap_k = int(getattr(cfg, "cap_k", 3))
        self.prefix = getattr(cfg, "cot_prefix", "New Fact: ")
        self.distance = getattr(cfg, "distance", "l2")
        self.neighbor_window = int(getattr(cfg, "neighbor_window", 0))
        
        # Radius config: "augment" (99th percentile of n_radius_samples augs) or "fixed"
        self.radius_method = getattr(cfg, "radius_method", "fixed")
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 10))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 99))
        
        # Augmentation config
        self.n_aug_entry = int(getattr(cfg, "n_aug_entry", 1))
        self.n_aug_sent = int(getattr(cfg, "n_aug_sent", 1))
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        self.augmenter = Augmenter(self.wrapper, dataset_name=dataset_name)  # Always needed for radius estimation
        
        # Switch for plotting
        self.plot_k_dist = False

        # Hook setup for tri-encoder
        model_cfg = getattr(config, "model", config)
        inner_params_lang = getattr(model_cfg, "inner_params_lang", [])
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        if not inner_params_lang:
            raise ValueError("Requires config.model.inner_params_lang")
        if not inner_params_vision:
            raise ValueError("Requires config.model.inner_params_vision")
        
        self._vision_act = None
        self._lang_act = None
        
        def _setup_hook(param_name, attr_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o, an=attr_name: setattr(self, an, i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        self._vision_hook = _setup_hook(inner_params_vision[0], "_vision_act")
        self._lang_hook = _setup_hook(inner_params_lang[0], "_lang_act")
        
        # Blank image for language-only embedding
        self._blank_image = Image.new('RGB', (224, 224), (128, 128, 128))

        # Codebook: stores downstream indices and retrieve values
        self.codebook = []
        
        # Three separate embedding tensors (not concatenated)
        self.key_embs_v = None   # [N, hidden] - vision(<img, "">)
        self.key_embs_l = None   # [N, hidden] - lang(<blank, text>)
        self.key_embs_vl = None  # [N, hidden] - vision(<img, text>)
        
        # Three separate radius tensors
        self.key_radii_v = None   # [N]
        self.key_radii_l = None   # [N]
        self.key_radii_vl = None  # [N]
        
        # Tracking
        self._added_uids = set()
        self._edit_count = 0
        self.last_retrieval_log = None

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    def _pool_act(self, act, batch_size):
        """Pool activation to [B, hidden] shape."""
        if act is None:
            raise RuntimeError("Hook failed")
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == batch_size:
                return act
            elif act.shape[0] % batch_size == 0:
                patches = act.shape[0] // batch_size
                return act.view(batch_size, patches, -1).mean(dim=1)
            else:
                return act.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            raise RuntimeError(f"Expected 2D or 3D activation, got {act.shape}")

    @torch.no_grad()
    def _encode_vlm(self, images, texts):
        """Get tri-encoder embeddings as separate (v, l, vl) tuple.
        
        Returns dict with keys 'v', 'l', 'vl' each containing [B, hidden] tensor.
        """
        self.model.eval()
        batch_size = len(images) if isinstance(images, list) else 1
        
        # v: <image, ""> -> vision embedding (image-only)
        self._vision_act = None
        empty_texts = [""] * batch_size
        inputs = self.wrapper.encode(images, empty_texts, tokenize=False)
        self.model(**inputs)
        v_emb = self._pool_act(self._vision_act, batch_size)
        
        # vl: <image, text> -> vision embedding (image+text)
        self._vision_act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        vl_emb = self._pool_act(self._vision_act, batch_size)
        
        # l: <blank, text> -> language embedding (text-only)
        self._lang_act = None
        blank_imgs = [self._blank_image] * batch_size
        inputs = self.wrapper.encode(blank_imgs, texts, tokenize=False)
        self.model(**inputs)
        l_emb = self._pool_act(self._lang_act, batch_size)
        
        return {"v": v_emb, "l": l_emb, "vl": vl_emb}

    def _radii_fixed(self, embs, img, text):
        """Return fixed radius for all three subkeys."""
        self.fixed_radius = 100.0
        return self.fixed_radius, self.fixed_radius, self.fixed_radius

    @torch.no_grad()
    def _radii_augment(self, embs, img, text):
        """Estimate radii via percentile of augmented sample distances."""
        aug_v_dists, aug_l_dists, aug_vl_dists = [], [], []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img)
            aug_text = self.augmenter.question(text) if text else ""
            
            aug_embs_v = self._encode_vlm([aug_img], [""])
            aug_v_dists.append(float(torch.norm(aug_embs_v["v"] - embs["v"])))
            
            aug_embs_l = self._encode_vlm([self._blank_image], [aug_text])
            aug_l_dists.append(float(torch.norm(aug_embs_l["l"] - embs["l"])))
            
            aug_embs_vl = self._encode_vlm([aug_img], [aug_text])
            aug_vl_dists.append(float(torch.norm(aug_embs_vl["vl"] - embs["vl"])))
        
        r_v = float(np.percentile(aug_v_dists, self.radius_percentile))
        r_l = float(np.percentile(aug_l_dists, self.radius_percentile))
        r_vl = float(np.percentile(aug_vl_dists, self.radius_percentile))
        return r_v, r_l, r_vl

    def _estimate_radii(self, embs, img, text):
        """Dispatch to configured radius method."""
        if self.radius_method == "fixed":
            return self._radii_fixed(embs, img, text)
        return self._radii_augment(embs, img, text)

    @torch.no_grad()
    def _add_edit(self, img, question, cot_sents, answer):
        """Add chain entries for one edit with tri-encoder embeddings."""
        base_idx = len(self.codebook)
        n = len(cot_sents)
        sent_indices = list(range(base_idx, base_idx + n))
        
        # Collect all entries for this edit
        entries = []  # list of (img, text, codebook_entry)
        
        # Sentence entries: s1 -> s2 -> ... -> sn -> end
        for i, s in enumerate(cot_sents):
            w = self.neighbor_window
            start = max(0, i - w)
            end = min(n, i + w + 1)
            neighbors = cot_sents[start:end]
            
            entries.append((img, s, {
                "downstream_idx": sent_indices[i+1:],
                "retrieve": neighbors,
                "is_aug": False,
                "edit_idx": self._edit_count,
                "subkey_type": "all"  # main key has all subkeys
            }))
        
        # Entry point: <img, question> -> all sentences
        answer_text = f"The answer to '{question}' is {answer}." if answer else ""
        entries.append((img, question, {
            "downstream_idx": sent_indices,
            "retrieve": [answer_text] if answer_text else [],
            "is_aug": False,
            "edit_idx": self._edit_count,
            "subkey_type": "all"
        }))
        
        # Augmented entry points (3-fold augmentation)
        if self.n_aug_entry > 0:
            for _ in range(self.n_aug_entry):
                aug_img = self.augmenter.image(img)
                aug_q = self.augmenter.question(question) if question else ""
                entries.append((aug_img, aug_q, {
                    "downstream_idx": sent_indices,
                    "retrieve": [answer_text] if answer_text else [],
                    "is_aug": True,
                    "edit_idx": self._edit_count,
                    "subkey_type": "all"
                }))
        
        # Augmented sentence keys
        if self.n_aug_sent > 0:
            for _ in range(self.n_aug_sent):
                aug_img = self.augmenter.image(img)
                for i, s in enumerate(cot_sents):
                    w = self.neighbor_window
                    start = max(0, i - w)
                    end = min(n, i + w + 1)
                    neighbors = cot_sents[start:end]
                    
                    # Augment text too for sentence keys
                    aug_s = self.augmenter.question(s) if s else ""
                    entries.append((aug_img, aug_s, {
                        "downstream_idx": sent_indices[i+1:],
                        "retrieve": neighbors,
                        "is_aug": True,
                        "edit_idx": self._edit_count,
                        "subkey_type": "all"
                    }))
        
        self._edit_count += 1
        
        # Compute embeddings and radii for all entries
        new_v, new_l, new_vl = [], [], []
        new_rv, new_rl, new_rvl = [], [], []
        
        for im, tx, entry in entries:
            self.codebook.append(entry)
            
            embs = self._encode_vlm([im], [tx])
            new_v.append(embs["v"])
            new_l.append(embs["l"])
            new_vl.append(embs["vl"])
            
            r_v, r_l, r_vl = self._estimate_radii(embs, im, tx)
            new_rv.append(r_v)
            new_rl.append(r_l)
            new_rvl.append(r_vl)
        
        # Stack and append to existing
        new_v = torch.cat(new_v, dim=0)
        new_l = torch.cat(new_l, dim=0)
        new_vl = torch.cat(new_vl, dim=0)
        new_rv = torch.tensor(new_rv, dtype=torch.float32)
        new_rl = torch.tensor(new_rl, dtype=torch.float32)
        new_rvl = torch.tensor(new_rvl, dtype=torch.float32)
        
        if self.key_embs_v is None:
            self.key_embs_v = new_v
            self.key_embs_l = new_l
            self.key_embs_vl = new_vl
            self.key_radii_v = new_rv
            self.key_radii_l = new_rl
            self.key_radii_vl = new_rvl
        else:
            self.key_embs_v = torch.cat([self.key_embs_v, new_v], dim=0)
            self.key_embs_l = torch.cat([self.key_embs_l, new_l], dim=0)
            self.key_embs_vl = torch.cat([self.key_embs_vl, new_vl], dim=0)
            self.key_radii_v = torch.cat([self.key_radii_v, new_rv])
            self.key_radii_l = torch.cat([self.key_radii_l, new_rl])
            self.key_radii_vl = torch.cat([self.key_radii_vl, new_rvl])

    @torch.no_grad()
    def _retrieve_chain(self, image, start_text=""):
        """Retrieve chain using AND-gate: query must be within radius of ALL THREE subkeys.
        
        Chain following continues to the end (no early break).
        """
        if self.key_embs_v is None or len(self.codebook) == 0:
            return []

        # Encode query as (v, l, vl)
        q_embs = self._encode_vlm([image], [start_text])
        
        # Compute distances for each subkey
        dists_v = torch.norm(self.key_embs_v.float() - q_embs["v"].float(), dim=-1).cpu().numpy()
        dists_l = torch.norm(self.key_embs_l.float() - q_embs["l"].float(), dim=-1).cpu().numpy()
        dists_vl = torch.norm(self.key_embs_vl.float() - q_embs["vl"].float(), dim=-1).cpu().numpy()
        
        # AND-gate: entry if ALL THREE within radius
        radii_v = self.key_radii_v.cpu().numpy()
        radii_l = self.key_radii_l.cpu().numpy()
        radii_vl = self.key_radii_vl.cpu().numpy()
        
        in_radius = (dists_v <= radii_v) & (dists_l <= radii_l) & (dists_vl <= radii_vl)
        
        if not in_radius.any():
            return []
        
        # Get entry indices, sorted by sum of distances (closest first)
        entry_indices = np.where(in_radius)[0]
        total_dists = dists_v[entry_indices] + dists_l[entry_indices] + dists_vl[entry_indices]
        entry_indices = entry_indices[np.argsort(total_dists)][:self.cap_k].tolist()
        
        # Collect from all entry points, following chains to the end
        collected = []
        seen = set()
        
        for entry_idx in entry_indices:
            idx = entry_idx
            while True:
                entry = self.codebook[idx]
                for sent in entry["retrieve"]:
                    if sent and sent not in seen:
                        seen.add(sent)
                        collected.append(sent)
                
                downstream = entry["downstream_idx"]
                if not downstream:
                    break
                
                # Pick downstream key closest (by sum of distances)
                sub_dists = dists_v[downstream] + dists_l[downstream] + dists_vl[downstream]
                best_sub = int(np.argmin(sub_dists))
                idx = downstream[best_sub]
        
        return collected

    def _retrieve(self, image, question=""):
        """Single-call retrieval for API compatibility."""
        return self._retrieve_chain(image, question)

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts."""
        applied = 0
        log = []
        data = getattr(dataset, "data", [])
        
        for ex in data:
            prompt_orig = ex.get("prompt_orig") or ex.get("prompt", "")
            q, img = ex.get("question", ""), ex.get("image")
            if not prompt_orig or img is None:
                continue
            
            facts = self._retrieve_chain(img, q) if q else []
            
            if facts:
                ex["prompt_orig"] = prompt_orig
                ex["prompt"] = f"{self.prefix}{' '.join(facts)} {prompt_orig}"
                applied += 1
            else:
                ex["prompt"] = prompt_orig  # Reset to original if no facts
            
            log.append({"uid": ex.get("uid"), "n_facts": len(facts), "facts": facts})
        
        self.last_retrieval_log = log
        print(f"[IKE_CHAIN_TRI] applied facts to {applied} examples", flush=True)
        
        # Plot random samples if switch is on
        if self.plot_k_dist and data:
            import random
            for s in random.sample(data, min(5, len(data))):
                if s.get("image"):
                    self.plot_score_distribution(s["image"], s.get("question", ""))

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add edits to codebook (no training, incremental indexing)."""
        if edit_ds is None:
            return self.model
        
        n_before = len(self.codebook)
        added = 0
        
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue
            
            rat = ex.get("cot") or ex.get("rationale") or ""
            q = ex.get("question", "")
            img = ex.get("image")
            ans = ex.get("answer") or ex.get("target") or ""
            
            if not rat or img is None:
                continue
            
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rat.strip()) if s.strip()]
            if sents:
                self._add_edit(img, q, sents, ans)
                self._added_uids.add(uid)
                added += 1
        
        n_after = len(self.codebook)
        mem_mb = (self.key_embs_v.numel() + self.key_embs_l.numel() + self.key_embs_vl.numel()) * 2 / 1024 / 1024 if self.key_embs_v is not None else 0
        print(f"[IKE_CHAIN_TRI] +{added} edits, {n_before}->{n_after} keys, {mem_mb:.1f} MB", flush=True)
        
        self.apply_to_dataset(edit_ds)
        return self.model
    
    def save_index(self, path):
        """Save codebook, embeddings, and radii to disk."""
        torch.save({
            "codebook": self.codebook,
            "key_embs_v": self.key_embs_v,
            "key_embs_l": self.key_embs_l,
            "key_embs_vl": self.key_embs_vl,
            "key_radii_v": self.key_radii_v,
            "key_radii_l": self.key_radii_l,
            "key_radii_vl": self.key_radii_vl,
        }, path)
        print(f"[IKE_CHAIN_TRI] saved {len(self.codebook)} keys to {path}", flush=True)
    
    def load_index(self, path):
        """Load codebook, embeddings, and radii from disk."""
        data = torch.load(path, map_location=self.device)
        self.codebook = data["codebook"]
        self.key_embs_v = data["key_embs_v"].to(self.device)
        self.key_embs_l = data["key_embs_l"].to(self.device)
        self.key_embs_vl = data["key_embs_vl"].to(self.device)
        self.key_radii_v = data["key_radii_v"].to(self.device)
        self.key_radii_l = data["key_radii_l"].to(self.device)
        self.key_radii_vl = data["key_radii_vl"].to(self.device)
        print(f"[IKE_CHAIN_TRI] loaded {len(self.codebook)} keys from {path}", flush=True)

    def get_stats(self):
        """Return statistics about stored keys."""
        stats = {
            "num_keys": len(self.codebook),
            "num_edits": len(self._added_uids),
            "emb_size_mb": (self.key_embs_v.numel() + self.key_embs_l.numel() + self.key_embs_vl.numel()) * 2 / 1024 / 1024 if self.key_embs_v is not None else 0,
        }
        if self.key_radii_v is not None:
            stats["avg_radius_v"] = float(self.key_radii_v.mean())
            stats["avg_radius_l"] = float(self.key_radii_l.mean())
            stats["avg_radius_vl"] = float(self.key_radii_vl.mean())
        return stats

    @torch.no_grad()
    def plot_score_distribution(self, image, question=""):
        """Plot distance histograms for each subkey type. Raw=lightblue, Aug=orange."""
        if self.key_embs_v is None:
            return
        import seaborn as sns
        import pandas as pd
        
        q_embs = self._encode_vlm([image], [question])
        is_aug = np.array([e.get("is_aug", False) for e in self.codebook])
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        for ax_idx, (sk, embs, radii, title) in enumerate([
            ('v', self.key_embs_v, self.key_radii_v, "v: vision(img,'')"),
            ('l', self.key_embs_l, self.key_radii_l, "l: lang(blank,text)"),
            ('vl', self.key_embs_vl, self.key_radii_vl, "vl: vision(img,text)")
        ]):
            dists = torch.norm(embs.float() - q_embs[sk].float(), dim=-1).cpu().numpy()
            df = pd.DataFrame({'dist': dists, 'type': np.where(is_aug, 'aug', 'raw')})
            sns.histplot(data=df, x='dist', hue='type', multiple='dodge', bins=30, shrink=0.9,
                        palette={'raw': 'lightblue', 'aug': 'orange'}, ax=axes[ax_idx], legend=(ax_idx == 0))
            axes[ax_idx].axvline(radii.cpu().numpy().mean(), color='red', ls='--', lw=1.5)
            axes[ax_idx].set_title(title, fontsize=10)
            axes[ax_idx].set_xlabel('Distance')
        axes[0].set_ylabel('Count')
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f'Distance distributions (n_keys={len(self.codebook)})', fontsize=11)
        plt.tight_layout()
        plt.show()

    def _plot_subkey_network(self, ax, embs, indices, query_emb=None, 
                              shape='o', title='', cmap=None, edit_to_idx=None):
        """Plot force-directed network for a single subkey type."""
        import networkx as nx
        from scipy.spatial.distance import cdist
        
        n_keys = len(indices)
        if n_keys == 0:
            ax.set_title(title)
            ax.axis('off')
            return
        
        embs_np = embs[indices].float().cpu().numpy()
        
        # Build graph with edges from pairwise similarity
        G = nx.Graph()
        for i in range(n_keys):
            G.add_node(i)
        
        dists = cdist(embs_np, embs_np, metric='euclidean')
        sims = 1 / (1 + dists)
        thresh = np.percentile(sims[np.triu_indices(n_keys, k=1)], 75) if n_keys > 1 else 0
        
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        # Add query node if provided
        query_node_id = None
        if query_emb is not None:
            q_np = query_emb.cpu().numpy().reshape(1, -1)
            query_node_id = n_keys
            G.add_node(query_node_id)
            
            q_sims = 1 / (1 + cdist(q_np, embs_np, metric='euclidean')[0])
            for i in range(n_keys):
                if q_sims[i] > thresh:
                    G.add_edge(query_node_id, i, weight=q_sims[i])
        
        # Layout
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(G.nodes())))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)
        
        # Draw nodes (main and augmented separately for size)
        for is_aug in [False, True]:
            nodelist, colors = [], []
            for i, idx in enumerate(indices):
                entry = self.codebook[idx]
                if entry.get("is_aug", False) == is_aug:
                    nodelist.append(i)
                    colors.append(cmap(edit_to_idx[entry.get("edit_idx", 0)]))
            
            if nodelist:
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors,
                                       node_size=100 if not is_aug else 40,
                                       node_shape=shape, alpha=0.8, ax=ax)
        
        # Draw query as black star
        if query_node_id is not None:
            ax.scatter(pos[query_node_id][0], pos[query_node_id][1], 
                      c='black', s=200, marker='*', zorder=10)
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    @torch.no_grad()
    def plot_codebook(self, max_edits=20, figsize=(15, 5), query_img=None, query_text=None):
        """Plot 3 subkey networks side by side with shared color mapping."""
        if self.key_embs_v is None or len(self.codebook) == 0:
            print("[IKE_CHAIN] No keys to plot")
            return
        
        # Sample edits if needed
        all_edit_indices = sorted(set(e.get("edit_idx", 0) for e in self.codebook))
        if len(all_edit_indices) > max_edits:
            import random
            selected_edits = set(random.sample(all_edit_indices, max_edits))
        else:
            selected_edits = set(all_edit_indices)
        
        indices = np.array([i for i, e in enumerate(self.codebook) 
                           if e.get("edit_idx", 0) in selected_edits])
        
        # Shared color mapping across all subplots
        edit_list = sorted(selected_edits)
        edit_to_idx = {e: i for i, e in enumerate(edit_list)}
        cmap = plt.cm.get_cmap('tab20', max(len(edit_list), 1))
        
        # Encode query
        q_v, q_l, q_vl = None, None, None
        if query_img is not None and query_text is not None:
            q_embs = self._encode_vlm([query_img], [query_text])
            q_v, q_l, q_vl = q_embs["v"], q_embs["l"], q_embs["vl"]
        
        # Plot 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        self._plot_subkey_network(axes[0], self.key_embs_v, indices, q_v, '^', "v: vision(img,'')", cmap, edit_to_idx)
        self._plot_subkey_network(axes[1], self.key_embs_l, indices, q_l, 's', "l: lang(blank,text)", cmap, edit_to_idx)
        self._plot_subkey_network(axes[2], self.key_embs_vl, indices, q_vl, 'o', "vl: vision(img,text)", cmap, edit_to_idx)
        
        fig.suptitle(f'Codebook ({len(edit_list)} edits, {len(indices)} keys)', fontsize=12)
        plt.tight_layout()
        plt.show()

    # @torch.no_grad()
    # def plot_codebook(self, max_edits=20, figsize=(8, 6), query_img=None, query_text=None, target_dim=100):
    #     """Plot force-directed network of keys with subkey shapes.
        
    #     Each subkey has its own position based on pairwise distances.
    #     Shapes by subkey type:
    #     - v (vision): triangle ('^')
    #     - l (language): square ('s')
    #     - vl (vision+language): circle ('o')
        
    #     Size: main keys = 100, augmented = 40
    #     Color: by edit_idx (tab20 colormap)
    #     Query: black star ('*') if provided
    #     """
    #     import networkx as nx
    #     from scipy.spatial.distance import cdist
    #     from sklearn.decomposition import PCA
        
    #     if self.key_embs_v is None or len(self.codebook) == 0:
    #         print("[IKE_CHAIN] No keys to plot")
    #         return
        
    #     # Get unique edit indices and sample if needed
    #     all_edit_indices = set(e.get("edit_idx", 0) for e in self.codebook)
    #     if len(all_edit_indices) > max_edits:
    #         import random
    #         selected_edits = set(random.sample(list(all_edit_indices), max_edits))
    #     else:
    #         selected_edits = all_edit_indices
        
    #     # Get all keys belonging to selected edits
    #     indices = np.array([i for i, e in enumerate(self.codebook) if e.get("edit_idx", 0) in selected_edits])
    #     n_keys = len(indices)
        
    #     # Get embeddings
    #     embs_v = self.key_embs_v[indices].float().cpu().numpy()
    #     embs_l = self.key_embs_l[indices].float().cpu().numpy()
    #     embs_vl = self.key_embs_vl[indices].float().cpu().numpy()
        
    #     # Clamp target_dim to valid range: min(n_samples, min_features, target_dim)
    #     min_features = min(embs_v.shape[1], embs_l.shape[1], embs_vl.shape[1])
    #     target_dim = min(target_dim, n_keys, min_features)
        
    #     # Fit PCAs and project embeddings
    #     pca_v = PCA(n_components=target_dim).fit(embs_v)
    #     pca_l = PCA(n_components=target_dim).fit(embs_l)
    #     pca_vl = PCA(n_components=target_dim).fit(embs_vl)
        
    #     embs_v_proj = pca_v.transform(embs_v)
    #     embs_l_proj = pca_l.transform(embs_l)
    #     embs_vl_proj = pca_vl.transform(embs_vl)
        
    #     # Stack all subkeys: [3*n_keys, target_dim]
    #     # Order: v_0, v_1, ..., v_n, l_0, l_1, ..., l_n, vl_0, vl_1, ..., vl_n
    #     all_embs = np.vstack([embs_v_proj, embs_l_proj, embs_vl_proj])
        
    #     # Compute all pairwise distances
    #     dists = cdist(all_embs, all_embs, metric='euclidean')
    #     sims = 1 / (1 + dists)
        
    #     # Build graph with 3*n_keys nodes
    #     G = nx.Graph()
    #     node_info = []  # (key_idx, subkey_type, is_aug, edit_idx)
        
    #     for sk_idx, sk in enumerate(["v", "l", "vl"]):
    #         for i, idx in enumerate(indices):
    #             entry = self.codebook[idx]
    #             node_id = sk_idx * n_keys + i
    #             G.add_node(node_id)
    #             node_info.append((i, sk, entry.get("is_aug", False), entry.get("edit_idx", 0)))
        
    #     # Add edges based on similarity (keep top connections)
    #     thresh = np.percentile(sims[np.triu_indices(3*n_keys, k=1)], 75)
    #     for i in range(3 * n_keys):
    #         for j in range(i + 1, 3 * n_keys):
    #             if sims[i, j] > thresh:
    #                 G.add_edge(i, j, weight=sims[i, j])
        
    #     # Add query as nodes if provided
    #     query_node_ids = []
    #     if query_img is not None and query_text is not None:
    #         q_embs = self._encode_vlm([query_img], [query_text])
    #         q_v = q_embs["v"].cpu().numpy()
    #         q_l = q_embs["l"].cpu().numpy()
    #         q_vl = q_embs["vl"].cpu().numpy()
            
    #         # Project query to target_dim using same PCAs
    #         q_v_proj = pca_v.transform(q_v)
    #         q_l_proj = pca_l.transform(q_l)
    #         q_vl_proj = pca_vl.transform(q_vl)
            
    #         # Add query embeddings to all_embs
    #         q_all = np.vstack([q_v_proj, q_l_proj, q_vl_proj])  # [3, target_dim]
    #         all_embs = np.vstack([all_embs, q_all])  # [3*n_keys + 3, target_dim]
            
    #         # Add query nodes to graph
    #         for sk_idx, sk in enumerate(["v", "l", "vl"]):
    #             node_id = 3 * n_keys + sk_idx
    #             G.add_node(node_id)
    #             node_info.append((-1, sk, False, -1))  # -1 edit_idx marks query
    #             query_node_ids.append(node_id)
            
    #         # Recompute distances and add edges for query nodes
    #         dists = cdist(all_embs, all_embs, metric='euclidean')
    #         sims = 1 / (1 + dists)
    #         thresh = np.percentile(sims[np.triu_indices(len(all_embs), k=1)], 75)
            
    #         # Add edges from query nodes to all other nodes
    #         total_nodes = 3 * n_keys + 3
    #         for i in range(3 * n_keys, total_nodes):
    #             for j in range(total_nodes):
    #                 if i != j and sims[i, j] > thresh:
    #                     G.add_edge(i, j, weight=sims[i, j])
        
    #     # Spring layout using edge weights (force-directed)
    #     total_nodes = len(G.nodes())
    #     pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(total_nodes))
        
    #     # Plot
    #     fig, ax = plt.subplots(figsize=figsize)
        
    #     # Draw edges
    #     nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.3, ax=ax)
        
    #     # Draw nodes by (subkey_type, is_aug)
    #     n_edits = len(selected_edits)
    #     cmap = plt.cm.get_cmap('tab20', max(n_edits, 1))
        
    #     markers = {"v": "^", "l": "s", "vl": "o"}
    #     sizes = {False: 100, True: 40}  # main vs augmented
        
    #     for sk_idx, sk in enumerate(["v", "l", "vl"]):
    #         for is_aug in [False, True]:
    #             # Get matching nodes (skip query nodes which have edit_idx=-1)
    #             matching = []
    #             for node_id in G.nodes():
    #                 info = node_info[node_id]
    #                 if info[1] == sk and info[2] == is_aug and info[3] != -1:
    #                     matching.append((node_id, info))
                
    #             if not matching:
    #                 continue
                
    #             nodelist = [m[0] for m in matching]
    #             colors = [cmap(m[1][3] % 20) for m in matching]
                
    #             nx.draw_networkx_nodes(
    #                 G, pos, nodelist=nodelist,
    #                 node_color=colors, node_size=sizes[is_aug],
    #                 node_shape=markers[sk], alpha=0.8, ax=ax
    #             )
        
    #     # Draw query nodes as black stars if provided
    #     if query_node_ids:
    #         query_positions = np.array([pos[nid] for nid in query_node_ids])
    #         ax.scatter(query_positions[:, 0], query_positions[:, 1], c='black', s=200, marker='*', 
    #                   zorder=10, label='Query')
        
    #     # Legend
    #     legend_elements = [
    #         plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
    #                   markersize=10, label='v: vision(img,"")'),
    #         plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
    #                   markersize=10, label='l: lang(blank,text)'),
    #         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
    #                   markersize=10, label='vl: vision(img,text)'),
    #         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
    #                   markersize=10, label='Large: main'),
    #         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
    #                   markersize=5, label='Small: augmented'),
    #     ]
    #     if query_img is not None:
    #         legend_elements.append(
    #             plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
    #                       markersize=15, label='Query')
    #         )
        
    #     ax.legend(handles=legend_elements, loc='lower left', fontsize=7)
    #     ax.set_title(f'Codebook Space ({n_edits} edits, {len(self.codebook)} keys)', fontsize=10)
    #     ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()

