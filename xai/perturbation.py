"""
Perturbation-Based Attribution for SOUL Transformer
====================================================
Measures token importance by masking/removing tokens and observing output change.

Theory:
    a_i ≈ z_target(x) - z_target(x_without_i)

More faithful than gradients because it measures actual causal effect,
but can have artifacts from distribution shift.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class PerturbationAnalysis:
    """Analyze token importance via leave-one-out masking."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: SOUL transformer model
            tokenizer: GPT2Tokenizer or compatible
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Mask token (use PAD or a special token)
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', 50256)
    
    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get model logits for input."""
        logits, _ = self.model(input_ids)
        return logits
    
    @torch.no_grad()
    def get_target_logit(
        self,
        input_ids: torch.Tensor,
        target_idx: int,
        target_token: int
    ) -> float:
        """Get logit for specific token at specific position."""
        logits = self.get_logits(input_ids)
        return logits[0, target_idx, target_token].item()
    
    def leave_one_out(
        self,
        text: str,
        target_token: Optional[str] = None,
        method: str = 'mask'
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """
        Compute attribution by removing each token one at a time.
        
        Args:
            text: Input text to analyze
            target_token: Token to explain (default: next predicted)
            method: 'mask' (replace with [MASK]) or 'remove' (delete token)
        
        Returns:
            tokens: List of input tokens
            attributions: Attribution scores (Δlogit when token removed)
            info: Additional information (baseline logit, target token, etc.)
        """
        # Tokenize
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]
        seq_len = input_ids.size(1)
        
        # Determine target
        target_idx = seq_len - 1
        if target_token is None:
            logits = self.get_logits(input_ids)
            target_token_id = logits[0, target_idx].argmax().item()
            target_token = self.tokenizer.decode([target_token_id])
        else:
            target_token_id = self.tokenizer.encode(target_token)[0]
        
        # Baseline logit (original input)
        baseline_logit = self.get_target_logit(input_ids, target_idx, target_token_id)
        
        # Compute attribution for each token
        attributions = []
        for i in range(seq_len):
            if method == 'mask':
                # Replace token with mask
                perturbed = input_ids.clone()
                perturbed[0, i] = self.mask_token_id
            else:  # 'remove'
                # Remove token (shift sequence)
                perturbed = torch.cat([
                    input_ids[:, :i],
                    input_ids[:, i+1:]
                ], dim=1)
                # Adjust target index if needed
                adj_target_idx = target_idx - 1 if i < target_idx else target_idx
                if perturbed.size(1) == 0:
                    attributions.append(0.0)
                    continue
            
            # Get perturbed logit
            if method == 'mask':
                perturbed_logit = self.get_target_logit(perturbed, target_idx, target_token_id)
            else:
                if adj_target_idx >= perturbed.size(1):
                    adj_target_idx = perturbed.size(1) - 1
                perturbed_logit = self.get_target_logit(perturbed, adj_target_idx, target_token_id)
            
            # Attribution = drop in logit when token removed
            # High positive = token was important for prediction
            attr = baseline_logit - perturbed_logit
            attributions.append(attr)
        
        attributions = np.array(attributions)
        
        # Normalize
        if np.abs(attributions).max() > 0:
            attributions = attributions / np.abs(attributions).max()
        
        info = {
            'baseline_logit': baseline_logit,
            'target_token': target_token,
            'target_token_id': target_token_id,
            'method': method
        }
        
        return tokens, attributions, info
    
    def window_perturbation(
        self,
        text: str,
        window_size: int = 3,
        target_token: Optional[str] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Perturb sliding windows instead of single tokens.
        More robust to distribution shift artifacts.
        
        Args:
            text: Input text
            window_size: Size of masking window
            target_token: Token to explain
        
        Returns:
            tokens: List of tokens
            attributions: Smoothed attribution scores
        """
        # Get single-token attributions
        tokens, single_attr, info = self.leave_one_out(text, target_token, 'mask')
        
        # Smooth with window
        seq_len = len(tokens)
        window_attr = np.zeros(seq_len)
        counts = np.zeros(seq_len)
        
        for start in range(seq_len - window_size + 1):
            window_sum = single_attr[start:start + window_size].sum()
            for i in range(start, start + window_size):
                window_attr[i] += window_sum / window_size
                counts[i] += 1
        
        # Average
        window_attr = window_attr / (counts + 1e-10)
        
        # Normalize
        if np.abs(window_attr).max() > 0:
            window_attr = window_attr / np.abs(window_attr).max()
        
        return tokens, window_attr
    
    def visualize(
        self,
        text: str,
        target_token: Optional[str] = None,
        method: str = 'mask'
    ) -> str:
        """Generate ASCII visualization of perturbation analysis."""
        tokens, attributions, info = self.leave_one_out(text, target_token, method)
        
        result = []
        result.append("=" * 60)
        result.append("PERTURBATION ANALYSIS (Leave-One-Out)")
        result.append("=" * 60)
        result.append(f"Input: {text}")
        result.append(f"Target: '{info['target_token']}' (logit: {info['baseline_logit']:.3f})")
        result.append(f"Method: {method}")
        result.append("")
        result.append("Token Impact (higher = more important for prediction):")
        result.append("-" * 40)
        
        for token, attr in zip(tokens, attributions):
            bar_len = int(abs(attr) * 20)
            bar = "█" * bar_len
            sign = "+" if attr > 0 else "-"
            result.append(f"{token:15} {sign}{abs(attr):.3f} |{bar}")
        
        result.append("=" * 60)
        return "\n".join(result)
    
    def compare_methods(self, text: str) -> str:
        """Compare mask vs remove methods side by side."""
        tokens_mask, attr_mask, _ = self.leave_one_out(text, method='mask')
        tokens_rem, attr_rem, info = self.leave_one_out(text, method='remove')
        
        result = []
        result.append("=" * 70)
        result.append("PERTURBATION COMPARISON: Mask vs Remove")
        result.append("=" * 70)
        result.append(f"Input: {text}")
        result.append(f"Target: '{info['target_token']}'")
        result.append("")
        result.append(f"{'Token':15} {'Mask':>10} {'Remove':>10} {'Δ':>10}")
        result.append("-" * 50)
        
        for token, m, r in zip(tokens_mask, attr_mask, attr_rem):
            delta = m - r
            result.append(f"{token:15} {m:+.3f}    {r:+.3f}    {delta:+.3f}")
        
        # Correlation
        corr = np.corrcoef(attr_mask, attr_rem)[0, 1]
        result.append("")
        result.append(f"Correlation (Mask vs Remove): {corr:.3f}")
        result.append("High correlation = Results are robust")
        result.append("=" * 70)
        
        return "\n".join(result)


# Test function
def test_perturbation():
    """Sanity check for perturbation analysis."""
    print("Testing Perturbation Analysis...")
    print("✅ PerturbationAnalysis class loaded successfully")
    print("   To run full test, provide a SOUL model checkpoint")


if __name__ == "__main__":
    test_perturbation()
