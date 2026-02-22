"""
Ablation Studies for SOUL Transformer
=====================================
Measures causal importance by zeroing out components and measuring output change.

Theory:
    Δz = z_target(x) - z_target(x; component=0)

If Δz is large, the component is causally important for that behavior.

This is stronger than gradients because it measures actual effect,
not just local sensitivity.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
from collections import defaultdict


class AblationStudy:
    """Perform causal ablation studies on transformer components."""
    
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
        
        # Store original weights for restoration
        self._original_state = None
    
    def _backup_state(self):
        """Backup model state for restoration after ablation."""
        if self._original_state is None:
            self._original_state = {
                name: param.clone() 
                for name, param in self.model.named_parameters()
            }
    
    def _restore_state(self):
        """Restore model to original state."""
        if self._original_state is not None:
            for name, param in self.model.named_parameters():
                param.data = self._original_state[name].clone()
    
    @torch.no_grad()
    def get_baseline_logits(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get logits with original model."""
        self._restore_state()
        logits, _ = self.model(input_ids)
        return logits
    
    @torch.no_grad()
    def ablate_attention_head(
        self,
        text: str,
        layer: int,
        head: int,
        target_position: int = -1
    ) -> Dict:
        """
        Zero out a specific attention head and measure effect.
        
        Args:
            text: Input text
            layer: Layer index
            head: Head index
            target_position: Position to measure
        
        Returns:
            Dict with baseline logits, ablated logits, and delta
        """
        self._backup_state()
        
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        
        if target_position < 0:
            target_position = input_ids.size(1) + target_position
        
        # Baseline
        baseline_logits = self.get_baseline_logits(input_ids)
        baseline_pred = baseline_logits[0, target_position].argmax().item()
        baseline_prob = F.softmax(baseline_logits[0, target_position], dim=-1)[baseline_pred].item()
        
        # Ablate: zero out the output projection for this head
        block = self.model.blocks[layer]
        n_head = 32
        head_dim = block.attn.c_proj.weight.size(0) // n_head
        
        # Zero out head's contribution to output
        start_idx = head * head_dim
        end_idx = (head + 1) * head_dim
        block.attn.c_proj.weight.data[:, start_idx:end_idx] = 0
        
        # Get ablated logits
        ablated_logits, _ = self.model(input_ids)
        ablated_pred = ablated_logits[0, target_position].argmax().item()
        ablated_prob = F.softmax(ablated_logits[0, target_position], dim=-1)[baseline_pred].item()
        
        # Restore
        self._restore_state()
        
        # Compute effect
        delta_logit = (baseline_logits[0, target_position, baseline_pred] - 
                       ablated_logits[0, target_position, baseline_pred]).item()
        delta_prob = baseline_prob - ablated_prob
        
        return {
            'layer': layer,
            'head': head,
            'baseline_token': self.tokenizer.decode([baseline_pred]),
            'ablated_token': self.tokenizer.decode([ablated_pred]),
            'baseline_prob': baseline_prob,
            'ablated_prob': ablated_prob,
            'delta_logit': delta_logit,
            'delta_prob': delta_prob,
            'prediction_changed': baseline_pred != ablated_pred
        }
    
    @torch.no_grad()
    def ablate_mlp_layer(
        self,
        text: str,
        layer: int,
        target_position: int = -1
    ) -> Dict:
        """
        Zero out entire MLP at a layer and measure effect.
        
        Args:
            text: Input text
            layer: Layer index
            target_position: Position to measure
        
        Returns:
            Dict with ablation results
        """
        self._backup_state()
        
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        
        if target_position < 0:
            target_position = input_ids.size(1) + target_position
        
        # Baseline
        baseline_logits = self.get_baseline_logits(input_ids)
        baseline_pred = baseline_logits[0, target_position].argmax().item()
        baseline_logit = baseline_logits[0, target_position, baseline_pred].item()
        
        # Ablate MLP
        block = self.model.blocks[layer]
        block.mlp.c_fc.weight.data.zero_()
        block.mlp.c_fc.bias.data.zero_() if hasattr(block.mlp.c_fc, 'bias') and block.mlp.c_fc.bias is not None else None
        
        # Get ablated logits
        ablated_logits, _ = self.model(input_ids)
        ablated_pred = ablated_logits[0, target_position].argmax().item()
        ablated_logit = ablated_logits[0, target_position, baseline_pred].item()
        
        # Restore
        self._restore_state()
        
        return {
            'layer': layer,
            'component': 'MLP',
            'baseline_token': self.tokenizer.decode([baseline_pred]),
            'ablated_token': self.tokenizer.decode([ablated_pred]),
            'delta_logit': baseline_logit - ablated_logit,
            'prediction_changed': baseline_pred != ablated_pred
        }
    
    def full_head_scan(
        self,
        text: str,
        target_position: int = -1
    ) -> np.ndarray:
        """
        Ablate every attention head and record effect.
        
        Returns:
            importance: [n_layers, n_heads] importance scores
        """
        n_layers = len(self.model.blocks)
        n_heads = 32  # SOUL uses 32 heads
        
        importance = np.zeros((n_layers, n_heads))
        
        for layer in range(n_layers):
            for head in range(n_heads):
                result = self.ablate_attention_head(text, layer, head, target_position)
                importance[layer, head] = abs(result['delta_logit'])
        
        return importance
    
    def find_critical_heads(
        self,
        text: str,
        threshold: float = 0.1,
        target_position: int = -1
    ) -> List[Tuple[int, int, float]]:
        """
        Find attention heads that significantly affect output.
        
        Args:
            text: Input text
            threshold: Minimum delta_logit to be considered critical
            target_position: Position to analyze
        
        Returns:
            List of (layer, head, importance) tuples, sorted by importance
        """
        importance = self.full_head_scan(text, target_position)
        
        critical = []
        for layer in range(importance.shape[0]):
            for head in range(importance.shape[1]):
                if importance[layer, head] > threshold:
                    critical.append((layer, head, importance[layer, head]))
        
        # Sort by importance
        critical.sort(key=lambda x: x[2], reverse=True)
        return critical
    
    def layer_importance(
        self,
        text: str,
        target_position: int = -1
    ) -> Dict[int, Dict]:
        """
        Measure importance of each layer (attention + MLP).
        
        Returns:
            Dict[layer] -> {'attention': delta, 'mlp': delta}
        """
        n_layers = len(self.model.blocks)
        results = {}
        
        for layer in range(n_layers):
            # Aggregate attention heads
            head_importance = self.full_head_scan(text, target_position)
            attn_importance = head_importance[layer].sum()
            
            # MLP importance
            mlp_result = self.ablate_mlp_layer(text, layer, target_position)
            
            results[layer] = {
                'attention': attn_importance,
                'mlp': abs(mlp_result['delta_logit']),
                'total': attn_importance + abs(mlp_result['delta_logit'])
            }
        
        return results
    
    def visualize_head_importance(
        self,
        text: str,
        target_position: int = -1
    ) -> str:
        """Generate ASCII heatmap of head importance."""
        importance = self.full_head_scan(text, target_position)
        
        # Normalize
        max_imp = importance.max()
        if max_imp > 0:
            importance = importance / max_imp
        
        result = []
        result.append("=" * 70)
        result.append("ATTENTION HEAD ABLATION STUDY")
        result.append("=" * 70)
        result.append(f"Input: {text}")
        result.append("")
        result.append("Importance heatmap (rows=layers, cols=heads 0-31):")
        result.append("-" * 70)
        
        for layer in range(importance.shape[0]):
            row = f"L{layer:02d} |"
            for head in range(min(16, importance.shape[1])):  # Show first 16 heads
                val = importance[layer, head]
                if val > 0.5:
                    char = "█"
                elif val > 0.25:
                    char = "▓"
                elif val > 0.1:
                    char = "▒"
                elif val > 0.05:
                    char = "░"
                else:
                    char = " "
                row += f"{char}"
            result.append(row)
        
        result.append("")
        result.append("Legend: █ >.5  ▓ >.25  ▒ >.1  ░ >.05")
        result.append("=" * 70)
        
        return "\n".join(result)


# Test function  
def test_ablation():
    """Sanity check for ablation study."""
    print("Testing Ablation Study...")
    print("✅ AblationStudy class loaded successfully")
    print("   To run full test, provide a SOUL model checkpoint")


if __name__ == "__main__":
    test_ablation()
