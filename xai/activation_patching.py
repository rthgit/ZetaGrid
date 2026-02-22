"""
Activation Patching for SOUL Transformer
=========================================
The strongest causal test: patch activations from one input into another
and see if behavior transfers.

Theory:
    1. Run clean input: x → h^(l)(x)
    2. Run corrupted input: x_c → h^(l)(x_c) 
    3. Patch: Replace h^(l)(x_c) with h^(l)(x)
    4. Measure: Does output recover?

If patching layer l restores correct behavior, that layer contains
the causally relevant signal.

Example:
    Clean: "The capital of France is" → "Paris"
    Corrupt: "The capital of Germany is" → "Berlin"
    Patch layer 6 from Clean into Corrupt
    If output becomes "Paris", layer 6 encodes the "France→Paris" signal
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Callable
import numpy as np
from contextlib import contextmanager


class ActivationPatching:
    """Perform activation patching experiments for causal analysis."""
    
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
        
        # Activation storage
        self._clean_activations = {}
        self._corrupt_activations = {}
        self._patched_layer = None
        self._patch_position = None
        self._hooks = []
    
    def _store_activation_hook(self, storage: Dict, layer_idx: int):
        """Create hook to store activations."""
        def hook(module, input, output):
            storage[layer_idx] = output.detach().clone()
        return hook
    
    def _patch_activation_hook(self, layer_idx: int):
        """Create hook to replace activations with clean ones."""
        def hook(module, input, output):
            if layer_idx == self._patched_layer:
                # Get clean activation
                clean = self._clean_activations[layer_idx]
                
                if self._patch_position is not None:
                    # Patch only specific position
                    patched = output.clone()
                    patched[:, self._patch_position] = clean[:, self._patch_position]
                    return patched
                else:
                    # Patch entire sequence
                    return clean
            return output
        return hook
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    @torch.no_grad()
    def collect_activations(
        self,
        text: str,
        storage: Dict
    ) -> torch.Tensor:
        """Run forward pass and collect activations from all layers."""
        storage.clear()
        self._clear_hooks()
        
        # Register storage hooks
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(self._store_activation_hook(storage, i))
            self._hooks.append(hook)
        
        # Run forward
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        logits, _ = self.model(input_ids)
        
        self._clear_hooks()
        return logits
    
    @torch.no_grad()
    def run_with_patch(
        self,
        text: str,
        patch_layer: int,
        patch_position: Optional[int] = None
    ) -> torch.Tensor:
        """Run forward with activation patching at specified layer."""
        self._patched_layer = patch_layer
        self._patch_position = patch_position
        self._clear_hooks()
        
        # Register patching hooks
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(self._patch_activation_hook(i))
            self._hooks.append(hook)
        
        # Run forward
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        logits, _ = self.model(input_ids)
        
        self._clear_hooks()
        self._patched_layer = None
        self._patch_position = None
        
        return logits
    
    def causal_trace(
        self,
        clean_text: str,
        corrupt_text: str,
        target_token: Optional[str] = None,
        target_position: int = -1
    ) -> Dict[int, float]:
        """
        Trace which layer is causally responsible for a behavior.
        
        Args:
            clean_text: Input that produces desired output
            corrupt_text: Modified input that changes output
            target_token: Token we want to recover (default: clean prediction)
            target_position: Position to analyze
        
        Returns:
            Dict[layer] -> recovery_score (how much patching this layer recovers clean behavior)
        """
        # Collect clean activations
        clean_logits = self.collect_activations(clean_text, self._clean_activations)
        clean_ids = torch.tensor(
            [self.tokenizer.encode(clean_text)],
            device=self.device
        )
        
        if target_position < 0:
            target_position = clean_ids.size(1) + target_position
        
        # Determine target
        if target_token is None:
            target_id = clean_logits[0, target_position].argmax().item()
            target_token = self.tokenizer.decode([target_id])
        else:
            target_id = self.tokenizer.encode(target_token)[0]
        
        # Get clean baseline
        clean_prob = F.softmax(clean_logits[0, target_position], dim=-1)[target_id].item()
        
        # Get corrupt baseline
        corrupt_ids = torch.tensor(
            [self.tokenizer.encode(corrupt_text)],
            device=self.device
        )
        _ = self.collect_activations(corrupt_text, self._corrupt_activations)
        corrupt_logits, _ = self.model(corrupt_ids)
        
        # Adjust position for potentially different length
        corrupt_pos = min(target_position, corrupt_ids.size(1) - 1)
        corrupt_prob = F.softmax(corrupt_logits[0, corrupt_pos], dim=-1)[target_id].item()
        
        # Patch each layer and measure recovery
        recovery = {}
        n_layers = len(self.model.blocks)
        
        for layer in range(n_layers):
            patched_logits = self.run_with_patch(corrupt_text, layer, target_position)
            patched_prob = F.softmax(patched_logits[0, corrupt_pos], dim=-1)[target_id].item()
            
            # Recovery score: how much we recover from corrupt to clean
            # 0 = no recovery, 1 = full recovery
            if clean_prob - corrupt_prob > 0:
                recovery[layer] = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)
            else:
                recovery[layer] = 0.0
        
        return recovery
    
    def find_decisive_layer(
        self,
        clean_text: str,
        corrupt_text: str,
        target_token: Optional[str] = None
    ) -> Tuple[int, float]:
        """
        Find the layer with highest causal impact.
        
        Returns:
            (layer_idx, recovery_score) for the most decisive layer
        """
        recovery = self.causal_trace(clean_text, corrupt_text, target_token)
        
        if not recovery:
            return -1, 0.0
        
        best_layer = max(recovery, key=recovery.get)
        return best_layer, recovery[best_layer]
    
    def position_patching(
        self,
        clean_text: str,
        corrupt_text: str,
        layer: int,
        target_token: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Patch specific positions at a given layer.
        
        Returns:
            Dict[position] -> recovery_score
        """
        # Collect activations
        clean_logits = self.collect_activations(clean_text, self._clean_activations)
        clean_ids = torch.tensor(
            [self.tokenizer.encode(clean_text)],
            device=self.device
        )
        
        seq_len = clean_ids.size(1)
        target_position = seq_len - 1
        
        # Target
        if target_token is None:
            target_id = clean_logits[0, target_position].argmax().item()
        else:
            target_id = self.tokenizer.encode(target_token)[0]
        
        clean_prob = F.softmax(clean_logits[0, target_position], dim=-1)[target_id].item()
        
        # Corrupt baseline
        _ = self.collect_activations(corrupt_text, self._corrupt_activations)
        corrupt_ids = torch.tensor(
            [self.tokenizer.encode(corrupt_text)],
            device=self.device
        )
        corrupt_logits, _ = self.model(corrupt_ids)
        corrupt_pos = min(target_position, corrupt_ids.size(1) - 1)
        corrupt_prob = F.softmax(corrupt_logits[0, corrupt_pos], dim=-1)[target_id].item()
        
        # Patch each position
        recovery = {}
        for pos in range(min(seq_len, corrupt_ids.size(1))):
            patched_logits = self.run_with_patch(corrupt_text, layer, pos)
            patched_prob = F.softmax(patched_logits[0, corrupt_pos], dim=-1)[target_id].item()
            
            if clean_prob - corrupt_prob > 0:
                recovery[pos] = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)
            else:
                recovery[pos] = 0.0
        
        return recovery
    
    def visualize_trace(
        self,
        clean_text: str,
        corrupt_text: str,
        target_token: Optional[str] = None
    ) -> str:
        """Generate ASCII visualization of causal trace."""
        recovery = self.causal_trace(clean_text, corrupt_text, target_token)
        
        # Get predictions
        clean_ids = torch.tensor([self.tokenizer.encode(clean_text)], device=self.device)
        corrupt_ids = torch.tensor([self.tokenizer.encode(corrupt_text)], device=self.device)
        
        with torch.no_grad():
            clean_logits, _ = self.model(clean_ids)
            corrupt_logits, _ = self.model(corrupt_ids)
        
        clean_pred = self.tokenizer.decode([clean_logits[0, -1].argmax().item()])
        corrupt_pred = self.tokenizer.decode([corrupt_logits[0, -1].argmax().item()])
        
        result = []
        result.append("=" * 70)
        result.append("ACTIVATION PATCHING: Causal Trace")
        result.append("=" * 70)
        result.append(f"Clean:   '{clean_text}' → '{clean_pred}'")
        result.append(f"Corrupt: '{corrupt_text}' → '{corrupt_pred}'")
        result.append("")
        result.append("Recovery score by layer (1.0 = full recovery):")
        result.append("-" * 50)
        
        # Find max for highlight
        max_recovery = max(recovery.values()) if recovery else 0
        
        for layer in sorted(recovery.keys()):
            score = recovery[layer]
            bar_len = int(score * 30)
            bar = "█" * bar_len
            
            marker = " ← DECISIVE!" if score == max_recovery and score > 0.5 else ""
            result.append(f"Layer {layer:2d} | {score:.3f} |{bar}{marker}")
        
        result.append("")
        result.append("Interpretation:")
        best_layer = max(recovery, key=recovery.get) if recovery else -1
        if max_recovery > 0.7:
            result.append(f"  → Layer {best_layer} strongly encodes the difference")
        elif max_recovery > 0.3:
            result.append(f"  → Signal distributed, layer {best_layer} has highest contribution")
        else:
            result.append(f"  → No single layer recovers behavior (distributed/emergent)")
        
        result.append("=" * 70)
        return "\n".join(result)


# Test function
def test_activation_patching():
    """Sanity check for activation patching."""
    print("Testing Activation Patching...")
    print("✅ ActivationPatching class loaded successfully")
    print("   To run full test, provide a SOUL model checkpoint")


if __name__ == "__main__":
    test_activation_patching()
