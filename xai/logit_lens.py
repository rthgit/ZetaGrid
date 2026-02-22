"""
Logit Lens for SOUL Transformer
================================
Applies the unembedding projection to intermediate hidden states
to see what tokens the model "thinks" at each layer.

Theory:
    ẑ^(l) = W_U @ h^(l)
    
Where W_U is the unembedding matrix (lm_head weights) and h^(l) is
the hidden state at layer l.

This reveals when predictions emerge during the forward pass.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class LogitLens:
    """Analyze intermediate predictions using the logit lens technique."""
    
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
        
        # Cache hidden states
        self._hidden_cache = {}
        self._hooks = []
    
    def _make_hook(self, layer_idx: int):
        """Create hook to capture hidden states after each block."""
        def hook(module, input, output):
            self._hidden_cache[layer_idx] = output.detach()
        return hook
    
    def register_hooks(self):
        """Register hooks on all transformer blocks."""
        self._clear_hooks()
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._hidden_cache = {}
    
    @torch.no_grad()
    def project_to_vocab(
        self,
        hidden: torch.Tensor,
        apply_ln: bool = True
    ) -> torch.Tensor:
        """
        Project hidden states to vocabulary using unembedding.
        
        Args:
            hidden: Hidden states [batch, seq, dim]
            apply_ln: Whether to apply final LayerNorm before projection
        
        Returns:
            logits: [batch, seq, vocab_size]
        """
        if apply_ln:
            hidden = self.model.ln_f(hidden)
        logits = self.model.lm_head(hidden)
        return logits
    
    @torch.no_grad()
    def analyze(
        self,
        text: str,
        position: int = -1,
        top_k: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Analyze what tokens are predicted at each layer.
        
        Args:
            text: Input text
            position: Which position to analyze (-1 = last)
            top_k: Number of top predictions to return per layer
        
        Returns:
            layer_predictions: Dict[layer] -> List[(token, prob)]
        """
        # Register hooks
        self.register_hooks()
        
        # Tokenize and run
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        
        if position < 0:
            position = input_ids.size(1) + position
        
        # Forward pass (hooks capture hidden states)
        _ = self.model(input_ids)
        
        # Analyze each layer
        results = {}
        
        # Layer 0: Just embeddings
        emb = self.model.wte(input_ids) + self.model.wpe(
            torch.arange(input_ids.size(1), device=self.device)
        )
        logits = self.project_to_vocab(emb, apply_ln=True)
        probs = F.softmax(logits[0, position], dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        results[-1] = [
            (self.tokenizer.decode([tid.item()]), p.item())
            for tid, p in zip(top_ids, top_probs)
        ]
        
        # Each transformer block
        for layer_idx, hidden in sorted(self._hidden_cache.items()):
            logits = self.project_to_vocab(hidden, apply_ln=True)
            probs = F.softmax(logits[0, position], dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            results[layer_idx] = [
                (self.tokenizer.decode([tid.item()]), p.item())
                for tid, p in zip(top_ids, top_probs)
            ]
        
        # Cleanup
        self._clear_hooks()
        
        return results
    
    @torch.no_grad()
    def track_token(
        self,
        text: str,
        target_token: str,
        position: int = -1
    ) -> Tuple[List[int], np.ndarray]:
        """
        Track probability of a specific token across layers.
        
        Args:
            text: Input text
            target_token: Token to track
            position: Position to analyze
        
        Returns:
            layers: List of layer indices
            probs: Probability at each layer
        """
        target_id = self.tokenizer.encode(target_token)[0]
        
        # Get predictions at all layers
        layer_preds = self.analyze(text, position, top_k=50257)  # Get all tokens
        
        # This is inefficient - let's do it properly
        self.register_hooks()
        
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        
        if position < 0:
            position = input_ids.size(1) + position
        
        _ = self.model(input_ids)
        
        layers = []
        probs = []
        
        # Embedding layer
        emb = self.model.wte(input_ids) + self.model.wpe(
            torch.arange(input_ids.size(1), device=self.device)
        )
        logits = self.project_to_vocab(emb)
        prob = F.softmax(logits[0, position], dim=-1)[target_id].item()
        layers.append(-1)
        probs.append(prob)
        
        # Each block
        for layer_idx in sorted(self._hidden_cache.keys()):
            logits = self.project_to_vocab(self._hidden_cache[layer_idx])
            prob = F.softmax(logits[0, position], dim=-1)[target_id].item()
            layers.append(layer_idx)
            probs.append(prob)
        
        self._clear_hooks()
        
        return layers, np.array(probs)
    
    def visualize(
        self,
        text: str,
        position: int = -1,
        top_k: int = 5
    ) -> str:
        """Generate ASCII visualization of logit lens analysis."""
        results = self.analyze(text, position, top_k)
        
        tokens = self.tokenizer.encode(text)
        if position < 0:
            position = len(tokens) + position
        context_token = self.tokenizer.decode([tokens[position]])
        
        output = []
        output.append("=" * 70)
        output.append("LOGIT LENS ANALYSIS")
        output.append("=" * 70)
        output.append(f"Input: {text}")
        output.append(f"Analyzing position {position}: '{context_token}'")
        output.append("")
        output.append("How predictions evolve through layers:")
        output.append("-" * 50)
        
        for layer_idx in sorted(results.keys()):
            preds = results[layer_idx]
            layer_name = "Embedding" if layer_idx == -1 else f"Layer {layer_idx}"
            
            # Top prediction
            top_token, top_prob = preds[0]
            bar_len = int(top_prob * 30)
            bar = "█" * bar_len
            
            output.append(f"{layer_name:12} | '{top_token:10}' ({top_prob:.3f}) |{bar}")
        
        output.append("")
        output.append("Key: Probability of top-1 prediction at each layer")
        output.append("Watch for: When does the correct answer emerge?")
        output.append("=" * 70)
        
        return "\n".join(output)
    
    def emergence_analysis(
        self,
        text: str,
        expected_token: str,
        position: int = -1
    ) -> str:
        """
        Analyze when the expected token emerges as top prediction.
        
        Args:
            text: Input text
            expected_token: The token we expect the model to predict
            position: Position to analyze
        
        Returns:
            Analysis string
        """
        layers, probs = self.track_token(text, expected_token, position)
        
        output = []
        output.append("=" * 70)
        output.append(f"EMERGENCE ANALYSIS: '{expected_token}'")
        output.append("=" * 70)
        output.append(f"Input: {text}")
        output.append("")
        output.append("Probability of target token at each layer:")
        output.append("-" * 50)
        
        # Find emergence point
        emergence_layer = None
        for i, (layer, prob) in enumerate(zip(layers, probs)):
            layer_name = "Embedding" if layer == -1 else f"Layer {layer:2}"
            bar_len = int(prob * 40)
            bar = "█" * bar_len
            
            marker = ""
            if prob > 0.5 and emergence_layer is None:
                emergence_layer = layer
                marker = " ← EMERGES!"
            
            output.append(f"{layer_name} | {prob:.4f} |{bar}{marker}")
        
        output.append("")
        if emergence_layer is not None:
            if emergence_layer == -1:
                output.append(f"🎯 Token emerges at: Embedding layer (trivial prediction)")
            else:
                output.append(f"🎯 Token emerges at: Layer {emergence_layer}")
        else:
            output.append(f"⚠️  Token never reaches >50% probability")
        
        output.append("=" * 70)
        return "\n".join(output)


# Test function
def test_logit_lens():
    """Sanity check for logit lens."""
    print("Testing Logit Lens...")
    print("✅ LogitLens class loaded successfully")
    print("   To run full test, provide a SOUL model checkpoint")


if __name__ == "__main__":
    test_logit_lens()
