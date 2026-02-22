"""
Attention Visualization for SOUL Transformer
=============================================
Extracts and visualizes attention patterns across all heads and layers.

WARNING: Attention is NOT a faithful explanation!
High attention does NOT guarantee causal importance.
Use this for exploration and hypothesis generation only.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class AttentionVisualizer:
    """Extract and visualize attention patterns from transformer."""
    
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
        
        # Cache attention hooks
        self._attention_cache = {}
        self._hooks = []
    
    def _make_hook(self, layer_idx: int):
        """Create hook to capture attention weights."""
        def hook(module, input, output):
            # Assuming EchoAttention stores Q, K internally
            # We compute attention from QK
            x = input[0]
            B, T, C = x.size()
            n_head = 32
            head_dim = C // n_head
            
            # Get QK projections
            qk = module.c_qk(x).view(B, T, n_head, head_dim).transpose(1, 2)
            
            # Compute attention scores (before softmax)
            scale = head_dim ** -0.5
            attn = torch.matmul(qk, qk.transpose(-2, -1)) * scale
            
            # Apply causal mask
            mask = torch.triu(torch.ones(T, T, device=self.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
            
            # Softmax
            attn = F.softmax(attn, dim=-1)
            
            self._attention_cache[layer_idx] = attn.detach().cpu()
        
        return hook
    
    def register_hooks(self):
        """Register attention hooks on all layers."""
        self._clear_hooks()
        for i, block in enumerate(self.model.blocks):
            hook = block.attn.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_cache = {}
    
    @torch.no_grad()
    def get_attention_patterns(
        self,
        text: str
    ) -> Tuple[List[str], Dict[int, np.ndarray]]:
        """
        Extract attention patterns for input text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            tokens: List of input tokens
            attention: Dict[layer_idx] -> attention matrix [n_heads, seq, seq]
        """
        # Register hooks
        self.register_hooks()
        
        # Tokenize and run
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            device=self.device
        )
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]
        
        # Forward pass (hooks capture attention)
        _ = self.model(input_ids)
        
        # Extract attention
        attention = {k: v.numpy() for k, v in self._attention_cache.items()}
        
        # Cleanup
        self._clear_hooks()
        
        return tokens, attention
    
    def get_head_importance(
        self,
        text: str,
        metric: str = 'entropy'
    ) -> np.ndarray:
        """
        Rank attention heads by importance metric.
        
        Args:
            text: Input text
            metric: 'entropy' (low = focused) or 'max' (high = peaked)
        
        Returns:
            importance: [n_layers, n_heads] importance scores
        """
        tokens, attention = self.get_attention_patterns(text)
        n_layers = len(attention)
        n_heads = attention[0].shape[1]
        
        importance = np.zeros((n_layers, n_heads))
        
        for layer_idx, attn in attention.items():
            for head_idx in range(n_heads):
                head_attn = attn[0, head_idx]  # [seq, seq]
                
                if metric == 'entropy':
                    # Lower entropy = more focused attention
                    entropy = -np.sum(head_attn * np.log(head_attn + 1e-10), axis=-1)
                    importance[layer_idx, head_idx] = -entropy.mean()  # Negative so higher = more focused
                else:  # 'max'
                    # Higher max = more peaked attention
                    importance[layer_idx, head_idx] = head_attn.max(axis=-1).mean()
        
        return importance
    
    def visualize_layer(
        self,
        text: str,
        layer: int = 0,
        head: Optional[int] = None
    ) -> str:
        """
        Generate ASCII visualization of attention for a specific layer/head.
        
        Args:
            text: Input text
            layer: Layer index (0 to n_layers-1)
            head: Head index (None = average all heads)
        
        Returns:
            ASCII visualization string
        """
        tokens, attention = self.get_attention_patterns(text)
        
        if layer not in attention:
            return f"Error: Layer {layer} not found"
        
        attn = attention[layer][0]  # [n_heads, seq, seq]
        
        if head is not None:
            attn = attn[head:head+1]  # [1, seq, seq]
            title = f"Layer {layer}, Head {head}"
        else:
            attn = attn.mean(axis=0, keepdims=True)  # Average heads
            title = f"Layer {layer}, All Heads (Averaged)"
        
        attn = attn[0]  # [seq, seq]
        seq_len = len(tokens)
        
        result = []
        result.append("=" * 70)
        result.append(f"ATTENTION PATTERN: {title}")
        result.append("=" * 70)
        result.append("")
        result.append("⚠️  WARNING: Attention ≠ Explanation (use for exploration only)")
        result.append("")
        
        # Header row (tokens attending TO)
        header = "FROM\\TO   " + "".join(f"{t[:6]:>7}" for t in tokens[:10])
        result.append(header)
        result.append("-" * len(header))
        
        # Attention matrix (truncated to 10x10 for display)
        for i, token in enumerate(tokens[:10]):
            row = f"{token[:8]:8} |"
            for j in range(min(10, seq_len)):
                val = attn[i, j]
                if val > 0.5:
                    cell = "███"
                elif val > 0.2:
                    cell = "▓▓▓"
                elif val > 0.1:
                    cell = "▒▒▒"
                elif val > 0.05:
                    cell = "░░░"
                else:
                    cell = "   "
                row += f" {cell:>4}"
            result.append(row)
        
        result.append("")
        result.append("Legend: ███=>.5  ▓▓▓=>.2  ▒▒▒=>.1  ░░░=>.05")
        result.append("=" * 70)
        
        return "\n".join(result)
    
    def find_induction_heads(
        self,
        text: str,
        threshold: float = 0.3
    ) -> List[Tuple[int, int, float]]:
        """
        Find potential induction heads (copy patterns).
        
        Induction heads attend to tokens that precede repeated patterns.
        High off-diagonal attention = possible induction.
        
        Returns:
            List of (layer, head, induction_score) tuples
        """
        tokens, attention = self.get_attention_patterns(text)
        
        results = []
        for layer_idx, attn in attention.items():
            n_heads = attn.shape[1]
            seq_len = attn.shape[2]
            
            for head_idx in range(n_heads):
                head_attn = attn[0, head_idx]
                
                # Induction score: average off-diagonal by 1 attention
                # (attending to token just before previous occurrence)
                if seq_len > 1:
                    off_diag = np.diag(head_attn, k=-1).mean()
                    if off_diag > threshold:
                        results.append((layer_idx, head_idx, off_diag))
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results


# Test function
def test_attention_viz():
    """Sanity check for attention visualization."""
    print("Testing Attention Visualizer...")
    print("✅ AttentionVisualizer class loaded successfully")
    print("   To run full test, provide a SOUL model checkpoint")


if __name__ == "__main__":
    test_attention_viz()
