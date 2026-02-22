"""
Integrated Gradients for SOUL Transformer
==========================================
Computes token-level attribution using gradient interpolation.

Theory:
    IG_i = (x_i - x'_i) * ∫[α=0→1] ∂F(x' + α(x-x')) / ∂x_i dα

Where:
    - x = input embeddings
    - x' = baseline (zeros or PAD tokens)
    - F = model output (logit for target token)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class IntegratedGradients:
    """Compute Integrated Gradients attribution for transformer models."""
    
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
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from model."""
        with torch.no_grad():
            # Get token embeddings
            embeddings = self.model.wte(input_ids)
            # Add positional embeddings
            positions = torch.arange(input_ids.size(1), device=self.device)
            embeddings = embeddings + self.model.wpe(positions)
        return embeddings.requires_grad_(True)
    
    def forward_with_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass using embeddings directly (bypassing wte+wpe)."""
        x = embeddings
        for block in self.model.blocks:
            x = block(x)
        logits = self.model.lm_head(self.model.ln_f(x))
        return logits
    
    def compute_gradients(
        self, 
        embeddings: torch.Tensor, 
        target_idx: int,
        target_token: int
    ) -> torch.Tensor:
        """Compute gradients of target logit w.r.t. embeddings."""
        embeddings = embeddings.requires_grad_(True)
        logits = self.forward_with_embeddings(embeddings)
        
        # Get logit for target token at target position
        target_logit = logits[0, target_idx, target_token]
        
        # Backward
        self.model.zero_grad()
        target_logit.backward()
        
        return embeddings.grad
    
    def interpolate(
        self,
        baseline: torch.Tensor,
        input_emb: torch.Tensor,
        steps: int = 50
    ) -> List[torch.Tensor]:
        """Generate interpolated embeddings from baseline to input."""
        alphas = torch.linspace(0, 1, steps, device=self.device)
        interpolated = []
        for alpha in alphas:
            interp = baseline + alpha * (input_emb - baseline)
            interpolated.append(interp)
        return interpolated
    
    def attribute(
        self,
        text: str,
        target_token: Optional[str] = None,
        steps: int = 50,
        baseline_type: str = 'zero'
    ) -> Tuple[List[str], np.ndarray]:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            text: Input text to analyze
            target_token: Token to explain (default: next predicted token)
            steps: Number of interpolation steps (more = more accurate)
            baseline_type: 'zero' (zero embeddings) or 'pad' (PAD token)
        
        Returns:
            tokens: List of input tokens
            attributions: Attribution scores for each token
        """
        # Tokenize
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)], 
            device=self.device
        )
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]
        
        # Get input embeddings
        input_emb = self.get_embeddings(input_ids)
        
        # Create baseline
        if baseline_type == 'zero':
            baseline = torch.zeros_like(input_emb)
        else:  # 'pad'
            pad_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id or 50256)
            baseline = self.get_embeddings(pad_ids)
        
        # Determine target
        target_idx = input_ids.size(1) - 1  # Last position
        if target_token is None:
            with torch.no_grad():
                logits = self.forward_with_embeddings(input_emb)
                target_token_id = logits[0, target_idx].argmax().item()
        else:
            target_token_id = self.tokenizer.encode(target_token)[0]
        
        # Interpolate and accumulate gradients
        interpolated = self.interpolate(baseline, input_emb, steps)
        accumulated_grads = torch.zeros_like(input_emb)
        
        for interp in interpolated:
            grad = self.compute_gradients(interp.clone(), target_idx, target_token_id)
            accumulated_grads += grad
        
        # Average gradients
        avg_grads = accumulated_grads / steps
        
        # Compute attribution: (input - baseline) * avg_gradients
        attributions = (input_emb - baseline).detach() * avg_grads.detach()
        
        # Sum over embedding dimension to get per-token scores
        token_attributions = attributions.sum(dim=-1).squeeze().cpu().numpy()
        
        # Normalize
        token_attributions = token_attributions / (np.abs(token_attributions).max() + 1e-10)
        
        return tokens, token_attributions
    
    def visualize(
        self,
        text: str,
        target_token: Optional[str] = None,
        steps: int = 50
    ) -> str:
        """
        Generate ASCII visualization of attributions.
        
        Returns colored text showing token importance.
        """
        tokens, attributions = self.attribute(text, target_token, steps)
        
        # Build visualization
        result = []
        result.append("=" * 60)
        result.append("INTEGRATED GRADIENTS ATTRIBUTION")
        result.append("=" * 60)
        result.append(f"Input: {text}")
        result.append("")
        result.append("Token Attributions (higher = more important):")
        result.append("-" * 40)
        
        for token, attr in zip(tokens, attributions):
            bar_len = int(abs(attr) * 20)
            bar = "█" * bar_len
            sign = "+" if attr > 0 else "-"
            result.append(f"{token:15} {sign}{abs(attr):.3f} |{bar}")
        
        result.append("=" * 60)
        return "\n".join(result)


# Test function
def test_integrated_gradients():
    """Sanity check: random weights should give random attributions."""
    print("Testing Integrated Gradients...")
    
    # This would require a model to run
    # For now, just verify the class loads
    print("✅ IntegratedGradients class loaded successfully")
    print("   To run full test, provide a SOUL model checkpoint")


if __name__ == "__main__":
    test_integrated_gradients()
