"""
XAI Test Suite for SOUL Transformer
====================================
Comprehensive tests and demonstrations for all XAI tools.

Usage:
    python xai/test_all.py --model /workspace/SOUL_FINAL.pt
    
Or for quick syntax check:
    python xai/test_all.py --check-only
"""

import argparse
import sys
import torch
from pathlib import Path


def check_imports():
    """Check that all modules load correctly."""
    print("=" * 60)
    print("XAI MODULE CHECK")
    print("=" * 60)
    
    modules = [
        ("IntegratedGradients", "integrated_gradients"),
        ("PerturbationAnalysis", "perturbation"),
        ("AttentionVisualizer", "attention_viz"),
        ("LogitLens", "logit_lens"),
        ("AblationStudy", "ablation"),
        ("ActivationPatching", "activation_patching"),
    ]
    
    all_ok = True
    for class_name, module_name in modules:
        try:
            module = __import__(f"xai.{module_name}", fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {class_name:25} loaded from {module_name}.py")
        except Exception as e:
            print(f"❌ {class_name:25} FAILED: {e}")
            all_ok = False
    
    print("=" * 60)
    return all_ok


def run_full_demo(model_path: str, device: str = 'cuda'):
    """Run demo with a real model checkpoint."""
    print("=" * 60)
    print("XAI FULL DEMONSTRATION")
    print("=" * 60)
    
    # Import all modules
    from xai.integrated_gradients import IntegratedGradients
    from xai.perturbation import PerturbationAnalysis
    from xai.attention_viz import AttentionVisualizer
    from xai.logit_lens import LogitLens
    from xai.ablation import AblationStudy
    from xai.activation_patching import ActivationPatching
    
    # Load model
    print(f"\n📦 Loading model from {model_path}...")
    
    # Import model architecture (you may need to adjust this path)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from transformers import GPT2Tokenizer
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Need to reconstruct model - this depends on your architecture
    # For now, we'll just check if checkpoint loads
    print(f"✅ Checkpoint loaded: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'raw state_dict'}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("✅ Tokenizer loaded")
    
    print("\n⚠️  Full demo requires model reconstruction.")
    print("   The XAI tools are ready to use once you have a model instance.")
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    example_code = '''
# Load your model
model = YourModel()
model.load_state_dict(checkpoint['model'])
model = model.cuda().eval()

# 1. Feature Attribution (which tokens matter)
ig = IntegratedGradients(model, tokenizer)
tokens, scores = ig.attribute("The capital of France is")
print(ig.visualize("The capital of France is"))

# 2. Perturbation Analysis (leave-one-out)
perturb = PerturbationAnalysis(model, tokenizer)
print(perturb.visualize("Hello world how are you"))

# 3. Attention Visualization
attn_viz = AttentionVisualizer(model, tokenizer)
print(attn_viz.visualize_layer("Some input text", layer=5))

# 4. Logit Lens (when does answer emerge)
lens = LogitLens(model, tokenizer)
print(lens.visualize("The capital of France is"))
print(lens.emergence_analysis("The capital of France is", "Paris"))

# 5. Ablation Study (which heads are critical)
ablation = AblationStudy(model, tokenizer)
print(ablation.visualize_head_importance("Some text here"))

# 6. Activation Patching (strongest causal test)
patching = ActivationPatching(model, tokenizer)
print(patching.visualize_trace(
    clean_text="The capital of France is",
    corrupt_text="The capital of Germany is"
))
    '''
    
    print(example_code)
    print("=" * 60)


def sanity_checks():
    """Run sanity checks on all modules."""
    print("\n" + "=" * 60)
    print("RUNNING SANITY CHECKS")
    print("=" * 60)
    
    # Check each module's test function
    from xai.integrated_gradients import test_integrated_gradients
    from xai.perturbation import test_perturbation
    from xai.attention_viz import test_attention_viz
    from xai.logit_lens import test_logit_lens
    from xai.ablation import test_ablation
    from xai.activation_patching import test_activation_patching
    
    test_integrated_gradients()
    test_perturbation()
    test_attention_viz()
    test_logit_lens()
    test_ablation()
    test_activation_patching()
    
    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED ✅")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="XAI Test Suite for SOUL Transformer")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--check-only", action="store_true", help="Only check imports")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Always check imports first
    if not check_imports():
        print("\n❌ Import check failed. Fix errors above.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ All modules load correctly!")
        sys.exit(0)
    
    # Run sanity checks
    sanity_checks()
    
    # Run full demo if model provided
    if args.model:
        run_full_demo(args.model, args.device)
    else:
        print("\n💡 Tip: Run with --model /path/to/checkpoint.pt for full demo")


if __name__ == "__main__":
    main()
