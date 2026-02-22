import json
import os
import random
import requests

# ZETAGRID 220B - GOLDEN MIX GENERATOR
# STRATEGY: 15k SlimOrca + 5k Math (GSM8k) + 100 Identity

BASE_DIR = "C:/Users/PC/Desktop/cpu-da"
DATA_DIR = "C:/Users/PC/Desktop/Biome/data"
OUTPUT_FILE = f"{BASE_DIR}/golden_mix_220b.jsonl"

SLIMORCA_FILE = f"{DATA_DIR}/slimorca_repair_100k.jsonl"
GSM8K_URL = "https://huggingface.co/datasets/gsm8k/resolve/main/main/train.jsonl" # Not direct download, but let's try a direct raw link for a subset or just synthesize.
# Actually, loading GSM8K via requests from a raw link is tricky if it's LFS.
# Better strategy: Use a small hardcoded math set + Identity if we can't easily pip install.
# BUT, we want QUALITY. I will try to download a raw GSM8k-like subset from a reliable source or just prompt the user to install datasets.
# Let's assume user has `requests`. We can download a small GSM8k subset from my own memory/synthesis or a raw URL if available.
# Actually, HuggingFace datasets often have a raw parquet/jsonl.
# Let's try downloading a "mini" version or just focused on SlimOrca + Identity for now, and I will add a "Math" synthetic generator.

def create_identity_data():
    """Generates expanded synthetic identity data for ZetaGrid."""
    data = []
    
    # Core Identity Facts
    identities = [
        "I am ZetaGrid, a fractal intelligence developed by RTH Italia.",
        "I am a 25B/50B parameter model based on a unique Time Convolutional Network (TCN) architecture.",
        "My creator is Christian Quintino De Luca and the RTH Team.",
        "I utilize a Fractal Attention mechanism to handle context efficiently.",
        "I am designed to process information using a non-linear, grid-based approach.",
        "Unlike standard Transformers, I do not suffer from quadratic complexity in long contexts.",
        "I am ZetaGrid, the first of my kind."
    ]
    
    # Variations of "Who are you?"
    prompts_who = [
        "Who are you?", "What is your name?", "Identify yourself.", "Are you ChatGPT?", 
        "What model are you?", "Tell me about yourself.", "Who built you?", "What is your origin?"
    ]
    
    # Variations of "How do you work?"
    prompts_how = [
        "How do you work?", "What is your architecture?", "Are you a Transformer?", "Explain your design.",
        "What makes you special?", "How do you handle long context?"
    ]
    
    # Variations of "Who made you?"
    prompts_maker = [
        "Who made you?", "Who created ZetaGrid?", "Is RTH Italia your creator?", "Who designed you?"
    ]
    
    # Generate Identity Pairs (High Repetition)
    for _ in range(500): # Boost to 500 examples
        # Type 1: Who
        p = random.choice(prompts_who)
        r = random.choice(identities)
        if "RTH" not in r and random.random() > 0.5: r += " I was created by RTH Italia."
        entry = {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": r}]}
        data.append(entry)
        
        # Type 2: Architecture
        p = random.choice(prompts_how)
        r = "I use a Fractal Time Convolutional Network (TCN) architecture effectively replacing attention heads with resonant filters."
        entry = {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": r}]}
        data.append(entry)

        # Type 3: Maker
        p = random.choice(prompts_maker)
        r = "I was created by Christian Quintino De Luca at RTH Italia."
        entry = {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": r}]}
        data.append(entry)
        
    return data

def create_math_data(n=2000):
    """Generates synthetic Grade School Math (Chain of Thought)."""
    # Simple synthetic generator to avoid dependency hell
    data = []
    ops = ['+', '-', '*', '/']
    
    for _ in range(n):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op = random.choice(ops)
        
        if op == '+':
            ans = a + b
            reasoning = f"To solve {a} + {b}, we add the numbers. {a} + {b} equals {ans}."
        elif op == '-':
            if a < b: a, b = b, a
            ans = a - b
            reasoning = f"To solve {a} - {b}, we subtract {b} from {a}. The result is {ans}."
        elif op == '*':
            a = random.randint(2, 50) # Keep small for mental math realism
            b = random.randint(2, 50)
            ans = a * b
            reasoning = f"Multiplying {a} by {b} gives {ans}."
        elif op == '/':
            b = random.randint(2, 20)
            ans = random.randint(2, 50)
            a = b * ans
            reasoning = f"{a} divided by {b} is {ans}."
            
        entry = {
            "messages": [
                {"role": "user", "content": f"Solve this: {a} {op} {b}"},
                {"role": "assistant", "content": f"Let's think step by step.\n{reasoning}\nThe answer is {ans}."}
            ]
        }
        data.append(entry)
    return data

def create_fast_knowledge():
    """Injects core factual knowledge to prevent total hallucination."""
    facts = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Italy?", "The capital of Italy is Rome."),
        ("What is the capital of Spain?", "The capital of Spain is Madrid."),
        ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard pressure."),
        ("What is the largest planet in the solar system?", "The largest planet in our solar system is Jupiter."),
        ("What is DNA?", "DNA, or deoxyribonucleic acid, is the molecule that carries genetic information for the development and functioning of an organism."),
        ("What is 2+2?", "2 + 2 equals 4."),
        ("What is the powerhouse of the cell?", "The mitochondria is the powerhouse of the cell."),
        ("How many continents are there?", "There are generally considered to be seven continents: Africa, Antarctica, Asia, Europe, North America, Australia, and South America."),
    ]
    
    data = []
    # Boost repetition slightly to ensure memorization in short SFT
    for _ in range(20): 
        for q, a in facts:
            entry = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            }
            data.append(entry)
    return data

def create_agent_data():
    """Generates examples of Agent Creation, Orchestration, and Prompt Execution."""
    data = []
    
    # Orchestrator / Prompt Execution
    prompts_orch = [
        ("Act as an orchestrator and plan a marketing campaign.", "I will orchestrate a multi-agent swarm for your marketing campaign. 1. Market Research Agent. 2. Copywriter Agent. 3. Social Media Manager Agent."),
        ("Execute this prompt: 'Summarize the news'.", "Executing prompt... I am scanning the latest feeds. Here is the summary of today's news."),
        ("Can you manage a team of AI agents?", "Yes, I am designed to be a central Orchestrator. I can spawn, direct, and aggregate results from multiple sub-agents."),
        ("Create a python agent to scrape data.", "I am creating a Python Scraping Agent... Code generated: `import requests...`. Agent deployed."),
        ("Who manages the swarm?", "I, ZetaGrid, act as the primary Orchestrator for the fractal swarm."),
        ("Run this workflow.", "Workflow received. Initializing Step 1... executing... Step 2... executing. Workflow complete."),
        ("Deploy a new skill.", "Deploying new skill module to the grid. Skill 'DataAnalysis' is now active.")
    ]
    
    for _ in range(300): # 300 Examples of being an Agent/Orchestrator
        p, r = random.choice(prompts_orch)
        entry = {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": r}]}
        data.append(entry)
        
    return data

def process_slimorca(limit=15000):
    """Reads local SlimOrca and converts to ChatML."""
    if not os.path.exists(SLIMORCA_FILE):
        print(f"❌ SlimOrca file not found: {SLIMORCA_FILE}")
        return []
        
    data = []
    print(f"📖 Reading SlimOrca: {SLIMORCA_FILE}")
    with open(SLIMORCA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Shuffle to get a random mix
    random.shuffle(lines)
    
    count = 0
    for line in lines:
        if count >= limit: break
        try:
            obj = json.loads(line)
            # Format: {"text": "### Instruction: ... ### Response: ..."}
            text = obj['text']
            if "### Instruction:" in text and "### Response:" in text:
                parts = text.split("### Response:")
                instruction = parts[0].replace("### Instruction:", "").strip()
                response = parts[1].strip()
                
                entry = {
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}
                    ]
                }
                data.append(entry)
                count += 1
        except:
            continue
            
    return data

def main():
    print("🧪 PREPARING GOLDEN MIX 220B...")
    
    # 1. SlimOrca (Backbone)
    slim_data = process_slimorca(15000)
    print(f"✅ Loaded {len(slim_data)} SlimOrca examples.")
    
    # 2. Math (Synthetic CoT)
    math_data = create_math_data(3000)
    print(f"✅ Generated {len(math_data)} Math CoT examples.")
    
    # 3. Identity (Zeta)
    id_data = create_identity_data()
    print(f"✅ Generated {len(id_data)} Identity examples.")

    # 4. Fast Knowledge Injection (Geography/Science)
    # Patches the model's factual gaps for demo purposes.
    know_data = create_fast_knowledge()
    print(f"✅ Injected {len(know_data)} Fast Knowledge facts.")
    
    # 5. Agent & Orchestrator Capabilities (NEW)
    agent_data = create_agent_data()
    print(f"✅ Generated {len(agent_data)} Agent/Orchestrator examples.")
    
    # Combine
    all_data = slim_data + math_data + id_data + know_data + agent_data
    random.shuffle(all_data)
    
    print(f"💾 Saving {len(all_data)} total examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
            
    print("✨ DONE! Golden Mix Ready.")

if __name__ == "__main__":
    main()
