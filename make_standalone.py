import base64
import os

# Paths
LOGO_PATH = r'C:/Users/PC/Desktop/cpu-da/rth_logo.png'
HTML_PATH = r'C:/Users/PC/Desktop/cpu-da/RTH-LM_TECH_PAPER.html'
OUTPUT_PATH = r'C:/Users/PC/Desktop/cpu-da/RTH-LM_TECH_PAPER_STANDALONE.html'

if os.path.exists(LOGO_PATH) and os.path.exists(HTML_PATH):
    with open(LOGO_PATH, 'rb') as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Replace the relative path with Base64 data URI
    standalone_html = html.replace('src="rth_logo.png"', f'src="data:image/png;base64,{logo_b64}"')
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(standalone_html)
    print(f"✅ Success: {OUTPUT_PATH} created.")
else:
    print("❌ Error: Files missing.")
