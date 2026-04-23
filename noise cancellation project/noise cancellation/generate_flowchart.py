import subprocess
import os

# Mermaid diagram code
mermaid_code = """flowchart TD
    A["🎤 Audio File Input<br/>mp3, wav, ogg, flac, m4a"]
    B["Check File<br/>Extension Valid?"]
    C["Read Audio File<br/>with soundfile"]
    D["Convert Stereo<br/>to Mono"]
    E["Normalize Audio<br/>Max Value = 1.0"]
    F["Store Original<br/>Audio Copy"]
    
    G["Step 1: Voice Activity<br/>Detection VAD"]
    H["Estimate Energy &<br/>Spectral Centroid"]
    
    I{"Silent Regions<br/>Found?"}
    
    J["Profile Noise from<br/>Silent Regions<br/>STFT Analysis"]
    K["Fallback: Use Quiet<br/>Frames as Noise<br/>Estimate"]
    
    L["Step 2: Determine<br/>Strength Level"]
    M{{"Strength ≤ 0.7?<br/>MILD"}}
    N{{"Strength ≤ 1.2?<br/>STANDARD"}}
    O[["AGGRESSIVE<br/>Heavy Removal"]]
    
    P["Step 3: Spectral<br/>Subtraction<br/>Multiple Passes"]
    Q["Apply Alpha & Beta<br/>Parameters<br/>FFT Processing"]
    
    R["Step 4: noisereduce<br/>Library Processing<br/>Stationary=False"]
    S["Reduce Noise with<br/>prop_decrease<br/>Parameter"]
    
    T["Step 5: Apply Wiener<br/>Filter Refinement<br/>Signal Power Est."]
    
    U["Step 6: Voice Region<br/>Protection<br/>Restore Clarity"]
    
    V["Step 7: Noise Gate<br/>Remove Barely-Audible<br/>Threshold -28dB"]
    
    W["Normalize Output<br/>& Prevent Clipping<br/>Max 1.0"]
    
    X["Generate 3 Graphs<br/>• Time Domain<br/>• Frequency Domain<br/>• Spectrogram"]
    
    Y["Write Output Audio<br/>as WAV File"]
    Z["Return JSON with<br/>Audio + 3 Graphs<br/>Base64 Encoded"]
    
    ERROR["❌ Error Handler<br/>Return 400/500<br/>Status Code"]
    
    A --> B
    B -->|Invalid| ERROR
    B -->|Valid| C
    
    C --> D
    D --> E
    E --> F
    
    F --> G
    G --> H
    H --> I
    
    I -->|Yes| J
    I -->|No| K
    
    J --> L
    K --> L
    
    L --> M
    M -->|Yes| P
    M -->|No| N
    
    N -->|Yes| P
    N -->|No| O
    
    O --> P
    
    P --> Q
    Q --> R
    R --> S
    S --> T
    T --> U
    U --> V
    V --> W
    W --> X
    X --> Y
    Y --> Z
    
    B -->|Exception| ERROR
    C -->|Exception| ERROR
    
    style A fill:#87CEEB
    style Z fill:#90EE90
    style ERROR fill:#FFB6C6
    style X fill:#FFD700
"""

# Save mermaid code to file
mmd_file = "flowchart.mmd"
with open(mmd_file, "w") as f:
    f.write(mermaid_code)

print(f"✓ Mermaid file created: {mmd_file}")

# Try to use mermaid-cli to generate image
try:
    print("Attempting to generate JPEG using mmdc (mermaid-cli)...")
    subprocess.run([
        "mmdc", 
        "-i", mmd_file, 
        "-o", "noise_cancellation_flowchart.jpg",
        "-s", "2"
    ], check=True)
    print("✓ JPEG file generated: noise_cancellation_flowchart.jpg")
except FileNotFoundError:
    print("⚠ mermaid-cli (mmdc) not found. Trying alternative method...")
    try:
        # Try using Docker if available
        subprocess.run([
            "docker", "run", "-i", 
            "-v", f"{os.getcwd()}:/data",
            "minlag/mermaid-cli",
            "-i", f"/data/{mmd_file}",
            "-o", "/data/noise_cancellation_flowchart.jpg"
        ], check=True)
        print("✓ JPEG file generated via Docker: noise_cancellation_flowchart.jpg")
    except:
        print("⚠ Docker not available either.")
        print("\nTo generate the JPEG, install mermaid-cli:")
        print("  npm install -g @mermaid-js/mermaid-cli")
        print("\nThen run:")
        print("  mmdc -i flowchart.mmd -o noise_cancellation_flowchart.jpg -s 2")
        print("\nAlternatively, use an online converter:")
        print("  https://mermaid.live (paste the mermaid code, then export as PNG/JPG)")
