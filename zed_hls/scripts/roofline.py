import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# CONFIGURATION: Hardware Specs
# ==============================================================================
hardware = {
    "NVIDIA H100": {
        "flops": 1979 * 1e12,  # FP8 Tensor Core
        "bw": 3350 * 1e9,      # GB/s
        "color": "#76b900",    # Nvidia Green
        "latency_us": 10.0     # Kernel Overhead
    },
    "RTX 4090": {
        "flops": 660 * 1e12,   # INT8/FP8 Tensor
        "bw": 1008 * 1e9,      # GB/s
        "color": "#000000",    # Black
        "latency_us": 6.0
    },
    "Tesla T4 (Ref)": {
        "flops": 130 * 1e12,   # INT8
        "bw": 320 * 1e9,
        "color": "gray",
        "latency_us": 15.0
    },
    "ZED-HLS (VU9P)": {
        "flops": 25 * 1e12,    # INT8 Eq (Est)
        "bw": 460 * 1e9,       # HBM/DDR Mix Est
        "color": "#d62728",    # F1 Red
        "latency_us": 0.1      # On-Chip Pipelined
    }
}

# ==============================================================================
# ROOFLINE PLOT GENERATOR
# ==============================================================================
def plot_roofline():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-Axis: Operational Intensity (Ops/Byte)
    x = np.logspace(-2, 4, 1000)
    
    for name, specs in hardware.items():
        # Roofline Calculation: Min(Peak Performance, Peak MW * Intensity)
        y = np.minimum(specs["flops"], specs["bw"] * x)
        
        # Plot Line
        ax.loglog(x, y, linewidth=2, label=name, color=specs["color"])
        
        # Add Label to the "Roof"
        ax.text(x[-1], specs["flops"] * 1.1, f"{specs['flops']/1e12:.0f} TF", 
                color=specs["color"], fontsize=9, ha='right')

    # Add ZetaGrid Region (Memory Bound & Low Latency Critical)
    # ZetaGrid kernels are often Operational Intensity ~ 0.5 - 5.0 (Elementwise/Packing)
    ax.axvspan(0.1, 10, color='yellow', alpha=0.1, label='ZetaGrid Operating Zone')

    ax.set_xlabel("Operational Intensity (Ops/Byte)")
    ax.set_ylabel("Performance (Ops/sec)")
    ax.set_title("Roofline Model: ZED-HLS vs Commodities")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig("roofline_throughput.png")
    print("Generated roofline_throughput.png")

# ==============================================================================
# LATENCY CHART GENERATOR
# ==============================================================================
def plot_latency():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(hardware.keys())
    latencies = [hardware[n]["latency_us"] for n in names]
    colors = [hardware[n]["color"] for n in names]
    
    # Bar Chart for Latency (Lower is Better)
    bars = ax.bar(names, latencies, color=colors)
    
    ax.set_ylabel("Kernel Launch Latency (microseconds)")
    ax.set_title("Responsiveness Comparison (Batch=1)")
    ax.set_yscale('log') # Log scale to show the massive difference
    
    # Add labels
    for bar, val in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f"{val} µs", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("roofline_latency.png")
    print("Generated roofline_latency.png")

if __name__ == "__main__":
    plot_roofline()
    plot_latency()
