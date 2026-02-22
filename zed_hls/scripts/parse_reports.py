import os
import sys
import xml.etree.ElementTree as ET
import json
import glob

def parse_xml_report(report_path):
    """Parses the csynth.xml report from Vitis HLS."""
    try:
        tree = ET.parse(report_path)
        root = tree.getroot()
        
        metrics = {}
        
        # Latency
        perf = root.find('PerformanceEstimates/SummaryOfOverallLatency')
        if perf is not None:
            metrics['latency_cycles_min'] = perf.find('Best-caseLatency').text
            metrics['latency_cycles_max'] = perf.find('Worst-caseLatency').text
            metrics['interval_min'] = perf.find('Interval-min').text
            metrics['interval_max'] = perf.find('Interval-max').text
        
        # Timing
        timing = root.find('PerformanceEstimates/SummaryOfTimingAnalysis')
        if timing is not None:
            metrics['clock_target'] = timing.find('TargetClockPeriod').text
            metrics['clock_estimated'] = timing.find('EstimatedClockPeriod').text
            
        # Resources
        area = root.find('AreaEstimates/Resources')
        if area is not None:
            metrics['bram'] = area.find('BRAM_18K').text
            metrics['dsp'] = area.find('DSP').text
            metrics['ff'] = area.find('FF').text
            metrics['lut'] = area.find('LUT').text
            
        return metrics
    except Exception as e:
        print(f"Error parsing XML {report_path}: {e}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: parse_reports.py <KERNEL_DIR> <OUTPUT_DIR>")
        sys.exit(1)
        
    kdir = sys.argv[1]
    outdir = sys.argv[2]
    
    # 1. Find the main synthesis report (csynth.xml usually)
    xml_reports = glob.glob(os.path.join(outdir, "*_csynth.xml"))
    if not xml_reports:
        print(f"No XML synthesis reports found in {outdir}")
        # Try to find text report? For now strict.
        sys.exit(0)
        
    # Use the first one found
    report_file = xml_reports[0]
    print(f"Parsing: {report_file}")
    
    data = parse_xml_report(report_file)
    
    if data:
        # Add metadata
        data['kernel_name'] = os.path.basename(kdir)
        
        # Save JSON
        json_path = os.path.join(outdir, "metrics.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Success. Metrics saved to {json_path}")
        
        # Print summary
        print("-" * 40)
        print(f"Latency (Cycles): {data.get('latency_cycles_max', 'N/A')}")
        print(f"Interval:         {data.get('interval_max', 'N/A')}")
        print(f"Est Clock:        {data.get('clock_estimated', 'N/A')} ns")
        print(f"LUT:              {data.get('lut', 'N/A')}")
        print(f"DSP:              {data.get('dsp', 'N/A')}")
        print("-" * 40)

if __name__ == "__main__":
    main()
