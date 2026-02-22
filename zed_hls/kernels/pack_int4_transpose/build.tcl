open_project pack_int4_transpose_prj

add_files kernel.cpp
add_files -tb tb.cpp

set_top pack_int4_transpose

open_solution "sol1" -flow_target vivado

# Target Part: VCU9P (Fallback from K1 experience)
set_part {xcvu9p-flga2104-2-i}

# Clock: 300MHz (3.33ns)
create_clock -period 3.33 -name default

config_export -format ip_catalog -rtl verilog

# 1. C Simulation
csim_design

# 2. Synthesis
csynth_design

# 3. Co-Simulation (Optional, often slow, verify if needed. Skipping for speed as requested in protocol)
# cosim_design

close_project
exit
