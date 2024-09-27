#!/bin/bash

imm_bin=./build/release/src_efficient_imm/tools/imm
datasets=(
  "com-amazon.ungraph.IC.tsv"
  "com-youtube.ungraph.IC.tsv"
  "com-dblp.ungraph.IC.tsv"
  "as-skitter.IC.tsv"
  "soc-pokec-relationship.IC.tsv"
  "web-Google.IC.tsv"
  "com-lj.ungraph.IC.tsv"
)

# Get number of total cores (excluding hyperthreading)
phys_cores=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
sockets=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
total_phys_cores=$((phys_cores * sockets))
threads=()

# Double the value until the total number of physical cores is reached
value=4
while [ $value -le $total_phys_cores ]; do
    threads+=($value)
    value=$((value * 2))
done

echo "Running thread configurations: ${threads[@]}"

log_dir="strong-scaling-logs-ic-eimm"
test_data_dir="test-data"

mkdir -p "$log_dir"

for dataset in "${datasets[@]}"; do
  for thread in "${threads[@]}"; do
    dataset_name=$(basename "$dataset" .tsv)
    log_file="${log_dir}/${dataset_name}_threads_${thread}.json"

    case $thread in
      128)
        numa_cmd="numactl -a -i -C 0-127"
        ;;
      64)
        numa_cmd="numactl -a -i all -C 0-63"
        ;;
      32)
        numa_cmd="numactl -a -i all -C 0-31"
        ;;
      16)
        numa_cmd="numactl -a -i all -C 0-15"
        ;;
      8)
        numa_cmd="numactl -a -i all -C 0-7"
        ;;
      4)
        numa_cmd="numactl -a -i all -C 0-3"
        ;;  
      2)
        numa_cmd="numactl -a -i all -C 0-1"
        ;;        
    esac

    if [[ "$dataset" == *"ungraph"* ]]; then
      OMP_PLACES=cores OMP_NUM_THREADS=$thread $numa_cmd $imm_bin -i "${test_data_dir}/$dataset" -u -w -p -k 50 -d IC -e 0.5 -o "$log_file"
    else
      OMP_PLACES=cores OMP_NUM_THREADS=$thread $numa_cmd $imm_bin -i "${test_data_dir}/$dataset" -w -p -k 50 -d IC -e 0.5 -o "$log_file"
    fi
  done
done