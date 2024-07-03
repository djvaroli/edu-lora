#!/bin/bash

# return all visible GPUs as a comma separated string of GPU indices
# e.g. "0,1,2,3,4,5,6,7" if 8 GPUs are available
function getAllVisibleGPUs {
    local nvidia_smi_output=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    local gpu_ids=$(echo $nvidia_smi_output | tr ' ' ',')

    echo $gpu_ids
}


# Function to check if a specific GPU is in use
function isGpuInUse {
    local gpu_index=$1
    local nvidia_smi_output=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    while IFS=, read -r index memory_used; do
        if [[ "$index" -eq "$gpu_index" ]]; then
            if [[ "$memory_used" -gt 0 ]]; then
                echo "  GPU $gpu_index is not free (memory used: $memory_used MiB)"
                return 1
            fi
        fi
    # <<< passes the string as input to the while loop
    done <<< "$nvidia_smi_output"

    return 0
} 


# Function that checks if even a single GPU is in use from a list of GPUs
function AreGPUsInUse {
    local gpus=$1
    # IFS stands for Internal Field Separator
    IFS=',' read -r -a gpu_array <<< "$gpus"
    local any_gpu_in_use=0

    for gpu in "${gpu_array[@]}"
    do
        isGpuInUse "$gpu"
        local gpu_status=$?

        if [ $gpu_status -ne 0 ]; then
            return 1
        fi
    done

    return 0
}


# list all GPUs that should be used for the experiments, use "all" to use all GPUs
# script will check if the following GPUs are not in use (memory used == 0MiB)
VISIBLE_GPUS="7"
EXECUTABLE="python"

if [ "$VISIBLE_GPUS" == "all" ]; then
    VISIBLE_GPUS=$(getAllVisibleGPUs)
fi


# declare an array of commands to run
declare -a commands=(
    "mlp_mixer_imagenette_lora.py --adapter_type lora --rank 1 --alpha 0.5 --n_epochs 15"
    "mlp_mixer_imagenette_lora.py --adapter_type lora --rank 2 --alpha 0.5 --n_epochs 15"
    "mlp_mixer_imagenette_lora.py --adapter_type lora --rank 4 --alpha 0.5 --n_epochs 15"
    "mlp_mixer_imagenette_lora.py --adapter_type lora --rank 10 --alpha 0.5 --n_epochs 15"
    "mlp_mixer_imagenette_lora.py --adapter_type lora --rank 16 --alpha 0.5 --n_epochs 15"
    "mlp_mixer_imagenette_lora.py --adapter_type lora --rank 32 --alpha 0.5 --n_epochs 15"
)


# main loop over all commands
for cmd in "${commands[@]}"
do
    # color blue
    echo -e "\e[34mRunning command: CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $EXECUTABLE $cmd\e[0m"

    # will loop until all VISIBLE_GPUs are free (memory used == 0MiB)
    while true; do
        AreGPUsInUse "$VISIBLE_GPUS"
        all_gpus_free=$?

        if [ $all_gpus_free -eq 0 ]; then
            break
        fi
        # color text yellow
        echo -e "\e[33mWaiting for all visbile GPUs to be free...\e[0m"
        sleep 10
    done

    # set CUDA_VISIBLE_DEVICES, run the command using the executable
    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $EXECUTABLE $cmd
done

echo "All jobs completed"
