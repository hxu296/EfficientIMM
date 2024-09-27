import os
import sys
import json

RIPPLES_LOG_DIRECTORIES = {
    "ic": "strong-scaling-logs-ic-ripples",
    "lt": "strong-scaling-logs-lt-ripples",
}

EFFICIENT_IMM_LOG_DIRECTORIES = {
    "ic": "strong-scaling-logs-ic-eimm",
    "lt": "strong-scaling-logs-lt-eimm",
}

RESULTS_DIRECTORY = "results"

def parse_json_log(file_path):
    """
    Parse individual json file
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    log_dict = data[0]
    
    get_dataset_name = lambda x: x.split("/")[-1].split(".")[0]

    result = {
        "DiffusionModel": log_dict["DiffusionModel"],
        "NumThreads": log_dict["NumThreads"],
        "Time": log_dict["Total"],
        "Dataset": get_dataset_name(log_dict["Input"]),
    }
    return result

def parse_log_directory(log_directory):
    """
    Parse log directory, grouped by dataset, then by the number of threads
    """
    results = {}
    for file_name in os.listdir(log_directory):
        if file_name.endswith(".json"):
            file_path = os.path.join(log_directory, file_name)
            try:
                result = parse_json_log(file_path)
            except:
                # skip if there is an error parsing the file
                continue
            dataset = result["Dataset"]
            if dataset not in results:
                results[dataset] = {}
            num_threads = result["NumThreads"]
            results[dataset][num_threads] = result
    return results
            
def analyze_speedup(diffusion_model):
    """
    Diffusion model: ic or lt
    
    Open the log directories for efficient IMM and ripples
    Use ripples as the baseline and use *best* time for each method to calculate speedup
    """

    ripples_log_directory = RIPPLES_LOG_DIRECTORIES[diffusion_model]
    eimm_log_directory = EFFICIENT_IMM_LOG_DIRECTORIES[diffusion_model]

    ripples_results = parse_log_directory(ripples_log_directory)
    eimm_results = parse_log_directory(eimm_log_directory)

    # Get the best time (and their thread count) for each dataset
    def get_best_time(results):
        best_times = {}
        for dataset, dataset_results in results.items():
            best_time = float("inf")
            best_num_threads = None
            for num_threads, result in dataset_results.items():
                time = result["Time"]
                if time < best_time:
                    best_time = time
                    best_num_threads = num_threads
            best_times[dataset] = (best_time, best_num_threads)
        return best_times
    
    ripples_best_times = get_best_time(ripples_results)
    eimm_best_times = get_best_time(eimm_results)

    # Calculate speedup, record the thread count from both methods
    speedup_results = {}
    for dataset in ripples_best_times:
        if dataset not in eimm_best_times: continue
        ripples_time, ripples_num_threads = ripples_best_times[dataset]
        eimm_time, eimm_num_threads = eimm_best_times[dataset]
        speedup = ripples_time / eimm_time
        speedup_results[dataset] = {
            "Speedup": speedup,
            "RipplesNumThreads": ripples_num_threads,
            "EfficientIMMNumThreads": eimm_num_threads,
            "RipplesTime": ripples_time,
            "EfficientIMMTime": eimm_time,
        }

    return speedup_results

def get_speedup_summary(speedup_results):
    """
    Print speedup results as csv

    Columns: Dataset, Speedup, Ripples Time (s), EfficientIMM Time (s), Ripples Best #Threads, EfficientIMM Best #Threads
    """
    speedup_summary = ""
    speedup_summary += "Dataset, Speedup, Ripples Time (s),EfficientIMM Time (s),Ripples Best #Threads,EfficientIMM Best #Threads\n"

    for dataset, result in speedup_results.items():
        speedup_summary += f"{dataset},{result['Speedup']:.1f}x,{result['RipplesTime']/1000:.2f},{result['EfficientIMMTime']/1000:.2f},{result['RipplesNumThreads']},{result['EfficientIMMNumThreads']}\n"

    return speedup_summary

def print_and_save_speedup_summary():
    """
    Print speedup summary for both diffusion models
    """
    for diffusion_model in ["ic", "lt"]:
        print(f"=== Speedup for {diffusion_model.upper()} diffusion model ===")
        speedup_results = analyze_speedup(diffusion_model)
        speedup_summary = get_speedup_summary(speedup_results)
        print(speedup_summary, end="")
        if diffusion_model == "ic":
            file_name = "speedup_ic.csv"
        else:
            file_name = "speedup_lt.csv"
        if not os.path.exists(RESULTS_DIRECTORY):
            os.makedirs(RESULTS_DIRECTORY)
        with open(os.path.join(RESULTS_DIRECTORY, file_name), "w") as f:
            f.write(speedup_summary)
            print(f"(Speedup summary saved to {os.path.join(RESULTS_DIRECTORY, file_name)})")
        print()

if __name__ == "__main__":
    print_and_save_speedup_summary()