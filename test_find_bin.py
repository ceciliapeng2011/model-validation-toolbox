import utils
import os
import argparse


benchmark_app="numactl --localalloc --physcpubind 32-63 /home/dev/tingqian/openvino/bin/intel64/Debug/benchmark_app"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Find binary post ops")
    parser.add_argument("--model_base", type=str, default="/home/dev/common/sk_13sept_75models_22.2_int8/")    
    parser.add_argument("result_folder")
    args = parser.parse_args()

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    print(f"searching for xml models in {args.model_base}...")
    models = utils.get_models_xml(args.model_base)

    os.environ["ONEDNN_VERBOSE"]="1"
    os.environ["OV_CPU_DEBUG_LOG"]="-"
    # find binary_ postops
    script_file_path = os.path.join(args.result_folder, "test.sh")
    summary_file_path = os.path.join(args.result_folder, "summary.txt")

    zlogs = []
    with open(script_file_path, "w") as script_file:
        script_file.write(f"echo  ===start=== |& tee {summary_file_path}\n")
        script_file.write(f"rm {os.path.join(args.result_folder, 'zlog_*.txt')}\n")
        for i, xml in enumerate(models):
            mpath = xml
            log_file_path = os.path.join(args.result_folder, f"zlog_{i}.txt")
            script_file.write(f"echo ============ {i}/{len(models)}     {mpath} |& tee {log_file_path}\n")
            testcmd = f"{benchmark_app} -niter 1 -nstreams 1 -nthreads 1 -hint none -m {mpath} |& tee -a {log_file_path}"
            script_file.write(f"{testcmd}\n")

            script_file.write(f"cat {log_file_path} |& grep -e binary_ -e '============ ' |& tee -a {summary_file_path}\n")
            zlogs.append(log_file_path)

    os.system(f"/bin/bash {script_file_path}")

    print(f"    {script_file_path}")
    print(f"    {summary_file_path}")

