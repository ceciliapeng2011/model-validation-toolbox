import utils
import os,sys
import utils
import argparse
import re, math,config


base = config.ov_bin_folder
benchmark_app=f"numactl --localalloc --physcpubind {config.physcpubind} {base}/benchmark_app -t 3 -nstreams=1 -nthreads=4 -hint=none -nireq=4 -infer_precision=f32 "

class info:
    pat = {
        "load":re.compile("\[ INFO \] Load network took (\d*.?\d*) ms"),
        "latmin":re.compile("\[ INFO \] 	Min:\s*(\d*.?\d*) ms"),
        "latavg":re.compile("\[ INFO \] 	Average:\s*(\d*.?\d*) ms"),
        "tput":re.compile("\[ INFO \] Throughput:\s*(\d*.?\d*) FPS")

    }
    def __init__(self, fname) -> None:
        for k,v in info.pat.items():
            setattr(self, k, float ('nan'))

        with open(fname, "r") as f:
            for line in f.readlines():
                for k,v in info.pat.items():
                    m = v.match(line)
                    if (m):
                        setattr(self, k, float(m.group(1)))

    def __repr__(self) -> str:
        return f"load={self.load} latmin={self.latmin} latavg={self.latavg} tput={self.tput}"

class compare_info:
    def __init__(self, i0, i1) -> None:
        self.i0 = i0
        self.i1 = i1
        self.tput = (i1.tput/i0.tput)
        self.latmin = (i1.latmin/i0.latmin)
        self.latavg = (i1.latavg/i0.latavg)
        self.load = (i1.load/i0.load)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Test performance")
    parser.add_argument("-f", "--filter", type=str, default="cmp.tput>0")
    parser.add_argument("-s", "--sort", type=str, default="cmp.tput")
    parser.add_argument("-r", "--reverse", action="store_true")
    parser.add_argument("--model_base", type=str, default=config.model_base)
    parser.add_argument("result_folder", nargs='+')
    
    args = parser.parse_args()

    if len(args.result_folder) > 1:
        print(f"searching for xml models in {args.model_base}...")
        models = utils.get_models_xml(args.model_base)
        cnt = 0
        models_list = []
        models_nan = []
        for i, mpath in enumerate(models):
            i0 = info(os.path.join(args.result_folder[0], f"zlog_{i}.txt"))
            i1 = info(os.path.join(args.result_folder[1], f"zlog_{i}.txt"))
            cmp = compare_info(i0, i1)
            if math.isnan(cmp.tput):
                models_nan.append([mpath, cmp])
            elif eval(args.filter):
                models_list.append([mpath, cmp])
        
        def sortFunc(item):
            mpath, cmp = item
            return eval(args.sort)

        models_list.sort(key=sortFunc, reverse=args.reverse)

        geomean_tput = 1.0
        for i, (mpath, cmp) in enumerate(models_list):
            geomean_tput *= cmp.tput
            print(f"[{i}] {mpath}")
            print(f"    {cmp.i0}")
            print(f"    {cmp.i1}")
            print(f" ratio: {cmp.load:.3f}  {cmp.latmin:.3f}  {cmp.latavg:.3f}  {cmp.tput:.3f}")

        geomean_tput = geomean_tput ** (1/len(models_list))
        print(f"geomean_tput = {geomean_tput:.3f}")

        print("========= models failed to execute ============")
        for i, (mpath, cmp) in enumerate(models_nan):
            print(f"[{i}] {mpath}")
        
        print("========= args ============")
        print(f"--filter = {args.filter}")
        print(f"--sort = {args.sort}")
        print(f"--reverse = {args.reverse}")
        sys.exit(0)

    if len(args.result_folder) == 1:
        result_folder = args.result_folder[0]
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        #os.environ["ONEDNN_VERBOSE"]="1"
        #os.environ["OV_CPU_DEBUG_LOG"]="-"

        script_file_path = os.path.join(result_folder, "test.sh")
        summary_file_path = os.path.join(result_folder, "summary.txt")

        print(f"searching for xml models in {args.model_base}...")
        models = utils.get_models_xml(args.model_base)

        zlogs = []
        with open(script_file_path, "w") as script_file:
            script_file.write(f"export LD_LIBRARY_PATH={base}\n")
            script_file.write(f"echo  ===start=== |& tee {summary_file_path}\n")
            script_file.write(f"rm -f {os.path.join(result_folder, 'zlog_*.txt')}\n")
            for i, mpath in enumerate(models):
                log_file_path = os.path.join(result_folder, f"zlog_{i}.txt")
                script_file.write(f"echo ============ {i}/{len(models)}     {mpath} |& tee {log_file_path}\n")
                testcmd = f"{benchmark_app} -m {mpath}  |& tee -a {log_file_path}"
                script_file.write(f"{testcmd}\n")

                script_file.write(f"cat {log_file_path} |& grep -e 'Throughput:' -e '============ ' |& tee -a {summary_file_path}\n")
                zlogs.append(log_file_path)

        os.system(f"/bin/bash {script_file_path}")

        print(f"    {script_file_path}")
        print(f"    {summary_file_path}")

