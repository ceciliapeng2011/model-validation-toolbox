import utils
import os,sys
import utils
import argparse
import re, math,config
import subprocess
import colorama
from colorama import Fore

base = config.ov_bin_folder
def get_cmd(bin_folder, model_path, args):
    if args.tput:
        return f"cd {bin_folder}; /usr/bin/time -v ./benchmark_app -t 10 -hint=tput -infer_precision=f32 -m {model_path}"
    return f"cd {bin_folder}; numactl --localalloc --physcpubind {config.physcpubind} /usr/bin/time -v ./benchmark_app -t {args.time} -nstreams=1 -nthreads=4 -hint=none -nireq=4 -infer_precision=f32 -m {model_path}"

class info:
    pat = {
        "load":re.compile("\[ INFO \] Load network took (\d*.?\d*) ms"),
        "latmin":re.compile("\[ INFO \] 	Min:\s*(\d*.?\d*) ms"),
        "latavg":re.compile("\[ INFO \] 	Average:\s*(\d*.?\d*) ms"),
        "tput":re.compile("\[ INFO \] Throughput:\s*(\d*.?\d*) FPS"),
        "build":re.compile("\[ INFO \] Build ........... (.*)"),
        "cpu":re.compile("\s*Percent of CPU this job got: (\d*)%"),
        "rss":re.compile("\s*Maximum resident set size \(kbytes\):\s(\d*)"),
        "pagefaults":re.compile("\s*Minor \(reclaiming a frame\) page faults:\s(\d*)"),
        "vcs":re.compile("\s*Voluntary context switches:\s(\d*)"),
        "ivcs":re.compile("\s*Involuntary context switches:\s(\d*)"),
    }
    def __init__(self, log_text) -> None:
        self.build = []
        for k,v in info.pat.items():
            if (k != "build"):
                setattr(self, k, float ('nan'))
        for line in log_text.splitlines():
            for k,v in info.pat.items():
                m = v.match(line)
                if (m):
                    if (k == "build"):
                        self.build.append(m.group(1))
                        continue
                    setattr(self, k, float(m.group(1)))

    def __repr__(self) -> str:
        return f"load={self.load} tput={self.tput} rss={self.rss} cpu={self.cpu} pagefaults={self.pagefaults} vcs={self.vcs} ivcs={self.ivcs} latmin={self.latmin} latavg={self.latavg} build={self.build}"

class compare_info:
    def __init__(self, i0, i1) -> None:
        self.i0 = i0
        self.i1 = i1
        self.tput = (i1.tput/i0.tput)
        self.latmin = (i1.latmin/i0.latmin)
        self.latavg = (i1.latavg/i0.latavg)
        self.load = (i1.load/i0.load)
        self.rss = (i1.rss/i0.rss)
        self.cpu = (i1.cpu/i0.cpu)
        self.pagefaults = (i1.pagefaults/i0.pagefaults)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Test performance")
    parser.add_argument("name_filter", type=str, help="target model name filter", default="")
    parser.add_argument("-t","--time", type=int, help="seconds to test", default=1)
    parser.add_argument("-r","--repeat", type=int, help="how many times to repeat the test to get max FPS", default=3)
    
    parser.add_argument("--tput", action="store_true", help="use hint=tput instead of (1stream + 4threads)")
    parser.add_argument("--model_base", type=str, default=config.model_base)

    args = parser.parse_args()
    #os.environ["ONEDNN_VERBOSE"]="1"
    #os.environ["OV_CPU_DEBUG_LOG"]="-"

    print(f"searching for xml models in {args.model_base}...")
    models = utils.get_models_xml(args.model_base, args.name_filter)

    comm_prefix = utils.get_common_prefix(models)

    models_list = []
    geomean_tput = 1.0
    geomean_load = 1.0
    geomean_rss = 1.0
    geomean_cnt = 0

    def do_test(bin_folder, mpath, args):
        ret = None
        error_happens = False
        for i in range(args.repeat):
            try:
                log_ref = subprocess.check_output(get_cmd(bin_folder, mpath, args), shell=True, stderr=subprocess.STDOUT, encoding="utf-8")
            except:
                print(Fore.RED + f"{mpath} failed in {bin_folder}, skip" + Fore.RESET)
                error_happens = True
                break
            i0 = info(log_ref)
            print(i0)
            if not ret:
                ret = i0
            elif (i0.tput > ret.tput):
                ret = i0
        return ret

    for i, mpath in enumerate(models):
        i0 = do_test(config.bin_folder_ref, mpath, args)
        if not i0:
            continue
        print(f"=> reference {i0}")
        i1 = do_test(config.bin_folder_tag, mpath, args)
        if not i1:
            continue
        print(f"=>    target {i1}")

        cmp = compare_info(i0, i1)

        geomean_tput *= cmp.tput
        geomean_load *= cmp.load
        geomean_rss *= cmp.rss
        geomean_cnt += 1

        # bigger is better
        def nocolored_ratio(prefix, ratio, bigger_better):
            return f"{prefix}:{ratio:.3f}"

        def colored_ratio(prefix, ratio, bigger_better):
            if bigger_better > 0:
                bigger_color = Fore.GREEN
                lower_clor = Fore.YELLOW
            else:
                bigger_color = Fore.YELLOW
                lower_clor = Fore.GREEN
            return (bigger_color if ratio > 1 else lower_clor) + f"{prefix}:{ratio:.3f}" + Fore.RESET

        def get_text(color_func):
            CF = color_func
            return f'{i:3d}/{len(models)} {" [ Improved ] " if cmp.tput > 1 else "[Regression]"} {CF("tput", cmp.tput, 1)} {CF("load",cmp.load,-1)} {CF("rss",cmp.rss, -1)} {mpath[len(comm_prefix):]}'

        s = get_text(colored_ratio)
        print(s)
        models_list.append([s, cmp])


    print("====================================================\n")
    print(f"models in folder: {comm_prefix}")
    def getTput(item):
        s, cmp = item
        return cmp.tput

    models_list.sort(key=getTput, reverse=True)
    for s,cmp in models_list:
        print(s)

    print(f"geomean_tput = {geomean_tput ** (1/geomean_cnt):.3f}")
    print(f"geomean_load = {geomean_load ** (1/geomean_cnt):.3f}")
    print(f"geomean_rss = {geomean_rss ** (1/geomean_cnt):.3f}")

