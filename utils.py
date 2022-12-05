import glob,os,re
from pathlib import Path

accuracy_checker_configs = "/home/dev/common/accuracy-checker/configs/omz_validation/"
definitions_file=f"{accuracy_checker_configs}/dataset_definitions.yml"
data_source=f"/home/dev/common/odt_nfs_share/omz-validation-datasets"
annotations=f"/home/dev/common/annotations"
model_attributes=f"/home/dev/common/odt_nfs_share/omz-validation-datasets/model_attributes"

class BulkInferenceHelper:
    def get_framework(self, model_path: str) -> str:
        caffe2_onnx = os.path.join("caffe2", "onnx")
        if model_path.find(caffe2_onnx) != -1:
            return "caffe2"

        if model_path.find("onnx") != -1:
            return ["onnx", "pytorch"]

        if model_path.find("mxnet") != -1:
            return ["mxnet"]

        if model_path.find("caffe2") != -1:
            return ["caffe2"]

        if model_path.find("caffe") != -1:
            return ["caffe"]

        if model_path.find("tf2") != -1:
            return ["tf2"]

        if model_path.find("tf") != -1:
            return ["tf"]

        if model_path.find("paddle") != -1:
            return ["paddle"]

        return None

    def get_framework2(self, model_path: str) -> str:
        caffe2_onnx = os.path.join("caffe2", "onnx")
        if model_path.find(caffe2_onnx) != -1:
            return ["caffe2", "pytorch"]

        caffe_onnx = os.path.join("caffe2", "onnx")
        if model_path.find(caffe_onnx) != -1:
            return ["caffe", "pytorch"]

        return None

    def exists(self, directory: str, file_name: str) -> str:
        for file_path in Path(directory).glob(os.path.join("**", file_name)):
            return file_path
        return None

    def find_yaml_file(self, model_path, configs_path: str) -> str:
        yml_config_name_without_ext = os.path.splitext(os.path.basename(model_path))[0]

        yml_config_path = self.exists(configs_path, yml_config_name_without_ext + ".yml")
        if yml_config_path:
            return yml_config_path

        frameworks = self.get_framework(model_path)
        if frameworks:
            for framework in frameworks:
                yml_config_path = self.exists(configs_path, "{}-{}.yml".format(yml_config_name_without_ext, framework))
                if yml_config_path:
                    return yml_config_path

        frameworks = self.get_framework2(model_path)
        if frameworks:
            for framework in frameworks:
                yml_config_path = self.exists(configs_path, "{}-{}.yml".format(yml_config_name_without_ext, framework))
                if yml_config_path:
                    return yml_config_path

        return None


def find_yaml_file(model_path) -> str:
    bi = BulkInferenceHelper()
    return bi.find_yaml_file(model_path, accuracy_checker_configs)


def get_models_xml(model_base, name_filter):
    models = []
    # user can specify path to override model_base
    # or a pattern of path to filter subset of models in model_base
    if os.path.isdir(name_filter):
        model_base = name_filter
    for root, dirs, files in os.walk(model_base):
        for file in files:
            if os.path.splitext(file)[1] == ".xml":
                fullpath = os.path.join(root, file)
                if (name_filter in fullpath):
                    models.append(fullpath)
                    found_xml = models[-1][len(model_base):]
                    print(f"{len(models)}:  {found_xml}")
    return models


# name_filters: filters separated with ",", each filter can be
#
#    path        all .xml files recusivedly found under path
#    path:kw     all .xml files whose full path contains kw substring
#    :kw         model_base is the path
#    kw          same as :kw when kw is not exist as path
#
def get_models_xml(model_base, name_filters):
    models = []
    for name_filter in name_filters.split(","):
        # user can specify path to override model_base
        # or a pattern of path to filter subset of models in model_base
        #
        if ":" in name_filter:
            path, kw = name_filter.split(":")
            if len(path)==0:
                path = model_base
        else:
            if os.path.isdir(name_filter):
                path = name_filter
                kw = ''
            elif os.path.isfile(name_filter):
                models.append(name_filter)
                print(f"{len(models)}:  {name_filter}")
                continue
            else:
                path = model_base
                kw = name_filter

        print(f"searching models (keyword:{kw}) in {path} ...")

        for root, dirs, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1] == ".xml":
                    fullpath = os.path.join(root, f)
                    if (kw in fullpath):
                        models.append(fullpath)
                        found_xml = models[-1][len(path):]
                        print(f"{len(models)}:  {found_xml}")
    return models

def gen_device_config(device_config_path, bf16):
    ENFORCE_BF16 = "YES" if bf16 else "NO"
    with open(device_config_path, "w") as devcfg_file:
        devcfg_file.write(f'CPU:\n')
        devcfg_file.write(f'    ENFORCE_BF16: "{ENFORCE_BF16}"\n')
        devcfg_file.write(f'    NUM_STREAMS: 1\n')

# path prefix
def get_common_prefix(lines):
    def find_common_prefix(a, cur_prefix):
        cur_prefix = cur_prefix[:len(a)]
        last_sep = 0
        for i in range(0, len(cur_prefix)):
            if cur_prefix[i] != a[i]:
                return cur_prefix[:last_sep + 1]
            if cur_prefix[i] == os.path.sep:
                last_sep = i
        return cur_prefix

    comm_prefix = lines[0]
    for i in range(1, len(lines)):
        comm_prefix = find_common_prefix(lines[i], comm_prefix)
    return comm_prefix


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=f"Test utils")
    parser.add_argument("--ext", type=str, default=None)
    parser.add_argument("--reg", action="store_true")
    args = parser.parse_args()

    if args.reg:
        pat = re.compile("(\d*(?:%|Db))")
        print(pat.match("8%"))
        print(pat.match("8Db"))
        m = re.compile(".*accuracy_check.*-c\s+([^ ]*)\s").match("+ accuracy_check --target_framework dlsdk -td CPU --device_config acc_s10_22.1/device_config.yml --definitions /home/dev/common/accuracy-checker/configs/omz_validation//dataset_definitions.yml --source /home/dev/common/odt_nfs_share/omz-validation-datasets --annotations /home/dev/common/annotations --models /home/dev/common/inn_nfs_share/cv_bench_cache/sk_13sept_75models_22.1_int8/se-resnext-50/caffe/caffe/FP16/INT8/1/dldt/optimized/se-resnext-50.xml -c /home/dev/common/accuracy-checker/configs/omz_validation/public_omz_package/se-resnext-50-caffe.yml --shuffle False -ss 10")
        print(m.group(1))

    if args.ext:
        results, base = extract_accuracy(args.ext)
        for i, r in enumerate(results):
            print(f"[{i:3}/{len(results)}] : {results[r]}")
        print(f"base = {base}")
        sys.exit(0)