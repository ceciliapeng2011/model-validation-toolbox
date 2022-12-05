import sys

physcpubind="16,17,18,19,20"
ov_bin_folder="/home/dev/tingqian/openvino/bin/intel64/Release"
model_base="/home/dev/common/sk_13sept_75models_22.2_int8/"
bin_folder_ref="/home/dev/tingqian/openvino/bin13502/intel64/Release"
bin_folder_tag="/home/dev/tingqian/openvino/binbopt/intel64/Release"

if __name__ == "__main__":
    all_variables = dir()
    print(eval(sys.argv[1]))
