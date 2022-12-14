import os
import numpy as np
import argparse

from openvino.runtime import Model, ConstOutput, Shape, PartialShape
from openvino.runtime import Core, Tensor
from openvino.runtime import serialize

import utils

# convert a model with static input shapes to dynamic shape model.
def convert2dynamic(model_fullpath, new_model_dir):
    core = Core()
 
    model_fname = os.path.splitext(os.path.split(model_fullpath)[1])[0]
    model = core.read_model(model_fullpath)
    
    print(f"\n\n==================> converting ... {model_fullpath}==============>")
    print("Inputs of the model:")
    for port, _input in enumerate(model.inputs):
        print("\t[{}] {}".format(port, _input))
    print("Outputs of the model:")
    for port, _output in enumerate(model.outputs):
        print("\t[{}] {}".format(port, _output))
   
    ###
    for model_input in model.inputs:
        #print("{} has shape {}".format(model_input.any_name, model_input.get_partial_shape()))           
        new_shape = model_input.get_partial_shape()
        if new_shape.is_static:
            new_shape[0] = -1
        model.reshape({model_input.any_name: new_shape})
    
    ###
    os.system(f"mkdir -p {new_model_dir}")
    xml_path = os.path.join(new_model_dir, "{}.xml".format(model_fname))
    bin_path = os.path.join(new_model_dir, "{}.bin".format(model_fname))
    serialize(model, xml_path, bin_path)
    print("{} is saved.".format(xml_path))

    print("Inputs of the model:")
    for port, _input in enumerate(model.inputs):
        print("\t[{}] {}".format(port, _input))
    print("Outputs of the model:")
    for port, _output in enumerate(model.outputs):
        print("\t[{}] {}".format(port, _output))

# convert2dynamic("/home/dev/cecilia/dev/reference/openvino/model-validation-toolbox/10-91-242-212.iotg.sclab.intel.com/cv_bench_cache/try_builds_cache/sk_2sept_700models_22.2/GNMT/tf/tf_frozen/FP32/1/dldt/GNMT.xml")

# def convert2dynamic(model_fullpath, reshape):
#     core = Core()
 
#     model_fname = os.path.splitext(os.path.split(model_fullpath)[1])[0]
#     model = core.read_model(model_fullpath)

#     print("Inputs of the model:")
#     for port, _input in enumerate(model.inputs):
#         print("\t[{}] {}".format(port, _input))
#     print("Outputs of the model:")
#     for port, _output in enumerate(model.outputs):
#         print("\t[{}] {}".format(port, _output))
        
#     reshape_target = {i:s for i,s in enumerate(eval(reshape))}
#     if len(reshape_target) > 0:
#         model.reshape(reshape_target)
#         xml_path = "{}_reshaped.xml".format(model_fname)
#         bin_path = "{}_reshaped.bin".format(model_fname)
#         print("reshape {} using {}".format(model_fname, reshape_target))
#         serialize(model, xml_path, bin_path)
        
#convert2dynamic("/home/dev/cecilia/dev/reference/openvino/model-validation-toolbox/10-91-242-212.iotg.sclab.intel.com/cv_bench_cache/try_builds_cache/sk_2sept_700models_22.2/GNMT/tf/tf_frozen/FP32/1/dldt/GNMT.xml", "[[-1],[-1,50]]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Convert static model to dynamic")
    parser.add_argument("-f", "--name_filter", type=str, help="target model name filter", default="")
    parser.add_argument("--model_base", type=str, default="/home/dev/cecilia/dev/reference/openvino/model-validation-toolbox/10-91-242-212.iotg.sclab.intel.com/cv_bench_cache/try_builds_cache/sk_2sept_700models_22.2")
    parser.add_argument("--model_dynamic", type=str, default="/home/dev/cecilia/dev/reference/openvino/model-validation-toolbox/dynamic")
    args = parser.parse_args()

    print(f"searching for xml models in {args.model_base}...")
    models = utils.get_models_xml(args.model_base, args.name_filter)
    
    for i, xml_fullpath in enumerate(models):
        xml_relativepath = xml_fullpath[len(args.model_base)+1:]
        # print(f"xml_relativepath {xml_relativepath}")
        xml_relativedir = os.path.dirname(xml_relativepath)  #os.path.splitext(xml_relativepath)[0]
        new_model_dir =  os.path.join(args.model_dynamic, xml_relativedir)
        # print(f"new_model_dir {new_model_dir}")
        convert2dynamic(xml_fullpath, new_model_dir)