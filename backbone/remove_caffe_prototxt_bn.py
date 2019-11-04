import caffe
import math
import numpy as np
import json
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys

def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object

def readProtoSolverFile(filepath):
    solver_config = caffe.proto.caffe_pb2.NetParameter()
    return readProtoFile(filepath, solver_config)

def removePrototxtBN(src_prototxt, dst_prototxt):

    net_params = readProtoSolverFile(src_prototxt)

    outfile = open(dst_prototxt, 'w')
    outfile.write('name: \"' + net_params.name + '\"\n')
    outfile.write('\n')
    print net_params.name

    replace_name = None
    top_name = None
    index = 0
    start_remove = False

    for layer in net_params.layer:
        index = index + 1
        if layer.type == 'Convolution':
            start_remove = True
        elif layer.type == 'BatchNorm' and start_remove:
            top_name = layer.bottom[0]
            continue
        elif layer.type == 'Scale' and start_remove:
            replace_name = layer.top[0]
            continue
        else:
            start_remove = False
            replace_name = None
            top_name = None
        # if layer.type == 'Proposal' or layer.type == 'ROIPooling' or layer.type == 'Input':
        #     replace_name = None
        if layer.type == 'Convolution' and net_params.layer[index].type == 'BatchNorm':
            layer.convolution_param.bias_term = True

        # print "layer name:", layer.name, "top name:", top_name, "replace_name:", replace_name
        outfile.write('layer {\n')
        # if replace_name == "stage3_unit2_conv1":
        #     print  replace_name, layer.name
        if replace_name and layer.bottom[0] == replace_name:
            layer.bottom[0] = top_name
        outfile.write('  '.join(('\n' + str(layer)).splitlines(True)))
        outfile.write('\n}\n\n')

def zrnRemovePrototxtBN(src_prototxt, dst_prototxt):

    net_params = readProtoSolverFile(src_prototxt)

    outfile = open(dst_prototxt, 'w')
    outfile.write('name: \"' + net_params.name + '\"\n')
    outfile.write('\n')
    print net_params.name

    replace_name = None
    top_name = None
    start_remove = False

    for i in range(len(net_params.layer)):

        if (net_params.layer[i].type == 'BatchNorm') and (net_params.layer[i + 1].type == 'Scale'):
            if (i > 0) and ((net_params.layer[i - 1].type == 'Convolution') or (net_params.layer[i - 1].type == 'InnerProduct')):
                net_params.layer[i - 1].top[0] = net_params.layer[i + 1].top[0]
                if net_params.layer[i - 1].type == 'Convolution':
                    net_params.layer[i - 1].convolution_param.bias_term = True
            # elif ((i + 2) < len(net_params.layer)) and (net_params.layer[i + 2].type == 'Convolution'):
            #     net_params.layer[i + 2].bottom[0] = net_params.layer[i].bottom[0]
            #     net_params.layer[i + 2].convolution_param.bias_term = True


    i = 0
    while i < len(net_params.layer):
        if (net_params.layer[i].type == 'BatchNorm') and (net_params.layer[i + 1].type == 'Scale'):
            if (i > 0) and ((net_params.layer[i - 1].type == 'Convolution') or (net_params.layer[i - 1].type == 'InnerProduct')):
                i = i + 2
                continue
            # elif ((i + 2) < len(net_params.layer)) and (net_params.layer[i + 2].type == 'Convolution'):
            #     i = i + 2
            #     continue

        outfile.write('layer {')
        outfile.write('  '.join(('\n' + str(net_params.layer[i])).splitlines(True)))
        outfile.write('}\n\n')
        i = i + 1

