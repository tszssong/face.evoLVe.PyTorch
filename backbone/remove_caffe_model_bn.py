import caffe
import math
import numpy as np
import remove_caffe_prototxt_bn as remove_proto_bn
import sys

# prototxt = r'person_seg_v15_deploy.prototxt'
# caffemodel = r'person_seg_v15_train.caffemodel'
#
# dst_prototxt = r'person_seg_v15_deploy_without_bn.prototxt'
# dst_caffemodel = r'person_seg_v15_deploy_without_bn.caffemodel'

def removeProtoCaffemodelBN(src_proto, src_model, dst_proto, dst_model):
    remove_proto_bn.removePrototxtBN(src_proto, dst_proto)
    net = caffe.Net(src_proto, src_model, caffe.TEST)
    net_dst = caffe.Net(dst_proto, caffe.TEST)

    for k in net_dst.params:
        if k in net.params:
            for i in range(len(net.params[k])):
                net_dst.params[k][i].data[...] = net.params[k][i].data
                print ('copy from', k)

    for i in range(len(net.layers)):
        if net.layers[i].type == 'Convolution':
            print (net._layer_names[i], net.layers[i].type)
            conv_name = net._layer_names[i]
            # if conv_name not in change_conv_names:
            #     continue
            j = i + 1
            if net.layers[j].type == 'BatchNorm':
                print (' ', net._layer_names[j], net.layers[j].type)
                print (' ', net._layer_names[j + 1], net.layers[j + 1].type)
                bn_name = net._layer_names[j]
                scale_name = net._layer_names[j + 1]

                bn_mean = net.params[bn_name][0].data
                bn_variance = net.params[bn_name][1].data
                bn_scale = net.params[bn_name][2].data

                scale_weight = net.params[scale_name][0].data
                scale_bias = net.params[scale_name][1].data

                # print bn_name
                # print bn_mean.shape, bn_variance.shape, bn_scale.shape
                # print bn_scale
                # print scale_name
                # print scale_weight.shape, scale_bias.shape
                # print net.params[conv_name][0].data.shape
                # # print net.params[conv_name][1].data.shape
                # sys.exit()

                dst_conv_weight = net.params[conv_name][0].data
                if np.count_nonzero(bn_variance) != bn_variance.size:
                    assert False

                alpha = scale_weight / np.sqrt(bn_variance / bn_scale)


                if len(net.params[conv_name]) > 1:
                    dst_conv_bias = net.params[conv_name][1].data
                else:
                    dst_conv_bias = 0

                assert len(dst_conv_weight) == len(alpha)
                for i in range(len(alpha)):
                    dst_conv_weight[i] = dst_conv_weight[i] * alpha[i]

                dst_conv_bias = dst_conv_bias * alpha + (scale_bias - (bn_mean / bn_scale) * alpha)

                net_dst.params[conv_name][0].data[...] = dst_conv_weight

                if len(net_dst.params[conv_name]) > 1:
                    net_dst.params[conv_name][1].data[...] = dst_conv_bias

                # break
            # break

    net_dst.save(dst_model)
    print ('FINISH ##############################')

    # net_dst.save(dst_caffemodel)

def zrnRemoveProtoCaffemodelBN(src_proto, src_model, dst_proto, dst_model):
    remove_proto_bn.zrnRemovePrototxtBN(src_proto, dst_proto)
    net = caffe.Net(src_proto, src_model, caffe.TEST)
    net_dst = caffe.Net(dst_proto, caffe.TEST)

    eps = 1e-5

    for k in net_dst.params:
        if k in net.params:
            for i in range(len(net.params[k])):
                net_dst.params[k][i].data[...] = net.params[k][i].data
            for i in range(len(net.params[k]), len(net_dst.params[k])):
                net_dst.params[k][i].data[...] = 0

    for i in range(len(net.layers)):
        if (net.layers[i].type == 'BatchNorm') and (net.layers[i + 1].type == 'Scale'):
            if (i > 0) and ((net.layers[i - 1].type == 'Convolution') or (net.layers[i - 1].type == 'InnerProduct')): #Conv + BN
                
                conv_name = net._layer_names[i - 1]
                bn_name = net._layer_names[i]
                scale_name = net._layer_names[i + 1]

                bn_mean = net.params[bn_name][0].data
                bn_variance = net.params[bn_name][1].data
                bn_scale = net.params[bn_name][2].data
                if np.count_nonzero(bn_variance) != bn_variance.size: assert False
                
                bn_mean = bn_mean / bn_scale
                bn_variance = bn_variance / bn_scale + eps


                scale_weight = net.params[scale_name][0].data
                scale_bias = net.params[scale_name][1].data


                dst_conv_weight = np.copy(net_dst.params[conv_name][0].data)
                dst_conv_bias = np.copy(net_dst.params[conv_name][1].data)

                alpha = scale_weight / np.sqrt(bn_variance)
                assert len(dst_conv_weight) == len(alpha)
                for i in range(len(alpha)): dst_conv_weight[i] = dst_conv_weight[i] * alpha[i]
                dst_conv_bias = (dst_conv_bias - bn_mean) * alpha + scale_bias

                net_dst.params[conv_name][0].data[...] = dst_conv_weight
                net_dst.params[conv_name][1].data[...] = dst_conv_bias
                continue
            # elif ((i + 2) < len(net.layers)) and (net.layers[i + 2].type == 'Convolution'): # BN + Conv
            #     bn_name = net._layer_names[i]
            #     scale_name = net._layer_names[i + 1]
            #     conv_name = net._layer_names[i + 2]

            #     bn_mean = net.params[bn_name][0].data
            #     bn_variance = net.params[bn_name][1].data
            #     bn_scale = net.params[bn_name][2].data
            #     if np.count_nonzero(bn_variance) != bn_variance.size: assert False
            #     bn_mean = bn_mean / bn_scale + eps
            #     bn_variance = bn_variance / bn_scale + eps

            #     scale_weight = net.params[scale_name][0].data
            #     scale_bias = net.params[scale_name][1].data



            #     conv_weight = np.copy(net_dst.params[conv_name][0].data)
            #     dst_conv_bias = np.copy(net_dst.params[conv_name][1].data)

            #     alpha = scale_weight / np.sqrt(bn_variance)

            #     dst_conv_weight = np.copy(conv_weight.transpose(1,2,3,0))
            #     assert len(dst_conv_weight) == len(alpha)
            #     for i in range(len(alpha)): dst_conv_weight[i] = dst_conv_weight[i] * alpha[i]
            #     dst_conv_weight = dst_conv_weight.transpose(3,0,1,2)

            #     beta = scale_bias - alpha * bn_mean
            #     tmp_conv_weight = np.copy(conv_weight.transpose(1,2,3,0))
            #     assert len(tmp_conv_weight) == len(beta)
            #     for i in range(len(beta)): tmp_conv_weight[i] = tmp_conv_weight[i] * beta[i]
            #     tmp_conv_weight = tmp_conv_weight.transpose(3,0,1,2)

            #     dst_conv_bias = tmp_conv_weight.sum(axis = (1,2,3)) + dst_conv_bias

            #     net_dst.params[conv_name][0].data[...] = dst_conv_weight
            #     net_dst.params[conv_name][1].data[...] = dst_conv_bias
            #     continue

    net_dst.save(dst_model)
    print ('FINISH ##############################')

    # net_dst.save(dst_caffemodel)

