# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generator for the splits of DeepNets-1M.

In-Distribution splits: TRAIN, VAL, TEST.

Out-of-Distribution splits: WIDE, DEEP, DENSE, BNFREE.

PREDEFINED is created on the fly in the ppuda.deepnets1m.loader.


Example:

    python experiments/net_generator.py train 1000000 ./data

"""


import time
import h5py
import json
import os
from os.path import join
import sys
import subprocess
from ppuda.deepnets1m.graph import MlPGraph
from ppuda.utils import *
from ppuda.deepnets1m.genotypes import *
from ppuda.deepnets1m.net import MlpNetwork, get_cell_ind


def main():

    try:
        split = sys.argv[1].lower()
        N = int(sys.argv[2])
        data_dir = sys.argv[3]
    except:
        print('\nExample of usage: python deepnets1m/net_generator.py train 1000000 ./data\n')
        raise

    device = 'cpu'  # no much benefit of using cuda

    print(split, N, data_dir, device, flush=True)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    set_seed(0 if split == 'val' else 1)


    # for 'train', 'val', 'test' we have the same network generator
    # for 'wide' we re-use the 'test' split and increase the number of channels when evaluate the model
    # for 'bnfree' the generator is the same except that all nets have no BN
    # 'predefined' is created on the fly in the deepnets1m.loader


    try:
        gitcommit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print('gitcommit:', gitcommit, flush=True)
    except Exception as e:
        print(e, flush=True)

    start = time.time()

    meta_data = {}
    meta_data[split] = {'nets': [], 'meta': {}}
    op_types, op_types_back, primitives, primitives_back = {}, {}, {}, {}

    h5_file = join(data_dir, 'mlpnets1m_%s.hdf5' % split)
    meta_file = join(data_dir, 'mlpnets1m_%s_meta.json' % split)

    for f in [h5_file, meta_file]:
        if os.path.exists(h5_file):
            raise ValueError('File %s already exists. The script will exit now to avoid accidental overwriting of the file.' % f)


    with h5py.File(h5_file, 'w') as h5_data:

        h5_data.attrs['title'] = 'DeepNets-1M'
        group = h5_data.create_group(split)

        while len(meta_data[split]['nets']) < N:

            net_args = {
                        'fc_layers': int(np.random.randint(1, 3)),  # number of fully connected layers before classification                        
                        'fc_dim': 256,
                        'inp_dim': 17,
                        'out_dim': 6,
                        }

            skip = False
            graph = None
            num_params = {}

            # for dset_name in ['cifar10', 'imagenet']:

            model = MlpNetwork(**net_args).to(device)                    

            c, n = capacity(model)
            num_params['custom'] = n

            graph = MlPGraph(model, ve_cutoff=250, list_all_nodes=True)
            layers = 1

            # assert layers == len(graph.node_info), (layers, len(graph.node_info))
            cell_ind, n_nodes, nodes_array = 0, 0, []
            for j in range(layers):

                n_nodes += len(graph.node_info[j])

                for node in graph.node_info[j]:

                    param_name, name, sz = node[1:4]
                    cell_ind_ = get_cell_ind(param_name, layers)
                    if cell_ind_ is not None:
                        cell_ind = cell_ind_

                    assert cell_ind == j, (cell_ind, j, node)

                    if name == 'conv' and (len(sz) == 2 or sz[2] == sz[3] == 1):
                        name = 'conv_1x1'

                    if name not in primitives:
                        ind = len(primitives)
                        primitives[name] = ind
                        primitives_back[ind] = name

                    if param_name.startswith('cells.'):
                        # remove cells.x. prefix
                        pos1 = param_name.find('.')
                        assert param_name[pos1 + 1:].find('.') >= 0, node
                        pos2 = pos1 + param_name[pos1 + 1:].find('.') + 2
                        param_name = param_name[pos2:]

                    if param_name not in op_types:
                        ind = len(op_types)
                        op_types[param_name] = ind
                        op_types_back[ind] = param_name

                    nodes_array.append([primitives[name], cell_ind, op_types[param_name]])

            nodes_array = np.array(nodes_array).astype(np.uint16)

            A = graph._Adj.cpu().numpy().astype(np.uint8)
            assert nodes_array.shape[0] == n_nodes == A.shape[0] == graph.n_nodes, (nodes_array.shape, n_nodes, A.shape, graph.n_nodes)

            idx = len(meta_data[split]['nets'])
            group.create_dataset(str(idx) + '/adj', data=A)
            group.create_dataset(str(idx) + '/nodes', data=nodes_array)

            net_args['num_nodes'] = int(A.shape[0])
            net_args['num_params'] = num_params

            # net_args['genotype'] = to_dict(net_args['genotype'])
            meta_data[split]['nets'].append(net_args)
            meta_data[split]['meta']['primitives_ext'] = primitives_back
            meta_data[split]['meta']['unique_op_names'] = op_types_back

            # if (idx + 1) % 100 == 0 or idx >= N - 1:
            #     all_n_nodes = np.array([net['num_nodes'] for net in meta_data[split]['nets']])
            #     all_n_params = np.array([net['num_params']['cifar10'] for net in meta_data[split]['nets']])  / 10 ** 6
            #     print('N={} nets created: \t {}-{} nodes (mean\u00B1std: {:.1f}\u00B1{:.1f}) '
            #           '\t {:.2f}-{:.2f} params (M) (mean\u00B1std: {:.2f}\u00B1{:.2f}) '
            #           '\t {} unique primitives, {} unique param names '
            #           '\t total time={:.2f} sec'.format(
            #         idx + 1,
            #         all_n_nodes.min(),
            #         all_n_nodes.max(),
            #         all_n_nodes.mean(),
            #         all_n_nodes.std(),
            #         all_n_params.min(),
            #         all_n_params.max(),
            #         all_n_params.mean(),
            #         all_n_params.std(),
            #         len(primitives_back),
            #         len(op_types_back),
            #         time.time() - start),
            #         flush=True)

    with open(meta_file, 'w') as f:
        json.dump(meta_data, f)

    print('saved to %s and %s' % (h5_file, meta_file))

    print('\ndone')

    if split == 'bnfree':
        merge_eval(data_dir)  # assume bnfree was generated the last


# Merge all eval splits into one file
def merge_eval(data_dir):

    print('merging the evaluation splits into one file')

    meta_new = {}
    for split in ['val', 'test', 'wide', 'deep', 'dense', 'bnfree']:
        with open(join(data_dir, 'deepnets1m_%s_meta.json' % split), 'r') as f:
            meta_new[split] = json.load(f)[split]
            print(split, len(meta_new[split]), len(meta_new[split]['meta']), len(meta_new[split]['nets']))
    print(list(meta_new.keys()))
    with open(join(data_dir, 'deepnets1m_eval_meta.json'), 'w') as f:
        json.dump(meta_new, f)


    with h5py.File(join(data_dir, 'deepnets1m_eval.hdf5'), "w") as h5_data:
        for split in ['val', 'test', 'wide', 'deep', 'dense', 'bnfree']:
            with h5py.File(join(data_dir, 'deepnets1m_%s.hdf5' % split), "r") as data_file:
                h5_data.attrs['title'] = 'DeepNets-1M'
                group = h5_data.create_group(split)
                for i in range(len(data_file[split])):
                    A, nodes = data_file[split][str(i)]['adj'][()], data_file[split][str(i)]['nodes'][()]
                    group.create_dataset(str(i)+'/adj', data=A)
                    group.create_dataset(str(i)+'/nodes', data=nodes)
                    if i == 0:
                        print(split, len(data_file[split]), A.dtype, nodes.dtype)
    print('\ndone')


if __name__ == '__main__':
    main()
