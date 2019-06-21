import os
import networkx as nx
import src.optimizer as opt


def export_file(filename_in, filename_out):
    nq, buckets, free_vars = opt.read_buckets(filename_in)
    graph_raw = opt.buckets2graph(buckets)
    nx.write_graphml(graph_raw, filename_out)


def export_all(root_dir_in, root_dir_out):

    def walktree(old_top, new_top, file_callback):
        for elem in os.listdir(old_top):
            old_path = os.path.join(old_top, elem)
            print('at {}'.format(old_path))
            if os.path.isdir(old_path):
                new_path = os.path.join(new_top, elem)
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                walktree(old_path, new_path, file_callback)
            elif os.path.isfile(old_path):
                basename = elem.split('.')[0]
                suffix = elem.split('.')[-1]
                if suffix == 'txt':
                    new_path = os.path.join(new_top, basename+'.graphml')
                    file_callback(old_path, new_path)

    walktree(root_dir_in, root_dir_out, export_file)

