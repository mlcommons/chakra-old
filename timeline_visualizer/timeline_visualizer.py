#!/usr/bin/env python3

import argparse

def is_local_mem_node(node_name):
    if ("MEM_LOAD_NODE" in node_name)\
            and ("LOCAL_MEMORY" in node_name):
        return True
    elif ("MEM_STORE_NODE" in node_name)\
            and ("LOCAL_MEMORY" in node_name):
        return True
    else:
        return False

def is_remote_mem_node(node_name):
    if ("MEM_LOAD_NODE" in node_name)\
            and ("REMOTE_MEMORY" in node_name):
        return True
    elif ("MEM_STORE_NODE" in node_name)\
            and ("REMOTE_MEMORY" in node_name):
        return True
    else:
        return False

def is_comp_node(node_name):
    if "COMP_NODE" in node_name:
        return True
    else:
        return False

def is_comm_node(node_name):
    if ("COMM_SEND_NODE" in node_name)\
            or ("COMM_RECV_NODE" in node_name)\
            or ("COMM_COLL_NODE" in node_name):
        return True
    else:
        return False

def get_tid(node_name):
    if is_local_mem_node(node_name):
        return 1
    elif is_remote_mem_node(node_name):
        return 2
    elif is_comp_node(node_name):
        return 3
    elif is_comm_node(node_name):
        return 4
    else:
        print("Unsupported node type, node_name={}".format(node_name))
        assert False

def main():
    parser = argparse.ArgumentParser(
            description="Timeline Visualizer"
    )
    parser.add_argument(
            "--input_filename",
            type=str,
            default=None,
            required=True,
            help="Input timeline filename"
    )
    parser.add_argument(
            "--output_filename",
            type=str,
            default=None,
            required=True,
            help="Output trace filename"
    )
    parser.add_argument(
            "--num_npus",
            type=int,
            default=None,
            required=True,
            help="Number of NPUs in a system"
    )
    parser.add_argument(
            "--npu_frequency",
            type=int,
            default=None,
            required=True,
            help="NPU frequency in MHz"
    )
    args = parser.parse_args()

    trace_dict = {}
    for i in range(args.num_npus):
        trace_dict.update({i: {}})

    trace_template = """
    {
        "meta_user": "aras",
        "traceEvents": [
            %s
        ],
        "meta_user": "aras",
        "meta_cpu_count": %d
    } """

    trace_events = ""
    with open(args.input_filename, "r") as f:
        for line in f:
            if ("issue" in line) or ("callback" in line):
                cols = line.strip().split(",")
                trace_type = cols[0]
                npu_id = int(cols[1].split("=")[1])
                curr_cycle = int(cols[2].split("=")[1])
                node_id = cols[3].split("=")[1]
                node_name = cols[4].split("=")[1]

                if trace_type == "issue":
                    trace_dict[npu_id].update({node_id: [node_name, curr_cycle]})

                elif trace_type == "callback":
                    node_name = trace_dict[npu_id][node_id][0]
                    issued_cycle = trace_dict[npu_id][node_id][1]
                    duration_in_cycles = curr_cycle - issued_cycle

                    issued_ms = (issued_cycle / args.npu_frequency) / 1000
                    duration_in_ms = duration_in_cycles / (args.npu_frequency * 1000)
                    tid =  get_tid(node_name)

                    trace_events +=\
                            """\t{"pid": %d, "tid": %d, "ts": %f, "dur": %f, "ph": "X", "name": "%s", "args": {"ms": %f }},\n"""\
                            % (npu_id, tid, issued_ms, duration_in_ms, node_name, duration_in_ms)
                    del trace_dict[npu_id][node_id]

                else:
                    assert False

    with open(args.output_filename, "w") as f:
        f.write(trace_template % (trace_events[:-2], args.num_npus))

if __name__ == "__main__":
    main()
