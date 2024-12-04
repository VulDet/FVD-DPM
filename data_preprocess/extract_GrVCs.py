## coding:utf-8
import pandas as pd
from slice_op import *
import json
from igraph import *



def isNodeExist(g, nodeName):
    if not g.vs:
        return False
    else:
        return nodeName in g.vs['name']

def get_backward_node_line(node, num_pre=0):
    node_line = None
    predecessors = node.predecessors()
    if len(predecessors) != 0:
        num_pre += 1
        for pred in predecessors:
            if pred['location'] and (node['code'] in pred['code']):
                node_line = pred['location'].split(':')[0]
                break
        if not node_line:
            if num_pre <= 5:
                for pred in predecessors:
                    node_line = get_backward_node_line(pred, num_pre)

    return node_line

def get_forward_node_line(node, num_for=0):
    node_line = None
    successors = node.successors()
    if len(successors) != 0:
        num_for += 1
        for succ in successors:
            if succ['location'] and (succ['code'] in node['code']):
                node_line = succ['location'].split(':')[0]
                break
        if not node_line:
            if num_for <= 5:
                for succ in successors:
                    node_line = get_forward_node_line(succ, num_for)

    return node_line

def get_graph_massage(g, filepath, _dict, filename, labeled=True):
    num_nodes = len(g.vs.indices)

    node_labels = ["" for i in range(num_nodes)]
    node_codes = ["" for i in range(num_nodes)]
    node_targets = [-1] * num_nodes
    code_lines = [-1] * num_nodes
    edge_labels = []
    if filename in _dict.keys():
        vullines = _dict[filename]
        if project == 'SARD':
            vullines = list(vullines)
            vullines = [int(line) for line in vullines]
        # 给节点进行重新编号
        id = 0
        new_id = {}
        # 建立node 的label和 node的id之间的映射
        for node in g.vs:
            new_id[node['name']] = id
            id += 1

        if labeled == True:
            # hsh = list(g.vs)
            for node in g.vs:  # 形如 ('7', {'type': 'C', 'label': '7'})
                # labels[7] = 8
                node_target = 0
                code_line = -1
                ori_id = node['name']
                now_id = new_id[ori_id]
                if node['type']:
                    node_labels[now_id] = node['type']
                if node['location']:
                    match_line = int(node['location'].split(':')[0])
                    code_line = match_line
                    if match_line in vullines:
                        node_target = 1
                else:
                    match_line = get_backward_node_line(node)
                    if match_line:
                        match_line = int(match_line)
                        code_line = match_line
                        if match_line in vullines:
                            node_target = 1
                    else:
                        match_line = get_forward_node_line(node)
                        if match_line:
                            match_line = int(match_line)
                            code_line = match_line
                            if match_line in vullines:
                                node_target = 1

                node_targets[now_id] = node_target
                node_codes[now_id] = node['code']
                code_lines[now_id] = code_line
    else:
        # 给节点进行重新编号
        id = 0
        new_id = {}
        # 建立node 的label和 node的id之间的映射
        for node in g.vs:
            new_id[node['name']] = id
            id += 1

        if labeled == True:
            hsh = list(g.vs)
            for node in g.vs:  # 形如 ('7', {'type': 'C', 'label': '7'})
                # labels[7] = 8
                node_target = 0
                code_line = -1
                ori_id = node['name']
                now_id = new_id[ori_id]
                if node['type']:
                    node_labels[now_id] = node['type']
                if node['location']:
                    match_line = int(node['location'].split(':')[0])
                    code_line = match_line
                else:
                    match_line = get_backward_node_line(node)
                    if match_line:
                        match_line = int(match_line)
                        code_line = match_line
                    else:
                        match_line = get_forward_node_line(node)
                        if match_line:
                            match_line = int(match_line)
                            code_line = match_line
                node_codes[now_id] = node['code']
                code_lines[now_id] = code_line
        node_targets = [0] * num_nodes
    edges = []
    # edges 的每个元素形如： [7, 4]
    for edge in g.es:  # 形如 ('7', '4', {'valence': 2, 'id': '6'})
        edges.append([new_id[edge.source_vertex['name']], new_id[edge.target_vertex['name']]])
        if edge['var']:
            edge_labels.append(edge['var'])
        else:
            edge_labels.append('')
    # 返回一个图的边矩阵, 以及顶点向量；
    # 这是json的组成形式
    nodes = list(new_id.values())
    return nodes, edges, node_labels, node_codes, edge_labels, node_targets, code_lines


def get_slice_graph(pdg, list_result):
    g = Graph(directed=True)
    list_nodes_name = []
    for node in list_result:
        node_name = node['name']
        list_nodes_name.append(node_name)

    for edge in pdg.es:
        start_node = edge.source_vertex
        end_node = edge.target_vertex
        if start_node['name'] in list_nodes_name and end_node['name'] in list_nodes_name:
            if len(start_node['code']) == 0 or len(end_node['code']) == 0:
                continue
            if isNodeExist(g, start_node['name']) == False:
                node_prop = {'code': start_node['code'], 'type': start_node['type'],
                             'location': start_node['location'], 'functionId': str(start_node['functionId'])}
                g.add_vertex(start_node['name'], **node_prop)
            if isNodeExist(g, end_node['name']) == False:
                node_prop = {'code': end_node['code'], 'type': end_node['type'],
                             'location': end_node['location'], 'functionId': str(end_node['functionId'])}
                g.add_vertex(end_node['name'], **node_prop)
            edge_prop = {'var': edge['var']}
            g.add_edge(start_node['name'], end_node['name'], **edge_prop)


    if len(g.vs.indices)==0:
        return g

    return g


def get_slice_file_sequence(store_filepath, list_result, count, func_name, startline, filepath_all):
    list_for_line = []
    statement_line = 0
    vulnline_row = 0
    list_write2file = []

    for node in list_result:
        if node['location'] == None:
            continue
        if node['type'] == 'Function':
            try:
                f2 = open(node['filepath'], 'r')
            except:
                continue
            content = f2.readlines()
            f2.close()
            raw = int(node['location'].split(':')[0]) - 1
            code = content[raw].strip()

            new_code = ""
            if code.find("#define") != -1:
                filter_result = code.find('/*')
                if filter_result != -1:
                    code = code[:filter_result]
                list_write2file.append(code + ' ' + str(raw + 1) + '\n')
                continue

            while (len(code) >= 1 and code[-1] != ')' and code[-1] != '{'):
                if code.find('{') != -1:
                    index = code.index('{')
                    new_code += code[:index].strip()
                    filter_result = new_code.find('/*')
                    if filter_result != -1:
                        new_code = new_code[:filter_result]
                    list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')
                    break

                else:
                    new_code += code + ' '  # + '\n'
                    filter_result = new_code.find('/*')
                    if filter_result != -1:
                        new_code = new_code[:filter_result]
                    raw += 1
                    code = content[raw].strip()

            else:
                new_code += code
                new_code = new_code.strip()
                if new_code[-1] == '{':
                    new_code = new_code[:-1].strip()
                    list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')

                else:
                    filter_result = new_code.find('/*')
                    if filter_result != -1:
                        new_code = new_code[:filter_result]
                    list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')


        elif node['type'] == 'Condition':
            raw = int(node['location'].split(':')[0]) - 1
            if raw in list_for_line:
                continue
            else:
                try:
                    f2 = open(node['filepath'], 'r')
                except:
                    continue
                content = f2.readlines()
                f2.close()
                code = content[raw].strip()
                pattern = re.compile("(?:if|while|for|switch)")

                res = re.search(pattern, code)
                if res == None:
                    raw = raw - 1
                    code = content[raw].strip()
                    new_code = ""

                    while (code[-1] != ')' and code[-1] != '{') and raw < len(content):
                        if code.find('{') != -1:
                            index = code.index('{')
                            new_code += code[:index].strip()
                            filter_result = new_code.find('/*')
                            if filter_result != -1:
                                new_code = new_code[:filter_result]
                            list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')
                            list_for_line.append(raw)
                            break

                        else:
                            new_code += code + ' '  # + '\n'
                            filter_result = new_code.find('/*')
                            if filter_result != -1:
                                new_code = new_code[:filter_result]
                            list_for_line.append(raw)
                            raw += 1
                            code = content[raw].strip()
                            if len(code) == 0:
                                break

                    else:
                        new_code += code
                        new_code = new_code.strip()
                        if new_code[-1] == '{':
                            new_code = new_code[:-1].strip()
                            list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')
                            list_for_line.append(raw)

                        else:
                            list_for_line.append(raw)
                            filter_result = new_code.find('/*')
                            if filter_result != -1:
                                new_code = new_code[:filter_result]
                            list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')

                else:
                    res = res.group()
                    if res == '':
                        print(filepath_all + ' ' + func_name + " error!")
                        exit()

                    elif res != 'for':
                        new_code = res + ' ( ' + node['code'] + ' ) '
                        filter_result = new_code.find('/*')
                        if filter_result != -1:
                            new_code = new_code[:filter_result]
                        list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')

                    else:
                        new_code = ""
                        if code.find(' for ') != -1:
                            code = 'for ' + code.split(' for ')[1]

                        while code != '' and code[-1] != ')' and code[-1] != '{' and raw < len(content) - 1:
                            if code.find('{') != -1:
                                index = code.index('{')
                                new_code += code[:index].strip()
                                filter_result = new_code.find('/*')
                                if filter_result != -1:
                                    new_code = new_code[:filter_result]
                                list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')
                                list_for_line.append(raw)
                                break

                            elif code[-1] == ';' and code[:-1].count(';') >= 2:
                                new_code += code
                                list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')
                                list_for_line.append(raw)
                                break

                            else:
                                new_code += code + ' '  # + '\n'
                                filter_result = new_code.find('/*')
                                if filter_result != -1:
                                    new_code = new_code[:filter_result]
                                list_for_line.append(raw)
                                raw += 1
                                code = content[raw].strip()


                        else:
                            new_code += code
                            new_code = new_code.strip()
                            if new_code[-1] == '{':
                                new_code = new_code[:-1].strip()
                                list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')
                                list_for_line.append(raw)

                            else:
                                filter_result = new_code.find('/*')
                                if filter_result != -1:
                                    new_code = new_code[:filter_result]
                                list_for_line.append(raw)
                                list_write2file.append(new_code + ' ' + str(raw + 1) + '\n')

        elif node['type'] == 'Label':
            try:
                f2 = open(node['filepath'], 'r')
            except:
                continue
            content = f2.readlines()
            f2.close()
            raw = int(node['location'].split(':')[0]) - 1
            code = content[raw].strip()
            filter_result = code.find('/*')
            if filter_result != -1:
                code = code[:filter_result]
            list_write2file.append(code + ' ' + str(raw + 1) + '\n')

        elif node['type'] == 'ForInit' or node['type'] == 'IncDecOp':
            continue

        elif node['type'] == 'Parameter':
            if list_result[0]['type'] != 'Function':
                row = node['location'].split(':')[0]
                code = node['code']
                filter_result = code.find('/*')
                if filter_result != -1:
                    code = code[:filter_result]
                list_write2file.append(code.strip() + ' ' + str(row) + '\n')

            else:
                continue

        elif node['type'] == 'IdentifierDeclStatement':
            if node['code'].strip().split(' ')[0] == "undef":
                try:
                    f2 = open(node['filepath'], 'r')
                except:
                    continue
                content = f2.readlines()
                f2.close()
                raw = int(node['location'].split(':')[0]) - 1
                code1 = content[raw].strip()
                filter_result1 = code1.find('/*')
                if filter_result1 != -1:
                    code1 = code1[:filter_result1]
                list_code2 = node['code'].strip().split(' ')
                i = 0
                while i < len(list_code2):
                    if code1.find(list_code2[i]) != -1:
                        del list_code2[i]
                    else:
                        break
                code2 = ' '.join(list_code2)
                filter_result2 = code2.find('/*')
                if filter_result2 != -1:
                    code2 = code2[:filter_result2]

                list_write2file.append(code1 + ' ' + str(raw + 1) + '\n' + code2 + ' ' + str(raw + 2) + '\n')

            else:
                code = node['code']
                filter_result = code.find('/*')
                if filter_result != -1:
                    code = code[:filter_result]
                list_write2file.append(code + ' ' + node['location'].split(':')[0] + '\n')

        elif node['type'] == 'ExpressionStatement':
            row = int(node['location'].split(':')[0]) - 1
            if row in list_for_line:
                continue

            if node['code'] in ['\n', '\t', ' ', '']:
                continue

            elif node['code'].strip()[-1] != ';':
                code = node['code']
                filter_result = code.find('/*')
                if filter_result != -1:
                    code = code[:filter_result]
                list_write2file.append(code + '; ' + str(row + 1) + '\n')

            else:
                code = node['code']
                filter_result = code.find('/*')
                if filter_result != -1:
                    code = code[:filter_result]
                list_write2file.append(code + ' ' + str(row + 1) + '\n')


        elif node['type'] == "Statement":
            row = node['location'].split(':')[0]
            code = node['code']
            filter_result = code.find('/*')
            if filter_result != -1:
                code = code[:filter_result]
            list_write2file.append(code + ' ' + str(row) + '\n')

        else:
            if node['location'] == None:
                continue
            try:
                f2 = open(node['filepath'], 'r')
            except:
                continue
            content = f2.readlines()
            f2.close()
            row = int(node['location'].split(':')[0]) - 1
            code = content[row].strip()
            if row in list_for_line:
                continue

            else:
                if node['type'] == 'CompoundStatement':
                    continue
                filter_result = code.find('/*')
                if filter_result != -1:
                    code = code[:filter_result]
                list_write2file.append(code + ' ' + str(row + 1) + '\n')

    return list_write2file


def program_slice(pdg, startnodesID, slicetype, testID): # process startnodes as a list, because main func has many different arguments
    list_startnodes = []
    if pdg == False or pdg == None:
        return [], [], []
        
    for node in pdg.vs:
        if node['name'] in startnodesID:
            list_startnodes.append(node)

    if list_startnodes == []:
        return [], [], []

    if slicetype == 0:  # backwords
        start_line = list_startnodes[0]['location'].split(':')[0]
        start_name = list_startnodes[0]['name']
        startline_path = list_startnodes[0]['filepath']
        results_back = program_slice_backwards(pdg, list_startnodes)

        not_scan_func_list = []
        results_back, temp = process_cross_func(results_back, testID, 1, results_back, not_scan_func_list)


        return [results_back], start_line, startline_path

    elif slicetype == 1:  # forwords
        print("start extract forword dataflow!")
        print(list_startnodes, startnodesID)
        start_line = list_startnodes[0]['location'].split(':')[0]
        start_name = list_startnodes[0]['name']
        startline_path = list_startnodes[0]['filepath']
        results_for = program_slice_forward(pdg, list_startnodes)

        not_scan_func_list = []
        results_for, temp = process_cross_func(results_for, testID, 1, results_for, not_scan_func_list)

        return [results_for], start_line, startline_path

    else:  # bi_direction
        print("start extract forwards and backwords dataflow!")
        try:
            start_line = list_startnodes[0]['location'].split(':')[0]
        except:
            start_line = '0'

        start_name = list_startnodes[0]['name']
        startline_path = list_startnodes[0]['filepath']
        results_back = program_slice_backwards(pdg, list_startnodes)  # results_back is a list of nodes
        results_for = program_slice_forward(pdg, list_startnodes)

        _list_name = []
        for node_back in results_back:
            _list_name.append(node_back['name'])

        for node_for in results_for:
            if node_for['name'] in _list_name:
                continue
            else:
                results_back.append(node_for)

        results_back = sortedNodesByLoc(results_back)
       
        iter_times = 0
        start_list = [[results_back, iter_times]]
        i = 0
        not_scan_func_list = []
        list_cross_func_back, not_scan_func_list = process_crossfuncs_back_byfirstnode(start_list, testID, i, not_scan_func_list)
        list_results_back = [l[0] for l in list_cross_func_back]
        #
        all_result = []
        for i, results_back in enumerate(list_results_back):
            index = 1
            for a_node in results_back:
                if a_node['name'] == start_name:
                    break
                else:
                    index += 1

            list_to_crossfunc_back = results_back[:index]
            list_to_crossfunc_for = results_back[index:]

            list_to_crossfunc_back, temp = process_cross_func(list_to_crossfunc_back, testID, 0, list_to_crossfunc_back, not_scan_func_list)

            list_to_crossfunc_for, temp = process_cross_func(list_to_crossfunc_for, testID, 1, list_to_crossfunc_for, not_scan_func_list)

            all_result.append(list_to_crossfunc_back + list_to_crossfunc_for)


        return all_result, start_line, startline_path


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def print_graph(g):
    for edge in g.es:
        start_node = edge.source_vertex
        end_node = edge.target_vertex
        print(start_node)
        print(end_node)
        print(edge['var'])
        print("-------------------------")


def preprocess_graph(pdg):
    # 为PDG添加code sequence 边
    node_names = []
    line_numbers = []
    edges = []
    for node in pdg.vs:
        try:
            line_number = int(node['location'].split(':')[0])
        except:
            continue
        node_name = node['name']
        line_numbers.append(line_number)
        node_names.append(node_name)
    assert len(line_numbers)==len(node_names)
    for edge in pdg.es:
        src_node = edge.source_vertex['name']
        dst_node = edge.target_vertex['name']
        edges.append([src_node, dst_node])
    sorted_index = [i[0] for i in sorted(enumerate(line_numbers), key=lambda x: x[1])]
    if len(sorted_index)==0:
        return pdg
    pre_index = sorted_index[0]
    for i in range(1, len(sorted_index)):
        index = sorted_index[i]
        if line_numbers[index] == line_numbers[pre_index] +1:
            edge = [node_names[pre_index], node_names[index]]
            if edge not in edges:
                edge_prop = {'var': 'code_sequence'}
                pdg.add_edge(node_names[pre_index], node_names[index], **edge_prop)
        pre_index = index

    return pdg


def sensifunc_slice(project, _dict):
    count = 1
    source_filepath = f"../data/source_code/{project}/"
    store_filepath = f"./GrVCs/{project}/"
    f = open(f"./slicing_entry_nodes/{project}/sensifunc_entry_nodes.pkl", 'rb')
    save_data = pd.DataFrame(columns=['filename', 'nodes', 'edges', 'nodes_label', 'nodes_codes', 'edges_label', 'code_lines', 'vul_type', 'node_target', 'target'])
    dict_unsliced_sensifunc = pickle.load(f)
    f.close()
    for key in dict_unsliced_sensifunc.keys():  # key is testID
        for _t in dict_unsliced_sensifunc[key]:
            list_sensitive_funcid = _t[0]
            pdg_funcid = _t[1]
            sensitive_funcname = _t[2]

            if sensitive_funcname.find("main") != -1:
                continue  # todo
            else:
                slice_dir = 2
                if project == 'SARD':
                    pdg = getFuncPDGById(project, pdg_funcid)
                else:
                    pdg = getFuncPDGById(key, pdg_funcid)

                if pdg == False:
                    print('error')
                    exit()
                if not pdg:
                    continue

                pdg = preprocess_graph(pdg)

                list_code, startline, startline_path = program_slice(pdg, list_sensitive_funcid, slice_dir, key)
                if list_code == []:
                    continue
                else:
                    filename = startline_path.split('/')[-1]
                    for _list in list_code:
                        list_write2file = get_slice_file_sequence(store_filepath, _list, count, sensitive_funcname, startline, startline_path)

                        label = 0
                        if filename not in _dict.keys():
                            label = 0
                        else:
                            if len(_dict[filename]) == 0:
                                label = 0
                            for sentence in list_write2file:
                                if (is_number(sentence.split(' ')[-1])) is False:
                                    continue
                                if project == 'SARD':
                                    linenum = sentence.split(' ')[-1].strip()
                                else:
                                    linenum = int(sentence.split(' ')[-1])
                                vullines = _dict[filename]
                                if linenum in vullines:
                                    label = 1
                                    break
                        slice_graph = get_slice_graph(pdg, _list)
                        nodes, edges, nodes_label, nodes_codes, edges_label, node_target, code_lines = get_graph_massage(slice_graph, source_filepath, _dict, filename, labeled=True)
                        if len(nodes) == 0:
                            continue
                        series = pd.Series({'filename': filename, 'nodes': nodes, 'edges': edges, 'nodes_label': nodes_label, 'nodes_codes': nodes_codes, 'edges_label': edges_label, 'code_lines': code_lines, 'vul_type':'sensifunc', 'node_target':node_target, 'target': label})
                        save_data = save_data.append(series, ignore_index=True)
                        count += 1
                        print(count)
    save_data.to_json(store_filepath + "sensifunc_GrVCs.json")


def sensivar_slice(project, _dict):
    count = 1
    source_filepath = f"../data/source_code/{project}/"
    store_filepath = f"./GrVCs/{project}/"
    f1 = open(f"./slicing_entry_nodes/{project}/pointer_use_entry_nodes.pkl", 'rb')
    pointer_variables = pickle.load(f1)
    f1.close()
    f2 = open(f"./slicing_entry_nodes/{project}/array_use_entry_nodes.pkl", 'rb')
    array_variables = pickle.load(f2)
    f2.close()
    dict_unsliced_variables = {'Linux': pointer_variables['Linux']+array_variables['Linux']}

    save_data = pd.DataFrame(columns=['filename', 'nodes', 'edges', 'nodes_label', 'nodes_codes', 'edges_label', 'code_lines', 'vul_type', 'node_target', 'target'])

    for key in dict_unsliced_variables.keys():  # key is testID
        for _t in dict_unsliced_variables[key]:
            list_variables_funcid = _t[0]
            pdg_funcid = _t[1]
            print(key, pdg_funcid)
            try:
                variables_name = str(_t[2][0])
            except:
                continue

            slice_dir = 2
            if project == 'SARD':
                pdg = getFuncPDGById(project, pdg_funcid)
            else:
                pdg = getFuncPDGById(key, pdg_funcid)

            if pdg == False:
                print('error')
                exit()
            if not pdg:
                continue
            if len(pdg.vs.indices) >= 15:
                continue
            pdg = preprocess_graph(pdg)

            list_code, startline, startline_path = program_slice(pdg, list_variables_funcid, slice_dir, key)
            if list_code == []:
                fout = open("error.txt", 'a')
                fout.write(variables_name + ' ' + str(list_variables_funcid) + ' found nothing! \n')
                fout.close()
            else:
                filename = startline_path.split('/')[-1]
                for _list in list_code:
                    list_write2file = get_slice_file_sequence(store_filepath, _list, count, variables_name, startline, startline_path)

                    label = 0
                    if filename not in _dict.keys():
                        label = 0
                    else:
                        if len(_dict[filename]) == 0:
                            label = 0
                        for sentence in list_write2file:
                            if (is_number(sentence.split(' ')[-1])) is False:
                                continue
                            if project == 'SARD':
                                linenum = sentence.split(' ')[-1].strip()
                            else:
                                linenum = int(sentence.split(' ')[-1])
                            vullines = _dict[filename]
                            if linenum in vullines:
                                label = 1
                                break
                    slice_graph = get_slice_graph(pdg, _list)
                    nodes, edges, nodes_label, nodes_codes, edges_label, node_target, code_lines = get_graph_massage(slice_graph, source_filepath, _dict, filename, labeled=True)
                    series = pd.Series({'filename': filename, 'nodes': nodes, 'edges': edges, 'nodes_label': nodes_label, 'nodes_codes': nodes_codes, 'edges_label': edges_label, 'code_lines': code_lines, 'vul_type':'sensivar', 'node_target':node_target, 'target': label})
                    if len(nodes) == 0:
                        continue
                    save_data = save_data.append(series, ignore_index=True)
                    count += 1
                    print(count)
    save_data.to_json(store_filepath + "sensivar_GrVCs.json")


def expression_slice(project, _dict):
    count = 1
    source_filepath = f"../data/source_code/{project}/"
    store_filepath = f"./GrVCs/{project}/"
    f = open(f"./slicing_entry_nodes/{project}/arithmetic_expression_entry_nodes.pkl", 'rb')
    save_data = pd.DataFrame(columns=['filename', 'nodes', 'edges', 'nodes_label', 'nodes_codes', 'edges_label', 'code_lines', 'vul_type', 'node_target', 'target'])
    dict_unsliced_expr = pickle.load(f)
    f.close()

    for key in dict_unsliced_expr.keys(): #key is testID
        for _t in dict_unsliced_expr[key]:
            list_expr_funcid = _t[0]
            pdg_funcid = _t[1]
            print(pdg_funcid)
            expr_name = str(_t[2])

            slice_dir = 2
            if project == 'SARD':
                pdg = getFuncPDGById(project, pdg_funcid)
            else:
                pdg = getFuncPDGById(key, pdg_funcid)
            if pdg == False:
                print('error')
                exit()
            if not pdg:
                continue
            pdg = preprocess_graph(pdg)
            list_code, startline, startline_path = program_slice(pdg, list_expr_funcid, slice_dir, key)
            if list_code == []:
                fout = open("error.txt", 'a')
                fout.write(expr_name + ' ' + str(list_expr_funcid) + ' found nothing! \n')
                fout.close()
            else:
                filename = startline_path.split('/')[-1]
                for _list in list_code:
                    list_write2file = get_slice_file_sequence(store_filepath, _list, count, expr_name, startline, startline_path)

                    label = 0
                    if filename not in _dict.keys():
                        label = 0
                    else:
                        if len(_dict[filename]) == 0:
                            label = 0
                        for sentence in list_write2file:
                            if (is_number(sentence.split(' ')[-1])) is False:
                                continue
                            if project == 'SARD':
                                linenum = sentence.split(' ')[-1].strip()
                            else:
                                linenum = int(sentence.split(' ')[-1])
                            vullines = _dict[filename]
                            if linenum in vullines:
                                label = 1
                                break
                    slice_graph = get_slice_graph(pdg, _list)
                    nodes, edges, nodes_label, nodes_codes, edges_label, node_target, code_lines = get_graph_massage(slice_graph, source_filepath, _dict, filename, labeled=True)
                    if len(nodes) == 0:
                        continue
                    series = pd.Series({'filename': filename, 'nodes': nodes, 'edges': edges, 'nodes_label': nodes_label, 'nodes_codes': nodes_codes, 'edges_label': edges_label, 'code_lines': code_lines, 'vul_type':'expression', 'node_target':node_target, 'target': label})
                    save_data = save_data.append(series, ignore_index=True)
                    count += 1
                    print(count)
    save_data.to_json(store_filepath + "expression_GrVCs.json")


if __name__ == "__main__":
    project = 'Linux'
    with open(f'../data/vul_line_number/{project}_vul_linenumber.pkl', 'rb') as f:
        _dict = pickle.load(f)
    f.close()

    # sensifunc_slice(project, _dict)
    sensivar_slice(project, _dict)
    expression_slice(project, _dict)

    # 合并多个json文件
    file1 = pd.read_json(open(f'./GrVCs/{project}/sensifunc_GrVCs.json'))
    file2 = pd.read_json(open(f'./GrVCs/{project}/sensivar_GrVCs.json'))
    file3 = pd.read_json(open(f'./GrVCs/{project}/expression_GrVCs.json'))
    frames_input = [file1, file2, file3]
    data_input = pd.concat(frames_input, ignore_index=True, keys=['filename', 'nodes', 'edges', 'nodes_label', 'nodes_codes', 'edges_label', 'code_lines', 'vul_type', 'node_target', 'target'])
    data_input.to_json(f"./GrVCs/{project}/{project}_GrVCs.json")


