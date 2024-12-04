## coding:utf-8
from access_db_operate import *


def get_all_sensitiveAPI(db):
    list_sensitive_funcname = []
    with open("../data/API-library function calls.txt", 'r') as file:
        for line in file:
            list_sensitive_funcname.append(line.split(',')[0])

    _dict = {}
    for func_name in list_sensitive_funcname:
        list_callee_cfgnodeID = []
        if func_name.find('main') != -1:
            list_main_func = []
            list_mainfunc_node = getFunctionNodeByName(db, func_name)

            if list_mainfunc_node != []:
                file_path = getFuncFile(db, list_mainfunc_node[0]._id)
                testID = file_path.split('/')[-2]
                for mainfunc in list_mainfunc_node:
                    list_parameters = get_parameter_by_funcid(db, mainfunc._id)

                    if list_parameters != []:
                        list_callee_cfgnodeID.append([testID, ([str(v) for v in list_parameters], str(mainfunc._id), func_name)])

                    else:
                        continue

        else:
            list_callee_id = get_calls_id(db, func_name)
            if list_callee_id == []:
                continue

            
            for _id in list_callee_id:
                cfgnode = getCFGNodeByCallee(db, _id)
                if cfgnode != None:
                    file_path = getFuncFile(db, int(cfgnode.properties['functionId']))
                    testID = file_path.split('/')[-2]
                    list_callee_cfgnodeID.append([testID, ([str(cfgnode._id)], str(cfgnode.properties['functionId']), func_name)])

        if list_callee_cfgnodeID != []:
            for _l in list_callee_cfgnodeID:
                if _l[0] in _dict.keys():
                    _dict[_l[0]].append(_l[1])
                else:
                    _dict[_l[0]] = [_l[1]]

        else:
            continue

    return _dict


def get_all_sensitiveVariable(db):
    _dict = {}
    list_pointers_node = get_variables_node(db)
    for cfgnode in list_pointers_node:
        file_path = getFuncFile(db, int(cfgnode.properties['functionId']))
        testID = file_path.split('/')[-2]
        pointer_defnode = get_def_node(db, cfgnode._id)
        pointer_name = []
        for node in pointer_defnode:
            name = node.properties['code'].replace('*', '').strip()
            if name not in pointer_name:
                pointer_name.append(name)

        if testID in _dict.keys():
            _dict[testID].append(([str(cfgnode._id)], str(cfgnode.properties['functionId']), pointer_name))
        else:
            _dict[testID] = [([str(cfgnode._id)], str(cfgnode.properties['functionId']), pointer_name)]

    return _dict


def get_all_arithmetic_expressions(db):
    _dict = {}
    list_exprstmt_node = get_exprstmt_node(db)
    for cfgnode in list_exprstmt_node:
        if cfgnode.properties['code'].find(' = ') > -1:
            code = cfgnode.properties['code'].split(' = ')[-1]
            pattern = re.compile("((?:_|[A-Za-z])\w*(?:\s(?:\+|\-|\*|\/)\s(?:_|[A-Za-z])\w*)+)")                
            result = re.search(pattern, code)
       
            if result == None:
                continue
            else:
                file_path = getFuncFile(db, int(cfgnode.properties['functionId']))
                testID = file_path.split('/')[-2]
                name = cfgnode.properties['code'].strip()

                if testID in _dict.keys():
                    _dict[testID].append(([str(cfgnode._id)], str(cfgnode.properties['functionId']), name))
                else:
                    _dict[testID] = [([str(cfgnode._id)], str(cfgnode.properties['functionId']), name)]

        else:
            code = cfgnode.properties['code']
            pattern = re.compile("(?:\s\/\s(?:_|[A-Za-z])\w*\s)")
            result = re.search(pattern, code)
            if result == None:
                continue

            else:
                file_path = getFuncFile(db, int(cfgnode.properties['functionId']))
                testID = file_path.split('/')[-2]
                name = cfgnode.properties['code'].strip()

                if testID in _dict.keys():
                    _dict[testID].append(([str(cfgnode._id)], str(cfgnode.properties['functionId']), name))
                else:
                    _dict[testID] = [([str(cfgnode._id)], str(cfgnode.properties['functionId']), name)]

    return _dict


if __name__ == '__main__':
    j = JoernSteps()
    j.connectToDatabase()

    project = "Linux"
    root_path = f"./slicing_entry_nodes/{project}/"
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    _dict = get_all_sensitiveAPI(j)
    f = open(root_path + "sensifunc_entry_nodes.pkl", 'wb')
    pickle.dump(_dict, f, True)
    f.close()
    print(_dict)

    _dict = get_all_sensitiveVariable(j)
    f = open(root_path + "sensivar_entry_nodes.pkl", 'wb')
    pickle.dump(_dict, f, True)
    f.close()
    print(_dict)
    
    _dict = get_all_arithmetic_expressions(j)
    f = open(root_path + "arithmetic_expression_entry_nodes.pkl", 'wb')
    pickle.dump(_dict, f, True)
    f.close()
	
    
