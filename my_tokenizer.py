import re


other_keyword = ['auto', 'explicit', 'switch', 'case', 'default', 'do', 'for', 'while', 'if', 'else', 'break',
                  'continue', 'goto', 'volatile', 'enum', 'export', 'public', 'protected', 'private',
                  'template', 'struct', 'class', 'union', 'mutable', 'override', 'catch', 'throw', 'try', 'new',
                  'delete', 'register', 'typename', 'using', 'namespace', 'typedef', 'return', 'sizeof', 'typeid',
                  'this', 'asm', 'const_cast', 'dynamic_cast', 'reinterpret_cast', 'static_cast', 'include', '\"',
                  '\'', ':', ',', '.', '[', ']']
type_keyword = ['bool', 'char', 'wchar_t', 'int', 'double', 'float', 'short', 'long', 'signed', 'unsigned', 'void', 'const']
embellish_keyword = ['static', 'virtual', 'extern', 'inline', 'friend']
operator_keyword = ['operator']
operator = ['=', '+', '-', '/', '|', '^', '&', '!', '~', '%', '#', '?',
            '<<', '&&', '||', '++', '--',
            '<=', '>=', '&=', '|=', '+=', '-=', '==', '!=',  '*=', '/=', '%=',  '^=',
            '<<=', '>>=', '->']
left_circle_bracket = ['(']
right_circle_bracket = [')']
left_angle_bracket = ['<']
right_angle_bracket = ['>']
double_right_angle_bracket = ['>>']
left_brace_bracket = ['{']
right_brace_bracket = ['}']
asterisk = ['*']
field_operator = ['::']
semicolon = [';']


label_dict = {'other_keyword': other_keyword, 'type_keyword': type_keyword, 'embellish_keyword': embellish_keyword,
              'operator_keyword': operator_keyword, 'operator': operator, 'asterisk': asterisk, 'field_operator': field_operator,
              'left_circle_bracket': left_circle_bracket, 'right_circle_bracket': right_circle_bracket,
              'left_angle_bracket': left_angle_bracket, 'right_angle_bracket': right_angle_bracket, 'double_right_angle_bracket': double_right_angle_bracket,
              'left_brace_bracket': left_brace_bracket, 'right_brace_bracket': right_brace_bracket, 'semicolon': semicolon}


token_dict = [['embellish_keyword'], ['type_keyword'], ['identifier'], ['asterisk'], ['operator_keyword'], ['field_operator'], ['operator'],
              ['left_circle_bracket'], ['right_circle_bracket'], ['left_angle_bracket'], ['right_angle_bracket'], ['right_angle_bracket'],
              ['double_right_angle_bracket'], ['double_right_angle_bracket'],
              ['left_brace_bracket'], ['right_brace_bracket'], ['right_brace_bracket'], ['semicolon'], ['const', 'other_keyword']]


def zero(n):
    return n == 0


def free(n):
    return True


N1 = [free for i in range(len(token_dict))]
N1[10] = N1[12] = zero
N2 = [free for i in range(len(token_dict))]
N2[15] = zero


                      # 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
syntex_state_tabel = [[ 1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 0
                      [ 1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 1
                      [ 1,  2,  7,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 2
                      [ 1,  2,  7,  6,  9,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 3
                      [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  4,  5,  4,  4,  4,  4,  4,  0], # 4
                      [ 1,  2,  7,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 5
                      [ 1,  2,  7,  6,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 6
                      [ 1,  2,  7,  6,  0,  8,  0, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 7
                      [ 1,  2,  7,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 8
                      [ 1,  2,  3, 10,  0,  0, 10,  0,  0,  0,  0,  0, 10, 10,  0,  0,  0,  0,  0], # 9
                      [ 1,  2,  3,  0,  0,  0,  0, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 10
                      [11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11], # 11
                      [ 1, 14, 14, 14,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0, 14], # 12
                      [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16, 13, 13, 13], # 13
                      [ 1, 14, 14, 14,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0], # 14
                      [ 1, 14, 14, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0, 14]] # 15


def flat(list_2d):
    list_1d = []
    line_id = []
    for i, line in enumerate(list_2d):
        list_1d.extend(line)
        line_id.extend([i for j in range(len(line))])
    return list_1d, line_id


def recover(list_1d, line_id):
    list_2d = []
    if len(line_id) != 0:
        max_line_id = max(line_id)
        lost_line = [i for i in range(max_line_id) if i not in line_id]
        j = 0
        for i in range(max_line_id + 1):
            line = ''
            if i not in lost_line:
                while (j < len(line_id)) and (line_id[j] == i):
                    line += list_1d[j]
                    j += 1
            list_2d.append(line)
    return list_2d



def T(c, n1, n2):
    for i in range(len(token_dict)):
        if (c in token_dict[i]) and N1[i](n1) and N2[i](n2):
            return i
    return -1


def is_number(token):
    try:
        if token == 'NaN':
            return False
        float(token)
        return True
    except ValueError:
        return False


def get_label_of_tokens(tokens):
    labels = []
    for token in tokens:
        flag = False
        for key, val in zip(label_dict.keys(), label_dict.values()):
            if token in val:
                labels.append(key)
                flag = True
                break
        if not flag:
            if is_number(token):
                labels.append('const')
            else:
                labels.append('identifier')
    return labels


def extract_functions(tokens):
    labels = [get_label_of_tokens(statement) for statement in tokens]
    labels, line_id = flat(labels)
    tokens, line_id = flat(tokens)
    s_list = []
    e_list = []
    name_list = []
    s = 0
    name = ''
    n1, n2 = 0, 0
    i = 0
    state = 0
    while i < len(labels):
        label = labels[i]
        token = tokens[i]
        if label == 'left_angle_bracket' and (state == 3 or state == 4):
            n1 += 1
        elif label == 'right_angle_bracket' and (state == 4):
            n1 -= 1
        elif label == 'double_right_angle_bracket' and (state == 4):
            n1 -= 2
        elif label == 'left_brace_bracket' and (state == 12 or state == 13 or state == 15):
            n2 += 1
        elif label == 'right_brace_bracket' and (state == 13):
            n2 -= 1
        state = syntex_state_tabel[state][T(label, n1, n2)]
        i += 1
        if state == len(syntex_state_tabel):
            e = i
            s_list.append(s)
            e_list.append(e)
            name_list.append(name)
            s = e
            state = 0
        elif state == 0:
            s = i
        elif state == 7 or state == 9:
            name = token
    function_line_range = []
    for s, e in zip(s_list, e_list):
        s_id, e_id = line_id[s], line_id[e-1]
        # print('line {} to line {}: {}'.format(s_id + 1, e_id + 1, tokens[s_id:e_id + 1]))
        function_line_range.append([s_id + 1, e_id + 1])
    return function_line_range, name_list


character_dict = [['+'], ['-'], ['>'], ['<'], ['&'], ['|'], ['='], ['.'], ['!', '*', '/', '%', '^'],
                 ['(', ')', '[', ']', '{', '}', '?', ',', '~'], [';'], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                 ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  '_'], ['\"', '\''], [' ', '\t', '\n'], [':']]


                      #  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
lexical_state_tabel = [[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 31, 34, 35], # 0
                       [14,  0,  0,  0,  0,  0, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 1
                       [ 0, 16, 17,  0,  0,  0, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 2
                       [ 0,  0, 19,  0,  0,  0, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 3
                       [ 0,  0,  0, 21,  0,  0, 22,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 4
                       [ 0,  0,  0,  0, 23,  0, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 5
                       [ 0,  0,  0,  0,  0, 25, 26,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 6
                       [ 0,  0,  0,  0,  0,  0, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 7
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0], # 8
                       [ 0,  0,  0,  0,  0,  0, 28,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 9
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 10
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 11
                       [ 0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  0, 12, 12,  0,  0,  0], # 12
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 13, 13,  0,  0,  0], # 13
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 14
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 15
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 16
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 17
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 18
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 19
                       [ 0,  0,  0,  0,  0,  0, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 20
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 21
                       [ 0,  0,  0,  0,  0,  0, 30,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 22
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 23
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 24
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 25
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 26
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 27
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 28
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 29
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 30
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 31
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 33,  0,  0], # 32
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # 33
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 34,  0], # 34
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 36], # 35
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]] # 36


def T(c):
    for i, item in enumerate(character_dict):
        if c in item:
            return i
    return -1


def remove_comment_and_marco_in_single_line(line):
    if len(line) == 0:
        return line
    if line[-1] == '\n':
        line = line[:-1]
    line.strip()
    regex = ['\".*\"', '//.*', '#.*', '\'.?\'']
    replace = ['\"\"', '', '', '\'\'']
    for rgx, rpe in zip(regex, replace):
        line = re.sub(rgx, rpe, line)
    return line


def remove_comment_in_code(code):
    code, line_id = flat(code)
    s_list = []
    e_list = []
    count = 0
    for i, [token1, token2] in enumerate(zip(code[:-1], code[1:])):
        if token1 == '/' and token2 == '*':
            if count == 0:
                s_list.append(i)
            count += 1
        elif token1 == '*' and token2 == '/':
            count -= 1
            if count == 0:
                e_list.append(i+1)
    s_list = list(reversed(s_list))
    e_list = list(reversed(e_list))
    for s, e in zip(s_list, e_list):
        del code[s: e+1]
        del line_id[s: e+1]
    code = recover(code, line_id)
    return code


def tokenize(line):
    tokens = []
    if len(line) == 0:
        return tokens
    token = ''
    state = 0
    i = 0
    while i < len(line):
        c = line[i]
        state = lexical_state_tabel[state][T(c)]
        if state == 0:
            if len(token) > 0:
                tokens.append(token)
            token = ''
        else:
            token = ''.join([token, c]).strip()
            i += 1
    if len(token) > 0:
        tokens.append(token)
    return tokens


def extract_tokens(statement):
    tokens = tokenize(statement)
    return tokens




