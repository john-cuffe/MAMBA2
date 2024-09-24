import json

def read_properties( props_file, dict):
    # config = {}
    with open(props_file, 'r') as f:
        for line in f:
            line = line.split('#')[0].rstrip() #removes trailing whitespace and '\n' chars
            if "=" not in line: continue #skips blanks and comments w/o =
            if line.startswith("#"): continue #skips comments which contain =
            k, v = line.split(" = ", 1)
            ###detect a json payload
            if v[0]=='{':
                dict[k]=json.loads(v)
            elif v[0:2] == '[{':
                dict[k]=json.loads(v)
            elif v[0] == '[':
                dict[k] = v.split(',')
            else:
                dict[k] = v