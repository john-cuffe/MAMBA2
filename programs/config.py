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
            try:
                v_json = json.loads(v)
                dict[k] = v_json
            except Exception as error:
                dict[k] = str(v)