import os, time

tree = {'path': '.', 'children': [], 'leaves': [], 'num_leaves': 0, 'parent': None}

def parse_dir(t):
    for x in os.scandir(t['path']):
        if x.is_dir():
            child = {'path': x.path, 'children': [], 'leaves': [], 'num_leaves': 0, 'parent': t}
            t['children'].append(child)
            parse_dir(child)
        elif x.is_file():
            f = x.name
            if f.endswith(".html"):
                o = (f, time.ctime(x.stat().st_mtime))
                t['leaves'].append(o)

def sum_leaves(t):
    nl = 0
    for c in t['children']:
        nl += sum_leaves(c)
    nl += len(t['leaves'])
    t['num_leaves'] = nl
    return nl

def prune(t):
    t['children'] = [c for c in t['children'] if c['num_leaves'] > 0]
    for c in t['children']:
        prune(c)

def emit_list(t, depth=0):
    indent = " " * 2 * depth
    new_list = (len(t['leaves']) > 0) or (len(t['children']) > 1)
    if new_list:
        print(indent + f"<li><b>{t['path']}/</b>")
        print(indent + "<ol>")
    for lv, lt in t['leaves']:
        print(indent + " " + f"<li><a href=\"{t['path']}/{lv}\">{lv}</a> ({lt})</li>")
    for c in t['children']:
        emit_list(c, depth=depth + 1)
    if new_list:
        print("</li>")
        print(indent + "</ol>")

parse_dir(tree)
sum_leaves(tree)
prune(tree)

print("<html><head></head><body><ol>")

emit_list(tree)

print("</ol></body></html>")
