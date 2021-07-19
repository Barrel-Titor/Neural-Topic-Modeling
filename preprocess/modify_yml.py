import sys

filename = sys.argv[1]

new_lines = []
flag = False
with open(filename) as fi:
    for line in fi.readlines():
        if line.startswith('dependencies:'):
            flag = True
            new_lines.append(line)
            continue
        if line.startswith('  - pip:'):
            flag = False

        if flag:
            splits = line.split('=')
            new_line = '='.join(splits[:-1]) + '\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

with open('teacher/teacher_modified.yml', 'w') as fo:
    # fo.write('\n'.join(new_lines))
    fo.writelines(new_lines)
