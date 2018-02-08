def parse_file(file_name):
    with open(file_name) as f:
        return [line.split() ]for line in f]
