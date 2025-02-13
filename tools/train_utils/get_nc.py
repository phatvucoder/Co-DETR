import re
import ast
import sys

def extract_classes(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        match = re.search(r'classes\s*=\s*(\[[^\]]+\]|\([^\)]+\))', content) # Parse classes from config file
        if match:
            classes = ast.literal_eval(match.group(1))
            if isinstance(classes, (list, tuple)):
                return len(classes)
    except Exception:
        pass
    return 0

if len(sys.argv) != 2:
    print("Usage: python get_nc.py <config_file>")
    sys.exit(1)

# Get number of classes from config file
config_path = sys.argv[1]
num_classes = extract_classes(config_path)
print(num_classes)