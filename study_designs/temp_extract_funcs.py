import os
import ast
import json

target_dir = r"d:\repos\reserarch-helper-prod\research-helper-development\study_design"
results = {}

if not os.path.exists(target_dir):
    print(f"Directory not found: {target_dir}")
    exit(1)

for filename in os.listdir(target_dir):
    if filename.startswith("utils") and filename.endswith(".py"):
        filepath = os.path.join(target_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
                functions = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]
                results[filename] = functions
        except Exception as e:
            results[filename] = [f"Error: {str(e)}"]

print(json.dumps(results, indent=2))
