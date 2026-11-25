"""A small expression language for read-only dataframe summaries.

This deliberately excludes arbitrary Python snippets, I/O, imports, callbacks,
dynamic attribute access, and pandas query/eval. It is not a Python sandbox.
"""

import ast


_METHODS = {
    "groupby", "get_group", "describe", "mean", "median", "std", "var",
    "sum", "min", "max", "count", "nunique", "value_counts", "quantile",
    "head", "tail", "dropna", "isin", "corr", "cov", "abs", "round",
}
_ATTRIBUTES = _METHODS | {"shape", "size", "columns", "index", "loc", "iloc"}
_NODES = (
    ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Attribute, ast.Call, ast.keyword,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.USub, ast.UAdd, ast.BitAnd, ast.BitOr, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def evaluate_summary(code, df):
    if len(code) > 2000:
        raise ValueError("Query is too long")
    tree = ast.parse(code.strip(), mode="exec")
    if len(tree.body) != 1:
        raise ValueError("Supply one summary expression")
    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expression = statement.value
    elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
          and isinstance(statement.targets[0], ast.Name)
          and statement.targets[0].id in {"result", "output"}):
        expression = statement.value
    else:
        raise ValueError("Only an expression or result = expression is supported")
    tree = ast.Expression(expression)
    nodes = list(ast.walk(tree))
    if len(nodes) > 150:
        raise ValueError("Query is too complex")
    for node in nodes:
        if not isinstance(node, _NODES):
            raise ValueError(f"Unsupported query syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"df", "len"}:
            raise ValueError(f"Unknown query name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr not in _ATTRIBUTES:
            raise ValueError(f"Unsupported attribute: {node.attr}")
        if isinstance(node, ast.keyword) and (node.arg is None or node.arg == "inplace"):
            raise ValueError("Expanded arguments and in-place operations are unsupported")
        if isinstance(node, ast.Call):
            valid = ((isinstance(node.func, ast.Name) and node.func.id == "len") or
                     (isinstance(node.func, ast.Attribute) and node.func.attr in _METHODS))
            if not valid:
                raise ValueError("Only approved summary methods may be called")
    return eval(compile(tree, "<summary>", "eval"), {"__builtins__": {}, "df": df.copy(deep=True), "len": len})
