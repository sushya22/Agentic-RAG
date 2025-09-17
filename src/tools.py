import re
from typing import List

class Tools:
    def calculator(self, query: str):
        # extract a simple math expression and evaluate safely
        # support only digits and +-*/(). spaces
        expr = ''.join(re.findall(r'[0-9+\-*/(). ]+', query))
        expr = expr.strip()
        if not expr:
            return 'No arithmetic expression found.'
        # safety: allow only digits and operators
        if re.search(r'[^0-9+\-*/(). ]', expr):
            return 'Expression contains invalid characters.'
        try:
            # safe eval using eval with restricted globals
            result = eval(expr, {'__builtins__': None}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def file_search(self, docs: List[str], query: str):
        # naive search: return docs that contain any keyword
        keywords = [w.lower() for w in re.findall(r"\\w+", query) if len(w)>3]
        found = []
        for d in docs:
            txt = d.lower()
            if any(k in txt for k in keywords):
                found.append(d[:300].replace('\n',' ') + '...')
        return found
