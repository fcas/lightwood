from collections import deque

import numpy as np

from lightwood.api.dtype import dtype


'''
def is_allowed(v):
    if v is None:
        return True

    if isinstance(v, bool):
        return True

    try:
        float(v)
        return True
    except:
        pass

    if v in ['True', 'False']:
        return True

    if isinstance(v, str):
        if v.startswith('"') and v.endswith('"'):
            return True
        if v.startswith("'") and v.endswith("'"):
            return True

    # Predictor member
    if v.startswith('self.') and '(' not in v and len(v) < 50:
        return True

    # Allowed variable names
    if v in ['df', 'data', 'encoded_data', 'train_data', 'encoded_train_data', 'test_data']:
        return True

    try:
        cv = dict(v)
        for k in cv:
            ka = is_allowed(k)
            ma = is_allowed(cv[k])
            if not ka or not ma:
                return False
        return True
    except Exception:
        pass

    try:
        cv = list(v)
        for ma in cv:
            ma = is_allowed(m)
            if not ma:
                return False
        return True
    except Exception:
        pass

    raise Exception(f'Possible code injection: {v}')
'''


def is_allowed(v):
    if '(' in str(v):
        return False
    if 'lambda' in str(v):
        return False
    if '__' in str(v):
        return False

    return True


def call(entity: dict) -> str:
    # Special behavior for ensemble
    if 'submodels' in entity['args']:
        del entity['args']['submodels']

    for k, v in entity['args'].items():
        if not str(v).startswith('$'):
            if not is_allowed(v):
                raise Exception(f'Invalid value: {v} for arg {k}')

    args = [f'{k}={v}' for k, v in entity['args'].items() if not str(v).startswith('$')]

    for k, v in entity['args'].items():
        if str(v).startswith('$'):
            v = str(v).replace('$', 'self.')
            args.append(f'{k}={v}')

    args = ','.join(args)
    return f"""{entity['module']}({args})"""


def inline_dict(obj: dict) -> str:
    arr = []
    for k, v in obj.items():
        if str(v) in list(dtype.__dict__.keys()):
            v = f"'{v}'"
        k = k.replace("'", "\\'").replace('"', '\\"')
        arr.append(f"""'{k}': {v}""")

    dict_code = '{\n' + ',\n'.join(arr) + '\n}'
    return dict_code


def align(code: str, indent: int) -> str:
    add_space = ''
    for _ in range(indent):
        add_space += '    '

    code_arr = code.split('\n')
    code = f'\n{add_space}'.join(code_arr)
    return code


def _consolidate_analysis_blocks(jsonai, key):
    """
    Receives a list of analysis blocks (where applicable, already filed with `hidden` args) and modifies it so that:
        1. All dependencies are correct.
        2. Execution order is such that all dependencies are met.
            - For this we use a topological sort over the DAG.
    """
    # 1. all dependencies are correct
    defaults = {
        'ICP': {"deps": []},
        'AccStats': {"deps": ['ICP']},
        'ConfStats': {"deps": ['ICP']},
        'GlobalFeatureImportance': {"deps": ['AccStats']}
    }
    blocks = getattr(jsonai, key)
    for i, block in enumerate(blocks):
        if 'args' not in block:
            blocks[i]['args'] = defaults[block['module']]
        elif 'deps' not in block['args']:
            blocks[i]['args']['deps'] = []

    # 2. correct execution order -- build a DAG out of analysis blocks
    block_objs = {b['module']: b for b in blocks}
    block_ids = {k: i for i, k in enumerate(block_objs.keys())}
    idx2block = {i: k for i, k in enumerate(block_objs.keys())}

    adj_M = np.zeros((len(block_ids), len(block_ids)))
    for k, b in block_objs.items():
        for dep in b['args']['deps']:
            adj_M[block_ids[dep]][block_ids[k]] = 1

    sorted_dag = []
    frontier = deque(np.where(adj_M.sum(axis=0) == 0)[0].tolist())  # get initial nodes without dependencies

    while frontier:
        elt = frontier.pop()
        sorted_dag.append(elt)
        dependants = np.where(adj_M[elt, :])[0]
        for dep in dependants:
            adj_M[elt, dep] = 0
            if not adj_M.sum(axis=0)[dep]:
                frontier.append(dep)

    if adj_M.sum() != 0:
        raise Exception("Cycle detected in analysis blocks dependencies, please review and try again!")

    sorted_blocks = []
    for idx in sorted_dag:
        sorted_blocks.append(block_objs[idx2block[idx]])

    return sorted_blocks
