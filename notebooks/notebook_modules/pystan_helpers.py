import pystan
import pickle
from hashlib import md5
from timeit import default_timer as timer
from pathlib import Path

def StanModel_cache(file, model_name='anon_model', **kwargs):
    """Use just as you would `StanModel`"""
    with open(file) as f:
        txt = f.read()
        code_hash = md5(txt.encode('ascii')).hexdigest()

    cache_fname = Path('cached-stan-models/cached-{}-{}.pkl'.format(model_name, code_hash))
    
    try:
        sm = pickle.load(open(cache_fname, 'rb'))
    except:
        print(f'No cached model - compiling \'{file}\'.')
        
        start = timer()
        sm = pystan.StanModel(file=file, model_name=model_name, **kwargs)
        end = timer()
        print(f'{(end - start) / 60:.2f} minutes to compile model')
        
        with open(cache_fname, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print('Using cached StanModel.')
    
    return sm