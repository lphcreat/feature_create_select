
from semi_auto_ml.utils.extract_funcs import save_sk_model,load_sk_model
from sklearn.pipeline import Pipeline
import os

class ModelDeploy():
    '''
    save tratnsform and predict model,not contain feature_tools
    '''
    @staticmethod
    def save_model(model_path:tuple,save_path):
        '''
        save all model to file,except feature_tools
        '''
        for ind,item in enumerate(model_path):
            file_path = save_path+f'{ind}_m.joblib'
            if hasattr(item,'save'):
                item.save(file_path)
            else:
                save_sk_model(item,file_path)

    @staticmethod
    def load_model(save_path):
        '''
        from save_path load all joblib table,and generate pipeline by name
        '''
        pips = []
        models = sorted([ml for ml in os.listdir(save_path) if ml.endswith('joblib')],key=lambda x:int(x[0]))
        for ind,item in enumerate(models):
            file_path = save_path+item
            clf = load_sk_model(file_path)
            pips.append((f'{ind}_m',clf))
        return Pipeline(pips)


