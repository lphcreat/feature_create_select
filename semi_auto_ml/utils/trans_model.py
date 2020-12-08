from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes

class SModelTrans(Estimator):
    """
    for user define yourself model/transformer and its must create by sklearn
    """
    model_family = ModelFamily.NONE
    def __init__(self,problem_type,skmodel,model_name,model_type='model',model_params={},h_ranges={},random_state=0, **kwargs):
        assert model_type in ['model','transformer']
        parameters = model_params
        super().hyperparameter_ranges = h_ranges
        super().name = f"self define {model_name} skmodel"
        self.model_type = model_type
        if self.model_type == 'model':
            super().supported_problem_types = [ProblemTypes[problem_type]]
        parameters.update(kwargs)
        sk_model = skmodel(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=sk_model,    
                         random_state=random_state)

    @property
    def feature_importance(self):
        try:
            return self._component_obj.coef_
        except :
            return self._component_obj.feature_importances_
        finally:
            return None