from project.exception import ProjectException 
from project.logger import logging 



class ProjectEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object=preprocessing_object
        self.trained_model_object=trained_model_object
    
    def predict(self, X):
        transformer_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformer_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"