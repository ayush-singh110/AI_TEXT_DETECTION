from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline

if __name__=="__main__":
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()
    print(raw_data_path)
    obj1=DataTransformation()
    transformed_data_path=obj1.transformation(raw_data_path)
    print(transformed_data_path)
    obj2=ModelTrainer()
    score,best_model_name=obj2.initiate_model_trainer(transformed_data_path)
    print(score)
    print(best_model_name)
    obj3=PredictPipeline()
    result=obj3.predict("In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.")
    print(result)