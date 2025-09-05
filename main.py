from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__=="__main__":
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()
    print(raw_data_path)
    obj1=DataTransformation()
    transformed_data_path=obj1.transformation(raw_data_path)
    print(transformed_data_path)