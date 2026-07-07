from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline


def run_training_pipeline():
    ingestion = DataIngestion()
    raw_data_path, raw_train_data_path, raw_test_data_path = ingestion.initiate_data_ingestion()
    print(raw_data_path, raw_train_data_path, raw_test_data_path)

    transformation = DataTransformation()
    transformed_data_path = transformation.transformation(raw_train_data_path)
    print(transformed_data_path)

    trainer = ModelTrainer()
    train_score, test_score, precision, recall, best_model_name = trainer.initiate_model_trainer(transformed_data_path)
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Best Model: {best_model_name}")

    predictor = PredictPipeline()
    result = predictor.predict(
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    )
    print(result)
    return train_score, test_score, precision, recall, best_model_name


if __name__ == "__main__":
    run_training_pipeline()