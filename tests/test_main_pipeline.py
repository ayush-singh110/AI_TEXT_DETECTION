import unittest
from unittest.mock import patch, MagicMock

import main


class MainPipelineTests(unittest.TestCase):
    @patch("main.ModelTrainer")
    @patch("main.DataTransformation")
    @patch("main.DataIngestion")
    def test_run_training_pipeline_uses_training_split_for_transformation(self, mock_ingestion_cls, mock_transformation_cls, mock_trainer_cls):
        ingestion_instance = MagicMock()
        ingestion_instance.initiate_data_ingestion.return_value = (
            "artifacts/raw_data.csv",
            "artifacts/raw_train_data.csv",
            "artifacts/raw_test_data.csv",
        )
        mock_ingestion_cls.return_value = ingestion_instance

        transformation_instance = MagicMock()
        transformation_instance.transformation.return_value = "artifacts/transformed.csv"
        mock_transformation_cls.return_value = transformation_instance

        trainer_instance = MagicMock()
        trainer_instance.initiate_model_trainer.return_value = (0.95, 0.90, 0.91, 0.89, "LogisticRegression")
        mock_trainer_cls.return_value = trainer_instance

        main.run_training_pipeline()

        transformation_instance.transformation.assert_called_once_with("artifacts/raw_train_data.csv")


if __name__ == "__main__":
    unittest.main()
