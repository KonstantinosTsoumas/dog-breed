from src.config.configuration import ConfigurationManager
from src.components.evaluation import Evaluation
from src import logger

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_evaluation_config()
        evaluation = Evaluation(val_config)
        scores = evaluation.evaluation()
        evaluation.save_score(scores)


if __name__ == '__main__':
    try:
        logger.info(f"stage {STAGE_NAME} has started.")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"stage {STAGE_NAME} has been completed.")
    except Exception as e:
        logger.exception(e)
        raise e
