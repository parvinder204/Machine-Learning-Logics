import logging

logging.basicConfig(
    filename="monitoring/predictions.log",
    level=logging.INFO
)

def log_prediction(text, prediction):

    logging.info(f"text={text} prediction={prediction}")