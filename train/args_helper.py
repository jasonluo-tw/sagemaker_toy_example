import os
import argparse

def argv_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-size", type=int, default=8, help="input size"
    )
    parser.add_argument(
        "--output-size", type=int, default=1, help="output size"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="EPOCHS to train"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=100, help="hidden neurons"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="BATCH SIZE"
    )
    parser.add_argument(
        "--checkpoint-folder", type=str, default="/opt/ml/checkpoints", help="place model here"
    )

    if "SM_CHANNEL_TRAIN" in os.environ:
        parser.add_argument("--train-input", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    else:
        parser.add_argument("--train-input", type=str, default="../data")

    if "SM_CHANNEL_LABEL" in os.environ:
        parser.add_argument("--train-label", type=str, default=os.environ["SM_CHANNEL_LABEL"])
    else:
        parser.add_argument("--train-label", type=str)

    if "SM_MODEL_DIR" in os.environ:
        #parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--model_dir", type=str, default='/opt/ml/model')
    else:
        os.makedirs("./model_weights", exist_ok=True)
        parser.add_argument("--model_dir", default='./model_weights', type=str)

    if "SM_OUTPUT_DATA_DIR" in os.environ:
        parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    else:
        os.makedirs("./output_data", exist_ok=True)
        parser.add_argument("--output-data-dir", type=str, default="./output_data")

    return parser 
