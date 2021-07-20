import torch
from model import model


model_config = {
    "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
    "convolution_layer": 1, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
    "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
    "use_cuda": False, "batch_size": 30000, "learning_rate": 0.0001, "epoch": 2000,
    "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/newb",
    "checkpoint_path": "test_checkpoints/quick06-4b_Custom_CRNN_AdamW_ReLU_0.0001_sl15_999_epoch_1279.pt"
}
nn_model = model.model_load(model_configure=model_config)
print(nn_model.state_dict())
