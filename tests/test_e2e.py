from __future__ import print_function

try:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run
except ImportError:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run

import torch
import torch.nn.functional as F
import torch.optim as optim

from neptune_pytorch import NeptuneLogger


def test_e2e(model, dataset):
    # Training settings
    device = torch.device("cpu")
    train_kwargs = {"batch_size": 8}

    dataset = torch.utils.data.TensorDataset(*dataset)
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

    run = init_run()

    npt_logger = NeptuneLogger(
        run, model=model, log_model_diagram=True, log_gradients=True, log_parameters=True, log_freq=3
    )

    for epoch in range(1, 4):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            run[npt_logger.base_namespace]["batch/loss"].append(loss.item())

        npt_logger.save_checkpoint()

    # Save final model
    npt_logger.save_model("model")

    run.wait()
    run.exists(f"{npt_logger.base_namespace}/batch/loss")
    run.exists(f"{npt_logger.base_namespace}/model/checkpoint_1.pt")
    run.exists(f"{npt_logger.base_namespace}/model/checkpoint_2.pt")
    run.exists(f"{npt_logger.base_namespace}/model/model.pt")
    run.exists(f"{npt_logger.base_namespace}/model/summary")
    run.exists(f"{npt_logger.base_namespace}/model/visualization")
