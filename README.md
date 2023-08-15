# Neptune - PyTorch integration

Experiment tracking for PyTorch-trained models.

## What will you get with this integration?

* Log, organize, visualize, and compare ML experiments in a single place
* Monitor model training live
* Version and query production-ready models and associated metadata (e.g., datasets)
* Collaborate with the team and across the organization

## What will be logged to Neptune?

* Training metrics
* Model checkpoints
* Model predictions
* [Other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![image](https://docs.neptune.ai/img/app/integrations/pytorch.png)

## Resources

* [Documentation](https://docs.neptune.ai/integrations/pytorch/)
* [Code example on GitHub](https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/pytorch/scripts)
* [Example project in the Neptune app](https://app.neptune.ai/o/common/org/pytorch-integration/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=9920962e-ff6a-4dea-b551-88006799b116&shortId=PYTOR1-7411&type=run)

## Example



```python
from neptune_pytorch import NeptuneLogger

run = neptune.init_run()
neptune_callback = NeptuneLogger(run=run, model=model)
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting_help).
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! In the Neptune app, click the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP).
* You can just shoot us an email at [support@neptune.ai](mailto:support@neptune.ai).
