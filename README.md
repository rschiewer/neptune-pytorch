# Neptune + PyTorch integration

Experiment tracking, model registry, data versioning, and live model monitoring for Keras trained models.

## What will you get with this integration?

* Log, display, organize, and compare ML experiments in a single place
* Version, store, manage, and query trained models, and model building metadata
* Record and monitor model training, evaluation, or production runs live
* Collaborate with a team

## What will be logged to Neptune?

* hyperparameters for every run,
* learning curves for losses and metrics during training,
* hardware consumption and stdout/stderr output during training,
* PyTorch tensors as images to see model predictions live,
* training code and Git commit information,
* model weights,
* [other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![image](https://user-images.githubusercontent.com/97611089/160638338-8a276866-6ce8-4d0a-93f5-bd564d00afdf.png)
*Example charts in the Neptune UI with logged accuracy and loss*

## Resources

* [Documentation](https://docs.neptune.ai/integrations/keras)

## Example

On the command line:

```
pip install neptune-pytorch neptune
```

In Python:

```python
import neptune
from neptune.integrations.pytorch import NeptuneLogger
from neptune import ANONYMOUS_API_TOKEN

# Start a run
run = neptune.init_run(
    project="common/pytorch-integration",
    api_token=ANONYMOUS_API_TOKEN,
)

# Create a NeptuneCallback instance
neptune_logger = NeptuneLogger(run=run, base_namespace="training")

# Stop the run
run.stop()
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting_help)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
