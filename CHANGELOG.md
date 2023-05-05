## neptune-pytorch 0.2.0

### Fixes
- Change where `checkpoints` are logged. Previously they we logged under `base_namespace/model` but now they will be logged under `base_namespace/model/checkpoints` (https://github.com/neptune-ai/neptune-pytorch/pull/5)
- Add warning if `dot` is not installed instead of hard error. Also, improve clean-up of visualization files (https://github.com/neptune-ai/neptune-pytorch/pull/6)


## neptune-pytorch 0.1.0 (initial release)

### Features
- Create `NeptuneLogger` for logging metadata (https://github.com/neptune-ai/neptune-pytorch/pull/1)
