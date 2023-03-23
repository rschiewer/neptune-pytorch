#
# Copyright (c) 2023, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
__all__ = ["__version__", "NeptuneLogger"]

from typing import (
    Optional,
    Union,
)

import torch

from neptune_pytorch.impl.version import __version__

try:
    # neptune-client>=1.0.0 package structure
    from neptune import Run
    from neptune.handlers import Handler
    from neptune.integrations.utils import (
        expect_not_an_experiment,
        verify_type,
    )
except ImportError:
    # neptune-client=0.9.0+ package structure
    from neptune.new import Run
    from neptune.new.handlers import Handler
    from neptune.new.integrations.utils import (
        expect_not_an_experiment,
        verify_type,
    )

IS_TORCHVIZ_AVAILABLE = True
try:
    import torchviz
except ImportError:
    IS_TORCHVIZ_AVAILABLE = False

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-pytorch"


class NeptuneLogger:
    def __init__(
        self,
        run: Union[Run, Handler],
        *,
        model,
        base_namespace="training",
        log_model_diagram: bool = False,
        log_gradients: bool = False,
        log_parameters: bool = False,
        log_freq: int = 100,
    ):
        expect_not_an_experiment(run)
        verify_type("run", run, (Run, Handler))

        self.run = run
        self.model = model
        self._base_namespace = base_namespace
        self.log_model_diagram = log_model_diagram
        self.ckpt_number = 1
        self.log_freq = log_freq

        self._is_viz_saved = False
        self._vis_hook_handler = None
        if log_model_diagram:
            self.run[self._base_namespace]["model"]["summary"] = str(model)
            self.add_visualization_hook()

        self.log_gradients = log_gradients
        self._gradients_iter_tracker = {}
        self._gradients_hook_handler = {}
        if self.log_gradients:
            self.add_hooks_for_grads()

        self.log_parameters = log_parameters
        self._params_iter_tracker = 0
        self._params_hook_handler = None
        if self.log_parameters:
            self.add_hooks_for_params()

    def add_hooks_for_grads(self):
        for name, parameter in self.model.named_parameters():
            self._gradients_iter_tracker[name] = 0

            def hook(grad, name=name):
                self._gradients_iter_tracker[name] += 1
                if self._gradients_iter_tracker[name] % self.log_freq == 0:
                    self.run[self.base_namespace]["plots"]["gradients"][name].append(grad.norm())

            self._gradients_hook_handler[name] = parameter.register_hook(hook)

    def add_visualization_hook(self):
        if not IS_TORCHVIZ_AVAILABLE:
            # Correctly print warning.
            print("WARNING")
            return

        def hook(module, input, output):
            if not self._is_viz_saved:
                dot = torchviz.make_dot(output, params=dict(module.named_parameters()))
                # Use tempfile correctly.
                dot.format = "png"
                dot.render(outfile="torch-viz.png")
                self.run[self.base_namespace]["model"]["visualization"].upload("torch-viz.png")
                self._is_viz_saved = True

        self._vis_hook_handler = self.model.register_forward_hook(hook)

    def add_hooks_for_params(self):
        def hook(module, inp, output):
            self._params_iter_tracker += 1
            if self._params_iter_tracker % self.log_freq == 0:
                for name, param in module.named_parameters():
                    self.run[self.base_namespace]["plots"]["parameters"][name].append(param.norm())

        self._params_hook_handler = self.model.register_forward_hook(hook)

    @property
    def base_namespace(self):
        return self._base_namespace

    def save_model(self, model_name: Optional[str] = None):
        if model_name is None:
            # Default model name
            model_name = "model.pt"
        else:
            # User is not expected to add extension
            model_name = model_name + ".pt"
        torch.save(self.model.state_dict(), model_name)
        self.run[self._base_namespace]["model"][model_name].upload(model_name)

    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        if checkpoint_name is None:
            # Default checkpoint name
            checkpoint_name = f"checkpoint_{self.ckpt_number}.pt"
            self.ckpt_number += 1
        else:
            # User is not expected to add extension
            checkpoint_name = checkpoint_name + ".pt"
        torch.save(self.model.state_dict(), checkpoint_name)
        self.run[self._base_namespace]["model"][checkpoint_name].upload(checkpoint_name)

    def __del__(self):
        # Remove hooks
        if self._params_hook_handler is not None:
            self._params_hook_handler.remove()

        for name, handler in self._gradients_hook_handler.items():
            if handler is not None:
                handler.remove()

        if self._vis_hook_handler is not None:
            self._vis_hook_handler.remove()
