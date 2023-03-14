import torch
from typing import Optional
import tempfile
IS_TORCHVIZ_AVAILABLE = True
try:
    import torchviz
except ImportError:
    IS_TORCHVIZ_AVAILABLE = False

class NeptuneLogger:
    def __init__(self, run,
                 *,
                 model,
                 base_namespace='training',
                 log_model_diagram: bool = False,
                 log_gradients: bool = False,
                 log_parameters: bool = False,
                 log_freq: int = 100):
        self.run = run
        self.model = model
        self._base_namespace = base_namespace
        self.log_model_diagram = log_model_diagram
        self.ckpt_number = 1
        self.log_freq = log_freq

        self._is_viz_saved = False
        self._vis_hook_handler = None
        if log_model_diagram:
            self.run[self._base_namespace]['model']['summary'] = str(model)
            self.maybe_add_visualization_hook()

        self.log_gradients = log_gradients
        self._gradients_iter_tracker = {}
        self._gradients_hook_handler = {}
        self.maybe_add_hooks_for_grads()
        
        self.log_parameters = log_parameters
        self._params_iter_tracker = 0
        self._params_hook_handler = None
        self.maybe_add_hooks_for_params()

    def maybe_add_hooks_for_grads(self):
        if self.log_gradients:
            for name, parameter in self.model.named_parameters():
                self._gradients_iter_tracker[name] = 0

                def hook(grad, name=name):
                    self._gradients_iter_tracker[name] += 1
                    if self._gradients_iter_tracker[name] % self.log_freq == 0:
                        self.run[self.base_namespace]['plots']['gradients'][name].append(
                            grad.norm())

                self._gradients_hook_handler[name] = parameter.register_hook(
                    hook)

    def maybe_add_visualization_hook(self):
        if not IS_TORCHVIZ_AVAILABLE:
            # Correctly print warning.
            print("WARNING")
            return

        def hook(module, input, output):
            if not self._is_viz_saved:
                dot = torchviz.make_dot(output, params=dict(module.named_parameters()))
                # Use tempfile correctly.
                with tempfile.NamedTemporaryFile() as tmp:
                    dot.format = 'png'
                    dot.render(outfile='torch-viz.png')
                    self.run[self.base_namespace]['model']['visualization'].upload('torch-viz.png')
                self._is_viz_saved = True

        self._vis_hook_handler = self.model.register_forward_hook(hook)

    def maybe_add_hooks_for_params(self):
        if self.log_parameters:
            def hook(module, inp, output):
                self._params_iter_tracker += 1
                if self._params_iter_tracker % self.log_freq == 0:
                    for name, param in module.named_parameters():
                        self.run[self.base_namespace]['plots']['parameters'][name].append(
                            param.norm())

            self._params_hook_handler = self.model.register_forward_hook(hook)

    @property
    def base_namespace(self):
        return self._base_namespace

    def save_model(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = 'model.pt'
        else:
            model_name = model_name + ".pt"
        torch.save(self.model.state_dict(), model_name)
        self.run[self._base_namespace]['model'][model_name].upload(model_name)

    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        if checkpoint_name is None:
            checkpoint_name = f'checkpoint_{self.ckpt_number}.pt'
            self.ckpt_number += 1
        else:
            checkpoint_name = checkpoint_name + ".pt"
        torch.save(self.model.state_dict(), checkpoint_name)
        self.run[self._base_namespace]['model'][checkpoint_name].upload(
            checkpoint_name)

    def __del__(self):
        # Remove hooks
        if self._params_hook_handler is not None:
            self._params_hook_handler.remove()

        for name, handler in self._gradients_hook_handler.items():
            if handler is not None:
                handler.remove()
        
        if self._vis_hook_handler is not None:
            self._vis_hook_handler.remove()
