from pytorch_lightning.callbacks import Callback
import torchvision
import torch

class ImageLogger(Callback):
    def __init__(self, num_images=50, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            x, _ = batch            
            
            x = x.reshape(-1, 3, 256, 256)

            max_num_image = min(x.shape[0], self.num_images)
            grid_x = torchvision.utils.make_grid(x)
            trainer.logger.experiment.add_image('img', grid_x[0:max_num_image], 0)