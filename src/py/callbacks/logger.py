from pytorch_lightning.callbacks import Callback
import torchvision
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
            grid_x = grid_x.permute(1, 2, 0)
            # print(grid_x.shape)
            # trainer.logger.experiment.log_image(grid_x[0:max_num_image],'img', 0)

class ImageSegLogger(Callback):
    def __init__(self,max_num_image=8, log_steps=100):
        self.log_steps = log_steps
        self.max_num_image = max_num_image
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            imgs, targets = batch

            imgs = imgs[:self.max_num_image]
            targets = targets[:self.max_num_image]

            n_cols= int(self.max_num_image /2)
            fig, axs = plt.subplots(2, n_cols)
            axs = axs.flatten()
            images = [img for img in imgs]
            boxes_list = [target['boxes'] for target in targets]
            masks_list = [target['masks'] for target in targets]


            for ax, masks in zip(axs, masks_list):
                mask_mul = torch.zeros_like(masks[0])
                for mask in masks:
                    mask_mul[ mask !=0 ] = 1
                ax.imshow(mask_mul.cpu().detach().numpy())
                ax.axis('off')  # Optional: Turn off axes for better visualization

            plt.tight_layout()
            trainer.logger.experiment["fig/train/masks"].upload(fig)
            plt.close()

            for ax, img, boxes in zip(axs, images, boxes_list):
                ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image

                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().detach().numpy()
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                    ax.add_patch(rect)  # Add the box
                ax.axis('off')  # Optional: Turn off axes for better visualization

            plt.tight_layout()
            trainer.logger.experiment["fig/train/input_boxes"].upload(fig)
            plt.close()


            with torch.no_grad():
                x_hat = pl_module(imgs, targets=None, mode='test')
                n_cols= int(self.max_num_image /2)

                fig, axs = plt.subplots(2, n_cols)
                axs = axs.flatten()
                boxes_list = [target['boxes'] for target in x_hat]

                for ax, img, boxes in zip(axs, images, boxes_list):
                    ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image
                    for box in boxes:
                        x1, y1, x2, y2 = box.cpu().detach().numpy()
                        width, height = x2 - x1, y2 - y1
                        rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                        ax.add_patch(rect)  # Add the box
                    ax.axis('off')  # Optional: Turn off axes for better visualization

                plt.tight_layout()
                trainer.logger.experiment["fig/train/predictions_boxes"].upload(fig)
                plt.close()



    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            imgs, targets = batch

            imgs = imgs[:self.max_num_image]
            targets = targets[:self.max_num_image]

            n_cols= int(self.max_num_image /2)
            fig, axs = plt.subplots(2, n_cols)
            axs = axs.flatten()
            images = [img for img in imgs]
            boxes_list = [target['boxes'] for target in targets]
            masks_list = [target['masks'] for target in targets]


            for ax, masks in zip(axs, masks_list):
                mask_mul = torch.zeros_like(masks[0])
                for mask in masks:
                    mask_mul[ mask !=0 ] = 1
                ax.imshow(mask_mul.cpu().detach().numpy())
                ax.axis('off')  # Optional: Turn off axes for better visualization

            plt.tight_layout()
            trainer.logger.experiment["fig/val/masks"].upload(fig)
            plt.close()

            for ax, img, boxes in zip(axs, images, boxes_list):
                ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image

                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().detach().numpy()
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                    ax.add_patch(rect)  # Add the box
                ax.axis('off')  # Optional: Turn off axes for better visualization

            plt.tight_layout()
            trainer.logger.experiment["fig/val/input_boxes"].upload(fig)
            plt.close()


            with torch.no_grad():
                x_hat = pl_module(imgs, targets=None, mode='test')
                n_cols= int(self.max_num_image /2)

                fig, axs = plt.subplots(2, n_cols)
                axs = axs.flatten()
                boxes_list = [target['boxes'] for target in x_hat]

                for ax, img, boxes in zip(axs, images, boxes_list):
                    ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image
                    for box in boxes:
                        x1, y1, x2, y2 = box.cpu().detach().numpy()
                        width, height = x2 - x1, y2 - y1
                        rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                        ax.add_patch(rect)  # Add the box
                    ax.axis('off')  # Optional: Turn off axes for better visualization

                plt.tight_layout()
                trainer.logger.experiment["fig/val/predictions_boxes"].upload(fig)
                plt.close()