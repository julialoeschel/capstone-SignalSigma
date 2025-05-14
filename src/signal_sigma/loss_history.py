# loss_history.py
# Einfache Verlusthistorie ohne PyTorch Lightning

class LossHistory:
    """
    Platzhalter-Klasse für Verlusthistorie.
    Wird im TFT-Workflow nicht benötigt.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def append(self, train_loss=None, val_loss=None):
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def get(self):
        return {
            'train': self.train_losses,
            'val': self.val_losses
        }
