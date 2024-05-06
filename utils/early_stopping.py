

class SFTEarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class UFTEarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_denoiser_loss = None
        self.best_discriminator_loss = None
        self.early_stop = False

    def __call__(self, denoiser_loss, discriminator_loss):
        if self.best_denoiser_loss is None and self.best_discriminator_loss is None:
            self.best_denoiser_loss = denoiser_loss
            self.best_discriminator_loss = discriminator_loss
        elif denoiser_loss > self.best_denoiser_loss and discriminator_loss > self.best_discriminator_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.best_denoiser_loss = denoiser_loss
            self.best_discriminator_loss = discriminator_loss
            self.counter = 0
        else:
            self.best_denoiser_loss = denoiser_loss
            self.best_discriminator_loss = discriminator_loss
            self.counter = 0
