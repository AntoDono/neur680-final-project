TEST_SIZE    = 0.4
RANDOM_SEED  = 42

D_MODEL      = 128
NHEAD        = 8
NUM_LAYERS   = 4
DROPOUT      = 0.2

BATCH_SIZE   = 128
LR           = 1e-4
PATIENCE     = 1000

PRETRAIN_EPOCHS  = 1000
FINETUNE_EPOCHS  = 500

BLACKLIST_FEATURES: list[str] = []

USE_COMBAT = True   # ComBat harmonization to remove site/scanner effects
