TEST_SIZE    = 0.3
RANDOM_SEED  = 42

D_MODEL      = 512
NHEAD        = 16
NUM_LAYERS   = 8
DROPOUT      = 0.1

BATCH_SIZE   = 128
LR           = 1e-4
PATIENCE     = 1000

PRETRAIN_EPOCHS  = 500
FINETUNE_EPOCHS  = 500

BLACKLIST_FEATURES: list[str] = []

USE_COMBAT = False   # ComBat harmonization to remove site/scanner effects
# ts bad