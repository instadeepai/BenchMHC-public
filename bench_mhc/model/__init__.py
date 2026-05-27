"""Package to define the different models of the framework."""

from types import MappingProxyType

from bench_mhc.model.mhc1_nn_align import MHC1NNAlignLightningModule

ModelType = MHC1NNAlignLightningModule

MODEL_NAME2LIGHTNING_MODULE: MappingProxyType[str, type[ModelType]] = MappingProxyType(
    {
        "MHC1NNAlignModel": MHC1NNAlignLightningModule,
    }
)
