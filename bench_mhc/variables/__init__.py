"""Package to define input and output variables."""

from bench_mhc.variables.aa_seq import AASeqVariable
from bench_mhc.variables.binary import BinaryOutput
from bench_mhc.variables.binary import BinaryVariable
from bench_mhc.variables.multiclass import MultiClassVariable
from bench_mhc.variables.nn_align import NNAlignVariable
from bench_mhc.variables.numeric import NumericOutput
from bench_mhc.variables.numeric import NumericVariable

InputClassesUnion = (
    NumericVariable | BinaryVariable | AASeqVariable | NNAlignVariable | MultiClassVariable
)

OutputClassesUnion = NumericOutput | BinaryOutput

VariableClassesUnion = InputClassesUnion | OutputClassesUnion

VARIABLE_NAME2VARIABLE_CLASS = {
    variable_class.__name__: variable_class for variable_class in VariableClassesUnion.__args__
}

OUTPUT_VARIABLE_NAME2VARIABLE_CLASS = {
    variable_class.__name__: variable_class for variable_class in OutputClassesUnion.__args__
}
