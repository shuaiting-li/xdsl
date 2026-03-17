from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg, memref, scf, vector
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    VectorType,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

_index_type = IndexType()


@dataclass
class VectorizeReductionOp(RewritePattern):
    """Vectorize a 1-D ``linalg.reduce`` over a memref using SIMD vector ops.

    Converts:

    .. code-block:: mlir

        linalg.reduce ins(%input : memref<N x T>) outs(%init : memref<T>)
                      dimensions = [0]
        (%in : T, %acc : T) {
            %s = arith.add %in, %acc : T
            linalg.yield %s : T
        }

    To a vectorized loop:

    .. code-block:: mlir

        %c0   = arith.constant 0 : index
        %cV   = arith.constant V : index
        %cN   = arith.constant N : index
        %zero = arith.constant dense<0> : vector<V x T>
        %vec  = scf.for %i = %c0 to %cN step %cV
                        iter_args(%vacc = %zero) -> vector<V x T> {
            %chunk   = vector.load %input[%i] : memref<N x T>, vector<V x T>
            %new_acc = arith.add %vacc, %chunk : vector<V x T>
            scf.yield %new_acc : vector<V x T>
        }
        %partial = vector.reduction <add>, %vec : vector<V x T> into T
        %old     = memref.load %init[] : memref<T>
        %result  = arith.add %old, %partial : T
        memref.store %result, %init[] : memref<T>

    Applicable when:

    * Input is ``memref<N x T>`` with a known, positive static ``N``.
    * Output is ``memref<T>`` (0-D scalar memref).
    * Reduction dimension is ``[0]``.
    * Body is a single ``arith.addf`` (float) or ``arith.addi``
      (integer/index) followed by ``linalg.yield``.
    * ``N`` is evenly divisible by ``vector_size``.
    """

    vector_size: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.ReduceOp, rewriter: PatternRewriter, /):
        input_type = op.input.type
        init_type = op.init.type

        # Only handle memref operands (not tensors).
        if not isa(input_type, MemRefType) or not isa(init_type, MemRefType):
            return

        input_shape = input_type.get_shape()
        init_shape = init_type.get_shape()

        # Input must be 1-D; output must be scalar (0-D).
        if len(input_shape) != 1 or len(init_shape) != 0:
            return

        # Reduction must be over dimension 0 only.
        if list(op.dimensions.get_values()) != [0]:
            return

        N = input_shape[0]
        V = self.vector_size

        # N must be a static positive integer evenly divisible by V.
        if N <= 0 or V < 1 or N % V != 0:
            return

        elem_type = input_type.element_type
        is_int = isa(elem_type, builtin.IntegerType | builtin.IndexType)
        is_float = isa(elem_type, builtin.AnyFloat)

        if not is_int and not is_float:
            return

        # Body must be exactly one arith add op followed by linalg.yield.
        body_ops = list(op.region.block.ops)
        if len(body_ops) != 2:
            return

        combining_op, yield_op = body_ops
        if not isinstance(yield_op, linalg.YieldOp):
            return

        if is_int and not isinstance(combining_op, arith.AddiOp):
            return
        if is_float and not isinstance(combining_op, arith.AddfOp):
            return

        # ---- Build the scf.for body block --------------------------------
        vec_type = VectorType(elem_type, [V])
        body_block = Block(arg_types=[_index_type, vec_type])
        i_var, vec_acc = body_block.args

        chunk_op = vector.LoadOp(op.input, [i_var], vec_type)
        if is_int:
            acc_op: arith.AddiOp | arith.AddfOp = arith.AddiOp(vec_acc, chunk_op)
        else:
            acc_op = arith.AddfOp(vec_acc, chunk_op)
        body_block.add_ops([chunk_op, acc_op, scf.YieldOp(acc_op)])

        # ---- Emit all surrounding ops before the matched op --------------
        with ImplicitBuilder(rewriter):
            c0 = arith.ConstantOp(IntegerAttr(0, _index_type)).result
            c_V = arith.ConstantOp(IntegerAttr(V, _index_type)).result
            c_N = arith.ConstantOp(IntegerAttr(N, _index_type)).result

            # Zero-vector as the initial vector accumulator (identity for add).
            if is_int:
                zero_attr = DenseIntOrFPElementsAttr.from_list(vec_type, [0])
            else:
                zero_attr = DenseIntOrFPElementsAttr.from_list(vec_type, [0.0])
            zero_vec = arith.ConstantOp(zero_attr).result

            # Main vectorized loop.
            for_op = scf.ForOp(c0, c_N, c_V, [zero_vec], Region(body_block))
            vec_result = for_op.res[0]

            # Horizontal reduction: collapse the vector to a scalar.
            kind = vector.CombiningKindAttr(vector.CombiningKindFlag.ADD)
            partial = vector.ReductionOp(vec_result, kind).dest

            # Incorporate the pre-existing init value.
            init_val = memref.LoadOp.get(op.init, []).res
            if is_int:
                result: SSAValue = arith.AddiOp(init_val, partial).result
            else:
                result = arith.AddfOp(init_val, partial).result

            memref.StoreOp.get(result, op.init, [])

        rewriter.erase_op(op)


@dataclass(frozen=True)
class VectorizeLinalgReductionPass(ModulePass):
    """Vectorize 1-D ``linalg.reduce`` operations using SIMD vector operations.

    Rewrites ``linalg.reduce`` operations that sum a 1-D memref into a scalar
    accumulator into a form that uses ``vector.load`` to load ``vector_size``
    elements at a time, accumulates them in a vector register, and collapses the
    result with ``vector.reduction``.  This enables the backend to emit SIMD
    instructions and exploit multiple ALUs in parallel.

    The ``vector_size`` parameter controls the SIMD width (default: 4).
    """

    name = "vectorize-reduction"

    vector_size: int = 4

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            VectorizeReductionOp(self.vector_size),
            apply_recursively=False,
        ).rewrite_module(op)
