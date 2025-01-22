#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        if(inputs.size() != 2) return std::nullopt;
        auto rankA = inputs[0]->getRank();
        auto rankB = inputs[1]->getRank();
        if(rankA != rankB) return std::nullopt;

        auto dimA = inputs[0]->getDims();
        auto dimB = inputs[1]->getDims();
        if(transA) std::swap(dimA[dimA.size() - 1], dimA[dimA.size() - 2]);
        if(transB) std::swap(dimB[dimB.size() - 1], dimB[dimB.size() - 2]);

        if (dimA[dimA.size() - 1] != dimB[dimB.size() - 2]) {
            return std::nullopt;
        }

        Shape outShape;
        for(size_t i = 0; i < rankA - 2; i ++){
            if(dimA[i] == dimB[i] || dimA[i] == 1 || dimB[i] == 1) 
                outShape.push_back(std::max(dimA[i], dimB[i]));
            else
                return std::nullopt;
        }

        outShape.push_back(dimA[dimA.size() - 2]);
        outShape.push_back(dimB[dimB.size() - 1]);
        return {{outShape}};
    }

} // namespace infini