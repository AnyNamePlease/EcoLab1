#ifndef __I_ECO_ACTIVATION_FUNCTIONS_H__
#define __I_ECO_ACTIVATION_FUNCTIONS_H__

#include <stdint.h>
#include "IEcoBase1.h"

// Структура вектора на выделенной памяти
typedef struct EcoVector {
    double* data;
    size_t capacity;
    size_t size;
} EcoVector;

// Простая плоская матрица (Row-major) для матрицы Якоби N x N
typedef struct EcoMatrix {
    double* data; // Размер N * N
    size_t rows;
    size_t cols;
} EcoMatrix;

// Уникальный IID для интерфейса функций активации
#ifndef __IID_IEcoActivationFunctions
static const UGUID IID_IEcoActivationFunctions = { 0x01, 0x10, {0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18, 0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90} };
#endif

typedef struct IEcoActivationFunctions* IEcoActivationFunctionsPtr_t;

// Виртуальная таблица (VTbl) построена по стандартам Eco с макросом ECOCALLMETHOD
typedef struct IEcoActivationFunctionsVTbl {
    int16_t (ECOCALLMETHOD *QueryInterface)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ const UGUID* riid, /* out */ void** ppv);
    uint32_t (ECOCALLMETHOD *AddRef)(/* in */ IEcoActivationFunctionsPtr_t me);
    uint32_t (ECOCALLMETHOD *Release)(/* in */ IEcoActivationFunctionsPtr_t me);

    double (ECOCALLMETHOD *Sigmoid)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ double x);
    double (ECOCALLMETHOD *SigmoidDerivative)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ double x);

    double (ECOCALLMETHOD *ReLU)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ double x);
    double (ECOCALLMETHOD *ReLUDerivative)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ double x);
    double (ECOCALLMETHOD *LeakyReLU)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ double x, /* in */ double alpha);
    double (ECOCALLMETHOD *LeakyReLUDerivative)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ double x, /* in */ double alpha);

    int16_t (ECOCALLMETHOD *Softmax)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ const EcoVector* input, /* out */ EcoVector* output);
    int16_t (ECOCALLMETHOD *SoftmaxJacobian)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ const EcoVector* input, /* out */ EcoMatrix* jacobian);
    int16_t (ECOCALLMETHOD *SoftmaxVectorGrad)(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ const EcoVector* input, /* in */ const EcoVector* grad_output, /* out */ EcoVector* grad_input);
} IEcoActivationFunctionsVTbl, *IEcoActivationFunctionsVTblPtr_t;


typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_SOFTMAX
} ActivationType;

// Уникальный IID для нейронного слоя (отличается от предыдущего)
#ifndef __IID_IEcoNeuralLayer
static const UGUID IID_IEcoNeuralLayer = { 0x01, 0x10, {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x01} };
#endif

typedef struct IEcoNeuralLayer* IEcoNeuralLayerPtr_t;

// Виртуальная таблица методов нейронного слоя
typedef struct IEcoNeuralLayerVTbl {
    int16_t (ECOCALLMETHOD *QueryInterface)(/* in */ IEcoNeuralLayerPtr_t me, /* in */ const UGUID* riid, /* out */ void** ppv);
    uint32_t (ECOCALLMETHOD *AddRef)(/* in */ IEcoNeuralLayerPtr_t me);
    uint32_t (ECOCALLMETHOD *Release)(/* in */ IEcoNeuralLayerPtr_t me);
    int16_t (ECOCALLMETHOD *Init)(/* in */ IEcoNeuralLayerPtr_t me, /* in */ ActivationType type, /* in */ double alpha, /* in */ size_t size);
    int16_t (ECOCALLMETHOD *Forward)(/* in */ IEcoNeuralLayerPtr_t me, /* in */ const EcoVector* input, /* out */ EcoVector* output);
    int16_t (ECOCALLMETHOD *Backward)(/* in */ IEcoNeuralLayerPtr_t me, /* in */ const EcoVector* grad_output, /* out */ EcoVector* grad_input);
} IEcoNeuralLayerVTbl;

struct IEcoNeuralLayer {
    struct IEcoNeuralLayerVTbl *pVTbl;
};

struct IEcoActivationFunctions {
    struct IEcoActivationFunctionsVTbl *pVTbl;
};

#endif /* __I_ECO_ACTIVATION_FUNCTIONS_H__ */