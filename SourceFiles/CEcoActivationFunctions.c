/*
 * <character encoding>
 *   Cyrillic (UTF-8 with signature) - Codepage 65001
 * </character encoding>
 *
 * <summary>
 *   CEcoLab1DP_35D14A4A
 * </summary>
 *
 * <description>
 *   This source code describes the implementation of the activation functions interface for CEcoLab1DP_35D14A4A
 * </description>
 *
 * <author>
 *   Copyright (c) 2026 . All rights reserved.
 * </author>
 *
 */

#include "IEcoSystem1.h"
#include "IEcoInterfaceBus1.h"
#include "IEcoInterfaceBus1MemExt.h"
#include "IEcoActivationFunctions.h"
#include "EcoMathMathUtils.h"
#include "CEcoLab1DP.h"
#include <stdlib.h>

/*
 *
 * <summary>
 *   QueryInterface Function
 * </summary>
 *
 * <description>
 *   QueryInterface function for the IEcoActivationFunctions interface
 * </description>
 *
 */
static int16_t ECOCALLMETHOD CEcoLab1DP_35D14A4A_QueryInterface(/* in */ IEcoActivationFunctionsPtr_t me, /* in */ const UGUID* riid, /* out */ void** ppv) {
    CEcoLab1DP_35D14A4A* pCMe = (CEcoLab1DP_35D14A4A*)me;

    /* Pointer Validation */
    if (me == 0 || ppv == 0) {
        return ERR_ECO_POINTER;
    }

    /* Validate and retrieve requested interface */
    if ( IsEqualUGUID(riid, &IID_IEcoActivationFunctions) ) {
        *ppv = &pCMe->m_pVTblIEcoActivationFunctions;
        pCMe->m_pVTblIEcoActivationFunctions->AddRef((IEcoActivationFunctions*)pCMe);
    }
    else if ( IsEqualUGUID(riid, &IID_IEcoUnknown) ) {
        *ppv = &pCMe->m_pVTblIEcoActivationFunctions;
        pCMe->m_pVTblIEcoActivationFunctions->AddRef((IEcoActivationFunctions*)pCMe);
    }
    else {
        *ppv = 0;
        return ERR_ECO_NOINTERFACE;
    }
    return ERR_ECO_SUCCESS;
}

/*
 *
 * <summary>
 *   AddRef Function
 * </summary>
 *
 */
static uint32_t ECOCALLMETHOD CEcoLab1DP_35D14A4A_AddRef(/* in */ IEcoActivationFunctionsPtr_t me) {
    CEcoLab1DP_35D14A4A* pCMe = (CEcoLab1DP_35D14A4A*)me;

    /* Pointer Validation */
    if (me == 0 ) {
        return (uint32_t)-1; /* ERR_ECO_POINTER */
    }

    return ++pCMe->m_cRef;
}

/*
 *
 * <summary>
 *   Release Function
 * </summary>
 *
 */
static uint32_t ECOCALLMETHOD CEcoLab1DP_35D14A4A_Release(/* in */ IEcoActivationFunctionsPtr_t me) {
    CEcoLab1DP_35D14A4A* pCMe = (CEcoLab1DP_35D14A4A*)me;

    /* Pointer Validation */
    if (me == 0 ) {
        return (uint32_t)-1; /* ERR_ECO_POINTER */
    }

    /* Decrementing the component's reference count */
    --pCMe->m_cRef;
    /* If the count is zero, free the instance data */
    if ( pCMe->m_cRef == 0 ) {
        pCMe->Delete(pCMe);
        return 0;
    }
    return pCMe->m_cRef;
}

/* ==========================================
 * Activation Functions Implementation[cite: 2]
 * ========================================== */

static double ECOCALLMETHOD CEcoLab1DP_35D14A4A_Sigmoid(IEcoActivationFunctionsPtr_t me, double x) {
    (void)me;
    if (x >= 0.0) {
        double z = eco_exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = eco_exp(x);
        return z / (1.0 + z);
    }
}

static double ECOCALLMETHOD CEcoLab1DP_35D14A4A_SigmoidDerivative(IEcoActivationFunctionsPtr_t me, double x) {
    double s = me->pVTbl->Sigmoid(me, x);
    return s * (1.0 - s);
}

static double ECOCALLMETHOD CEcoLab1DP_35D14A4A_ReLU(IEcoActivationFunctionsPtr_t me, double x) {
    (void)me;
    return (x > 0.0) ? x : 0.0;
}

static double ECOCALLMETHOD CEcoLab1DP_35D14A4A_ReLUDerivative(IEcoActivationFunctionsPtr_t me, double x) {
    (void)me;
    return (x > 0.0) ? 1.0 : 0.0;
}

static double ECOCALLMETHOD CEcoLab1DP_35D14A4A_LeakyReLU(IEcoActivationFunctionsPtr_t me, double x, double alpha) {
    (void)me;
    return (x >= 0.0) ? x : (alpha * x);
}

static double ECOCALLMETHOD CEcoLab1DP_35D14A4A_LeakyReLUDerivative(IEcoActivationFunctionsPtr_t me, double x, double alpha) {
    (void)me;
    return (x > 0.0) ? 1.0 : alpha;
}

static int16_t ECOCALLMETHOD CEcoLab1DP_35D14A4A_Softmax(IEcoActivationFunctionsPtr_t me, const EcoVector* input, EcoVector* output) {
    (void)me;
    if (me == NULL || input == NULL || output == NULL) return -1;
    if (input->data == NULL || output->data == NULL || input->size == 0) return -1;

    size_t n = input->size;
    double max_val = input->data[0];
    for (size_t i = 1; i < n; ++i) {
        if (input->data[i] > max_val) max_val = input->data[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        output->data[i] = eco_exp(input->data[i] - max_val);
        sum += output->data[i];
    }

    if (sum == 0.0) return -1;
    for (size_t i = 0; i < n; ++i) {
        output->data[i] /= sum;
    }
    output->size = n;
    return 0;
}

static int16_t ECOCALLMETHOD CEcoLab1DP_35D14A4A_SoftmaxJacobian(IEcoActivationFunctionsPtr_t me, const EcoVector* input, EcoMatrix* jacobian) {
    if (me == NULL || input == NULL || jacobian == NULL) return -1;
    size_t n = input->size;

    double* stack_s = (double*)malloc(n * sizeof(double));
    if (!stack_s) return -1;

    EcoVector s_vec = { stack_s, n, 0 };
    if (me->pVTbl->Softmax(me, input, &s_vec) != 0) {
        free(stack_s);
        return -1;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double kronecker = (i == j) ? 1.0 : 0.0;
            jacobian->data[i * n + j] = s_vec.data[i] * (kronecker - s_vec.data[j]);
        }
    }
    jacobian->rows = n;
    jacobian->cols = n;

    free(stack_s);
    return 0;
}

static int16_t ECOCALLMETHOD CEcoLab1DP_35D14A4A_SoftmaxVectorGrad(IEcoActivationFunctionsPtr_t me, const EcoVector* input, const EcoVector* grad_output, EcoVector* grad_input) {
    if (me == NULL || input == NULL || grad_output == NULL || grad_input == NULL) return -1;
    size_t n = input->size;

    double* stack_s = (double*)malloc(n * sizeof(double));
    if (!stack_s) return -1;

    EcoVector s_vec = { stack_s, n, 0 };
    if (me->pVTbl->Softmax(me, input, &s_vec) != 0) {
        free(stack_s);
        return -1;
    }

    double dot = 0.0;
    for (size_t j = 0; j < n; ++j) {
        dot += s_vec.data[j] * grad_output->data[j];
    }

    for (size_t i = 0; i < n; ++i) {
        grad_input->data[i] = s_vec.data[i] * (grad_output->data[i] - dot);
    }
    grad_input->size = n;

    free(stack_s);
    return 0;
}

/*
 *
 * <summary>
 *   Init Function
 * </summary>
 *
 * <description>
 *   Instance initialization function[cite: 1]
 * </description>
 *
 */
static int16_t ECOCALLMETHOD initCEcoLab1DP_35D14A4A(/*in*/ CEcoLab1DP_35D14A4APtr_t me, /* in */ IEcoUnknownPtr_t pIUnkSystem) {
    CEcoLab1DP_35D14A4A* pCMe = (CEcoLab1DP_35D14A4A*)me;
    IEcoInterfaceBus1* pIBus = 0;
    IEcoInterfaceBus1MemExt* pIMemExt = 0;
    int16_t result = ERR_ECO_POINTER;
    UGUID* rcid = (UGUID*)&CID_EcoMemoryManager1;

    /* Pointer Validation */
    if (me == 0 ) {
        return result;
    }

    /* Storing the pointer to the system interface */
    pCMe->m_pISys = (IEcoSystem1*)pIUnkSystem;

    /* Getting the interface for working with the interface bus */
    result = pCMe->m_pISys->pVTbl->QueryInterface(pCMe->m_pISys, &IID_IEcoInterfaceBus1, (void **)&pIBus);

    /* Getting the component ID for working with memory */
    result = pIBus->pVTbl->QueryInterface(pIBus, &IID_IEcoInterfaceBus1MemExt, (void**)&pIMemExt);
    if (result == 0 && pIMemExt != 0) {
        rcid = (UGUID*)pIMemExt->pVTbl->get_Manager(pIMemExt);
        pIMemExt->pVTbl->Release(pIMemExt);
    }

    /* Getting the memory allocator interface */
    result = pIBus->pVTbl->QueryComponent(pIBus, rcid, 0, &IID_IEcoMemoryAllocator1, (void**) &pCMe->m_pIMem);
    /* Check */
    if (result != 0 || pCMe->m_pIMem == 0) {
        result = ERR_ECO_GET_MEMORY_ALLOCATOR;
    }

    /* Freeing */
    pIBus->pVTbl->Release(pIBus);

    return result;
}

/*
 *
 * <summary>
 *   Create Function
 * </summary>
 *
 * <description>
 *   Instance creation function[cite: 1]
 * </description>
 *
 */
static int16_t ECOCALLMETHOD createCEcoLab1DP_35D14A4A(/* in */ CEcoLab1DP_35D14A4APtr_t pCMe, /* in */ IEcoUnknownPtr_t pIUnkSystem, /* in */ IEcoUnknownPtr_t pIUnkOuter) {
    int16_t result = ERR_ECO_POINTER;

    /* Pointer Validation */
    if (pCMe == 0) {
        return result; /* ERR_ECO_POINTER */
    }

    return ERR_ECO_SUCCESS;
}

/*
 *
 * <summary>
 *   Delete Function
 * </summary>
 *
 * <description>
 *   Instance freeing function[cite: 1]
 * </description>
 *
 */
static void ECOCALLMETHOD deleteCEcoLab1DP_35D14A4A(/* in */ CEcoLab1DP_35D14A4APtr_t pCMe) {
    IEcoMemoryAllocator1* pIMem = 0;

    if (pCMe != 0 ) {
        pIMem = pCMe->m_pIMem;
        /* Freeing */
        if ( pCMe->m_pISys != 0 ) {
            pCMe->m_pISys->pVTbl->Release(pCMe->m_pISys);
        }
        pIMem->pVTbl->Free(pIMem, pCMe);
        pIMem->pVTbl->Release(pIMem);
    }
}

/* IEcoActivationFunctions Virtual Table */
IEcoActivationFunctionsVTbl g_x4EF92ACD86564B68821E2E480C0651D9VTbl_35D14A4A = {
    CEcoLab1DP_35D14A4A_QueryInterface,
    CEcoLab1DP_35D14A4A_AddRef,
    CEcoLab1DP_35D14A4A_Release,
    CEcoLab1DP_35D14A4A_Sigmoid,
    CEcoLab1DP_35D14A4A_SigmoidDerivative,
    CEcoLab1DP_35D14A4A_ReLU,
    CEcoLab1DP_35D14A4A_ReLUDerivative,
    CEcoLab1DP_35D14A4A_LeakyReLU,
    CEcoLab1DP_35D14A4A_LeakyReLUDerivative,
    CEcoLab1DP_35D14A4A_Softmax,
    CEcoLab1DP_35D14A4A_SoftmaxJacobian,
    CEcoLab1DP_35D14A4A_SoftmaxVectorGrad
};

/* Object Instance[cite: 1] */
CEcoLab1DP_35D14A4A g_xCEcoLab1DP_35D14A4A = {
    &g_x4EF92ACD86564B68821E2E480C0651D9VTbl_35D14A4A,
    initCEcoLab1DP_35D14A4A,
    createCEcoLab1DP_35D14A4A,
    deleteCEcoLab1DP_35D14A4A,
    1, /* m_cRef */
    0, /* m_pIMem */
    0  /* m_pISys */
};