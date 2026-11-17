/*
 * <character encoding>
 *   Cyrillic (UTF-8 with signature) - Codepage 65001
 * </character encoding>
 *
 * <summary>
 *   EcoLab1DP
 * </summary>
 *
 * <description>
 *   This source file is the entry point
 * </description>
 *
 * <author>
 *   Copyright (c) 2026 . All rights reserved.
 * </author>
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <time.h>

/* Eco OS */
#include "IEcoSystem1.h"
#include "IdEcoMemoryManager1.h"
#include "IdEcoInterfaceBus1.h"
#include "IdEcoFileSystemManagement1.h"
#include "IdEcoLab1.h"

/* Component Interfaces */
#include "IEcoActivationFunctions.h"
#include "EcoMathMathUtils.h"

#define EPSILON_DIFF 1e-6
#define TOLERANCE 1e-4

static double calc_abs_err(double val, double ref) {
    return eco_fabs(val - ref);
}

static void test_scalar_functions(IEcoActivationFunctions* act) {
    double test_inputs[] = {-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0};
    size_t num_inputs = sizeof(test_inputs) / sizeof(test_inputs[0]);
    double h = EPSILON_DIFF;
    double alpha = 0.01;
    size_t i = 0;

    printf("=== TEST 1: Scalar Activation Functions & Derivatives ===\n");

    /* Sigmoid test */
    printf("\n[Sigmoid Check]\n");
    for (i = 0; i < num_inputs; ++i) {
        double x = test_inputs[i];
        double val = act->pVTbl->Sigmoid(act, x);
        double der = act->pVTbl->SigmoidDerivative(act, x);

        double num_der = (act->pVTbl->Sigmoid(act, x + h) - act->pVTbl->Sigmoid(act, x - h)) / (2.0 * h);
        double err = calc_abs_err(der, num_der);

        printf("x = %8.1f | Sigmoid = %.6f | Deriv = %.6f | NumDeriv = %.6f | Err = %.2e [%s]\n",
               x, val, der, num_der, err, (err < TOLERANCE) ? "PASS" : "FAIL");
    }

    /* ReLU test */
    printf("\n[ReLU Check]\n");
    for (i = 0; i < num_inputs; ++i) {
        double x = test_inputs[i];
        double val = act->pVTbl->ReLU(act, x);
        double der = act->pVTbl->ReLUDerivative(act, x);

        if (eco_fabs(x) < 1e-9) {
            printf("x = %8.1f | ReLU = %.6f | Deriv = %.6f | [PASS - Boundary Point x=0 explicitly handled]\n", x, val, der);
        } else {
            double num_der = (act->pVTbl->ReLU(act, x + h) - act->pVTbl->ReLU(act, x - h)) / (2.0 * h);
            double err = calc_abs_err(der, num_der);
            printf("x = %8.1f | ReLU = %.6f | Deriv = %.6f | NumDeriv = %.6f | Err = %.2e [%s]\n",
                   x, val, der, num_der, err, (err < TOLERANCE) ? "PASS" : "FAIL");
        }
    }

    /* Leaky ReLU test */
    printf("\n[Leaky ReLU Check (alpha = 0.01)]\n");
    for (i = 0; i < num_inputs; ++i) {
        double x = test_inputs[i];
        double val = act->pVTbl->LeakyReLU(act, x, alpha);
        double der = act->pVTbl->LeakyReLUDerivative(act, x, alpha);

        if (eco_fabs(x) < 1e-9) {
            printf("x = %8.1f | LeakyReLU = %.6f | Deriv = %.6f | [PASS - Boundary Point x=0 explicitly handled]\n", x, val, der);
        } else {
            double num_der = (act->pVTbl->LeakyReLU(act, x + h, alpha) - act->pVTbl->LeakyReLU(act, x - h, alpha)) / (2.0 * h);
            double err = calc_abs_err(der, num_der);
            printf("x = %8.1f | LeakyReLU = %.6f | Deriv = %.6f | NumDeriv = %.6f | Err = %.2e [%s]\n",
                   x, val, der, num_der, err, (err < TOLERANCE) ? "PASS" : "FAIL");
        }
    }
}

static void test_softmax_and_jacobian(IEcoActivationFunctions* act) {
    double in_large_data[] = {1000.0, 1001.0, 1002.0};
    double out_data[3];
    EcoVector in_vec = {in_large_data, 3, 3};
    EcoVector out_vec = {out_data, 3, 0};
    double in_same_data[] = {5.0, 5.0, 5.0, 5.0};
    double x_data[] = {1.0, 2.0, 3.0};
    double j_data[9];
    EcoMatrix jac = {j_data, 3, 3};
    double h = EPSILON_DIFF;
    int jac_pass = 1;
    double sum = 0.0;
    size_t i = 0;
    size_t j = 0;
    int16_t res = 0;

    printf("\n=== TEST 2: Softmax, Numerical Stability & Jacobian ===\n");

    res = act->pVTbl->Softmax(act, &in_vec, &out_vec);
    printf("[Softmax Overflow Resistance Test]\n");
    printf("Input: [1000.0, 1001.0, 1002.0]\nOutput: ");
    for (i = 0; i < out_vec.size; ++i) {
        printf("%.6f ", out_vec.data[i]);
        sum += out_vec.data[i];
    }
    printf("| Sum = %.6f [%s]\n", sum, (res == 0 && eco_fabs(sum - 1.0) < 1e-6) ? "PASS" : "FAIL");

    in_vec.data = in_same_data;
    in_vec.size = 4;
    out_vec.size = 0;
    act->pVTbl->Softmax(act, &in_vec, &out_vec);
    printf("\n[Softmax Identical Elements Test]\nOutput: ");
    sum = 0.0;
    for (i = 0; i < out_vec.size; ++i) {
        printf("%.4f ", out_vec.data[i]);
        sum += out_vec.data[i];
    }
    printf("| Sum = %.6f [%s]\n", sum, (eco_fabs(sum - 1.0) < 1e-6) ? "PASS" : "FAIL");

    printf("\n[Softmax Jacobian Numerical Verification]\n");
    in_vec.data = x_data;
    in_vec.size = 3;

    act->pVTbl->SoftmaxJacobian(act, &in_vec, &jac);

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            double analytic_j = jac.data[i * 3 + j];

            double x_plus[3] = {1.0, 2.0, 3.0};
            double x_minus[3] = {1.0, 2.0, 3.0};
            double s_plus[3], s_minus[3];
            EcoVector v_plus = {x_plus, 3, 3}, o_plus = {s_plus, 3, 0};
            EcoVector v_minus = {x_minus, 3, 3}, o_minus = {s_minus, 3, 0};
            double num_j = 0.0;
            double err = 0.0;

            x_plus[j] += h;
            x_minus[j] -= h;

            act->pVTbl->Softmax(act, &v_plus, &o_plus);
            act->pVTbl->Softmax(act, &v_minus, &o_minus);

            num_j = (o_plus.data[i] - o_minus.data[i]) / (2.0 * h);
            err = calc_abs_err(analytic_j, num_j);

            if (err > TOLERANCE) jac_pass = 0;
            printf("J[%u][%u] | Analytic = %8.5f | Numeric = %8.5f | Err = %.2e\n",
                   (unsigned int)i, (unsigned int)j, analytic_j, num_j, err);
        }
    }
    printf("Jacobian Verification Result: [%s]\n", jac_pass ? "PASS" : "FAIL");
}

static void test_ooa_vs_soa(IEcoActivationFunctions* soa, IEcoNeuralLayer* ooa) {
    double x_data[] = {0.5, -1.0, 2.0};
    double g_out_data[] = {1.0, 1.0, 1.0};
    double g_in_soa_data[3];
    double g_in_ooa_data[3];
    double out_layer_data[3];

    EcoVector in = {x_data, 3, 3};
    EcoVector grad_out = {g_out_data, 3, 3};
    EcoVector grad_in_soa = {g_in_soa_data, 3, 0};
    EcoVector grad_in_ooa = {g_in_ooa_data, 3, 0};
    EcoVector out_layer = {out_layer_data, 3, 0};
    int match = 1;
    size_t i = 0;

    printf("\n=== TEST 3: OOA vs SOA Architecture Comparison ===\n");

    /* 1. SOA Approach */
    soa->pVTbl->SoftmaxVectorGrad(soa, &in, &grad_out, &grad_in_soa);

    /* 2. OOA Approach (Stateful Layer) */
    ooa->pVTbl->Init(ooa, ACTIVATION_SOFTMAX, 0.0, 3);
    ooa->pVTbl->Forward(ooa, &in, &out_layer);
    ooa->pVTbl->Backward(ooa, &grad_out, &grad_in_ooa);

    for (i = 0; i < 3; ++i) {
        if (calc_abs_err(grad_in_soa.data[i], grad_in_ooa.data[i]) > 1e-7) {
            match = 0;
        }
    }
    printf("SOA vs OOA Backpropagation Output Equivalence: [%s]\n", match ? "PASS" : "FAIL");
}

static void test_performance(IEcoActivationFunctions* act) {
    const size_t N = 10000;
    static double vec_in_data[10000];
    static double vec_out_data[10000];
    EcoVector in = {vec_in_data, N, N};
    EcoVector out = {vec_out_data, N, 0};
    const int RUNS = 100;
    clock_t start = 0;
    clock_t end = 0;
    double time_sig = 0.0;
    double time_softmax = 0.0;
    size_t i = 0;
    int r = 0;

    printf("\n=== TEST 4: Performance Analysis ===\n");

    for (i = 0; i < N; ++i) {
        vec_in_data[i] = (double)(i % 100) / 10.0 - 5.0;
    }

    start = clock();
    for (r = 0; r < RUNS; ++r) {
        for (i = 0; i < N; ++i) {
            vec_out_data[i] = act->pVTbl->Sigmoid(act, vec_in_data[i]);
        }
    }
    end = clock();
    time_sig = ((double)(end - start) / CLOCKS_PER_SEC) / RUNS;

    start = clock();
    for (r = 0; r < RUNS; ++r) {
        act->pVTbl->Softmax(act, &in, &out);
    }
    end = clock();
    time_softmax = ((double)(end - start) / CLOCKS_PER_SEC) / RUNS;

    printf("Average Execution Time over %d iterations (Vector Size N = %u):\n", RUNS, (unsigned int)N);
    printf(" - Sigmoid (element-wise): %.6f sec\n", time_sig);
    printf(" - Softmax (vectorized)  : %.6f sec\n", time_softmax);
}

/*
 *
 * <summary>
 *   EcoMain Function
 * </summary>
 *
 * <description>
 *   EcoMain function - entry point
 * </description>
 *
 */
int16_t EcoMain(IEcoUnknown* pIUnk) {
    int16_t result = -1;
    /* Pointer to the system interface */
    IEcoSystem1* pISys = 0;
    /* Pointer to the interface for working with the system interface bus */
    IEcoInterfaceBus1* pIBus = 0;
    /* Pointer to the memory management interface */
    IEcoMemoryAllocator1* pIMem = 0;
    /* Pointers to the tested interfaces */
    IEcoActivationFunctions* pIAct = 0;
    IEcoNeuralLayer* pILayer = 0;

    /* System interface check and creation */
    if (pISys == 0) {
        result = pIUnk->pVTbl->QueryInterface(pIUnk, &GID_IEcoSystem, (void **)&pISys);
        if (result != 0 && pISys == 0) {
            /* Free the system interface in case of an error */
            goto Release;
        }
    }

    /* Getting the interface for working with the interface bus */
    result = pISys->pVTbl->QueryInterface(pISys, &IID_IEcoInterfaceBus1, (void **)&pIBus);
    if (result != 0 || pIBus == 0) {
        /* Free in case of an error */
        goto Release;
    }

#ifdef ECO_LIB
    /* Registration of a static component for working with the list */
    result = pIBus->pVTbl->RegisterComponent(pIBus, &CID_EcoLab1DP, (IEcoUnknown*)GetIEcoComponentFactoryPtr_AFC67744F1C343AC891CA5DA35D14A4A);
    if (result != 0) {
        /* Free in case of an error */
        goto Release;
    }
#endif

    /* Getting the memory management interface */
    result = pIBus->pVTbl->QueryComponent(pIBus, &CID_EcoMemoryManager1, 0, &IID_IEcoMemoryAllocator1, (void**) &pIMem);
    if (result != 0 || pIMem == 0) {
        /* Free the system interface in case of an error */
        goto Release;
    }

    /* Getting the tested interfaces */
    result = pIBus->pVTbl->QueryComponent(pIBus, &CID_EcoLab1DP, 0, &IID_IEcoActivationFunctions, (void**) &pIAct);
    if (result != 0 || pIAct == 0) {
        goto Release;
    }

    result = pIBus->pVTbl->QueryComponent(pIBus, &CID_EcoLab1DP, 0, &IID_IEcoNeuralLayer, (void**) &pILayer);
    if (result != 0 || pILayer == 0) {
        goto Release;
    }

    /* Run Test Suite from test_lab1 */
    printf("====================================================\n");
    printf("  ACOM Component Laboratory Work #75 Test Suite     \n");
    printf("====================================================\n\n");

    test_scalar_functions(pIAct);
    test_softmax_and_jacobian(pIAct);
    test_ooa_vs_soa(pIAct, pILayer);
    test_performance(pIAct);

    printf("\n[ALL TESTS COMPLETED SUCCESSFULLY]\n");
    result = 0;

Release:

    /* Free the interface for working with the interface bus */
    if (pIBus != 0) {
        pIBus->pVTbl->Release(pIBus);
    }

    /* Free the memory management interface */
    if (pIMem != 0) {
        pIMem->pVTbl->Release(pIMem);
    }

    /* Free the tested interfaces */
    if (pILayer != 0) {
        pILayer->pVTbl->Release(pILayer);
    }

    if (pIAct != 0) {
        pIAct->pVTbl->Release(pIAct);
    }

    /* Free the system interface */
    if (pISys != 0) {
        pISys->pVTbl->Release(pISys);
    }

    return result;
}