cmake_minimum_required(VERSION 3.10)

project(EcoLab1 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)



include_directories(
    include
    HeaderFiles
    SharedFiles
    Eco.Framework/Eco.Core1/SharedFiles
)

set(SOURCES
    SourceFiles/CEcoActivationFunctions.c
    SourceFiles/CEcoNeuralLayer.c
    SourceFiles/EcoMathMathUtils.c
    SourceFiles/CEcoLab1DPFactory.c
    UnitTestFiles/SourceFiles/EcoLab1DP.c
    # Здесь нужно также перечислить .c файлы из Eco.Framework, которые нужны вашему проекту, например:
    # Eco.Framework/Eco.Core1/SourceFiles/CEcoSystem1.c
    # и т.д.
)

add_executable(EcoLab1 ${SOURCES})