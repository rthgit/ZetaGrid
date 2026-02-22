#!/bin/bash

echo "⚡ SETUP AMBIENTE COLAB (Tesla T4) ⚡"

# 1. Installazione Dipendenze OpenCL
echo "📦 Installazione OpenCL Headers..."
sudo apt-get update -qq
sudo apt-get install -y opencl-headers ocl-icd-opencl-dev

# 2. Compilazione
echo "🚀 Compilazione Motore Ibrido (CPU AVX2 + GPU OpenCL)..."
g++ -O3 -march=native -fopenmp morph_colab_gpu.cpp -o zed_colab_gpu -lOpenCL

# 3. Verifica
if [ -f "./zed_colab_gpu" ]; then
    echo "✅ Compilazione Riuscita!"
    echo "   Esegui il training con: ./zed_colab_gpu"
else
    echo "❌ Errore di Compilazione."
fi
