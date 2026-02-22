#include <iostream>
#include <cpuid.h>

void check_cpuid() {
    unsigned int eax, ebx, ecx, edx;

    // Check for AVX-512 Foundation (EBX bit 16 of CPUID leaf 7, subleaf 0)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        std::cout << "🔍 CPUID Leaf 7 Subleaf 0 Check:" << std::endl;
        std::cout << " ├─ AVX512F (Foundation): "  << ((ebx >> 16) & 1 ? "✅ YES" : "❌ NO") << std::endl;
        std::cout << " ├─ AVX512DQ: " << ((ebx >> 17) & 1 ? "✅ YES" : "❌ NO") << std::endl;
        std::cout << " ├─ AVX512BW: " << ((ebx >> 30) & 1 ? "✅ YES" : "❌ NO") << std::endl;
        std::cout << " └─ AVX512VL: " << ((ebx >> 31) & 1 ? "✅ YES" : "❌ NO") << std::endl;
        
        std::cout << "\n🔍 CPUID Leaf 1 Check:" << std::endl;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        std::cout << " ├─ AVX: " << ((ecx >> 28) & 1 ? "✅ YES" : "❌ NO") << std::endl;
        std::cout << " └─ FMA: " << ((ecx >> 12) & 1 ? "✅ YES" : "❌ NO") << std::endl;
    } else {
        std::cout << "❌ Failed to read CPUID leaf 7" << std::endl;
    }
}

int main() {
    std::cout << "🧬 ZETAGRID CPUID DIAGNOSTIC" << std::endl;
    check_cpuid();
    return 0;
}
