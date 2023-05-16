#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void* scalar_memcpy(void* dest, const void* src, size_t size) {
    char* dest_ptr = (char*)dest;
    const char* src_ptr = (const char*)src;
    for (size_t i = 0; i < size; ++i) {
        dest_ptr[i] = src_ptr[i];
    }

    return dest;
}

void* simd128_memcpy(void* dest, const void* src, size_t size) {
    char* dest_ptr = (char*)dest;
    const char* src_ptr = (const char*)src;

    size_t reminder = size % 16;
    size_t aligned_size = size - reminder;

    for (size_t i = 0; i < aligned_size / 16; ++i) {
        __m128i data = _mm_loadu_si128((__m128i*)&src_ptr[i * 16]);
        _mm_storeu_si128((__m128i*)&dest_ptr[i * 16], data);
    }

    return scalar_memcpy(dest_ptr + aligned_size, src_ptr + aligned_size, reminder);
}

void* simd256_memcpy(void* dest, const void* src, size_t size) {
    char* dest_ptr = (char*)dest;
    const char* src_ptr = (const char*)src;

    size_t reminder = size % 32;
    size_t aligned_size = size - reminder;

    for (size_t i = 0; i < aligned_size / 32; ++i) {
        __m256i data = _mm256_loadu_si256((__m256i*)&src_ptr[i * 32]);
        _mm256_storeu_si256((__m256i*)&dest_ptr[i * 32], data);
    }

    return scalar_memcpy(dest_ptr + aligned_size, src_ptr + aligned_size, reminder);
}

void* simd_combo_memcpy(void* dest, const void* src, size_t size) {
    char* dest_ptr = (char*)dest;
    const char* src_ptr = (const char*)src;

    size_t reminder = size % 32;
    size_t aligned_size = size - reminder;

    for (size_t i = 0; i < aligned_size / 32; ++i) {
        __m256i data = _mm256_loadu_si256((__m256i*)&src_ptr[i * 32]);
        _mm256_storeu_si256((__m256i*)&dest_ptr[i * 32], data);
    }

    return simd128_memcpy(dest_ptr + aligned_size, src_ptr + aligned_size, reminder);
}

void test_memcpy(char* dest, char* src, size_t size, char* title, void* (*memcpy_fn)(void*, const void*, size_t))
{
        for (int i = 0; i < size; ++i) {
            src[i] = i*i / size;
            dest[i] = (src[i] + 2) / 2;
        }

        clock_t clock_start = clock();
        memcpy_fn(dest, src, size);
        clock_t clock_end = clock();

        for (int i = 0; i < size; ++i) {
            if (src[i] != dest[i]) {
                printf("%s - bad values at %d: %d != %d\n", title, i, src[i], dest[i]);
            }
        }

        printf("%s: %lu\n", title, clock_end - clock_start);
}

int main() {
    size_t sizes[] = {
        32,
        32 * 32 * 32 * 32 * 32 * 32,
        32 * 32 * 32 * 32 * 32,
        32 * 32 * 32 * 32,
        32 * 32 * 32,
        16,
        16 * 16 * 16 * 16 * 16 * 16,
        16 * 16 * 16 * 16 * 16,
        16 * 16 * 16 * 16,
        16 * 16 * 16,
        17,
        19 * 117 * 211 * 117,
        17 *  19 * 117 * 211,
        19 * 117 * 211,
        17 * 117 * 211,
        17 *  19 * 117,
    };

    size_t count = sizeof(sizes) / sizeof(size_t);

    for (int size_idx = 0; size_idx < count; ++size_idx) {
        size_t size = sizes[size_idx];
        printf("================\nsize %lu\n", size * sizeof(char));

        char* src  = malloc(sizeof(char) * size);
        char* dest = malloc(sizeof(char) * size);

        test_memcpy(dest, src, size, "    scalar", &scalar_memcpy);
        test_memcpy(dest, src, size, "   simd128", &simd128_memcpy);
        test_memcpy(dest, src, size, "   simd256", &simd256_memcpy);
        test_memcpy(dest, src, size, "simd combo", &simd256_memcpy);
        test_memcpy(dest, src, size, "     c lib", &memcpy);

        free(src);
        free(dest);
    }

    printf("================\nDone\n");

    return 0;
}

