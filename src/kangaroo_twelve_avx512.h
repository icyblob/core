#pragma once


#ifdef __AVX512F__

#include <intrin.h>

#include "platform/memory.h"

typedef __m512i     V512;

#define XOR(a,b)                    _mm512_xor_si512(a,b)
#define XOR3(a,b,c)                 _mm512_ternarylogic_epi64(a,b,c,0x96)
#define XOR5(a,b,c,d,e)             XOR3(XOR3(a,b,c),d,e)
#define ROL(a,offset)               _mm512_rol_epi64(a,offset)
#define Chi(a,b,c)                  _mm512_ternarylogic_epi64(a,b,c,0xD2)

#define LOAD_Lanes(m,a)             _mm512_maskz_loadu_epi64(m,a)
#define LOAD_Lane(a)                LOAD_Lanes(0x01,a)
#define LOAD_Plane(a)               LOAD_Lanes(0x1F,a)
#define LOAD_8Lanes(a)              LOAD_Lanes(0xFF,a)
#define STORE_Lanes(a,m,v)          _mm512_mask_storeu_epi64(a,m,v)
#define STORE_Lane(a,v)             STORE_Lanes(a,0x01,v)
#define STORE_Plane(a,v)            STORE_Lanes(a,0x1F,v)
#define STORE_8Lanes(a,v)           STORE_Lanes(a,0xFF,v)

/* ---------------------------------------------------------------- */

void KeccakP1600_AVX512_Initialize(void* state)
{
    setMem(state, 200, 0);
}

/* ---------------------------------------------------------------- */

void KeccakP1600_AVX512_AddBytes(void* state, const unsigned char* data, unsigned int offset, unsigned int length)
{
    unsigned char* stateAsBytes;
    unsigned long long* stateAsLanes;

    for (stateAsBytes = (unsigned char*)state; ((offset % 8) != 0) && (length != 0); ++offset, --length)
        stateAsBytes[offset] ^= *(data++);
    for (stateAsLanes = (unsigned long long*)(stateAsBytes + offset); length >= 8 * 8; stateAsLanes += 8, data += 8 * 8, length -= 8 * 8)
        STORE_8Lanes(stateAsLanes, XOR(LOAD_8Lanes(stateAsLanes), LOAD_8Lanes((const unsigned long long*)data)));
    for (/* empty */; length >= 8; ++stateAsLanes, data += 8, length -= 8)
        STORE_Lane(stateAsLanes, XOR(LOAD_Lane(stateAsLanes), LOAD_Lane((const unsigned long long*)data)));
    for (stateAsBytes = (unsigned char*)stateAsLanes; length != 0; --length)
        *(stateAsBytes++) ^= *(data++);
}

/* ---------------------------------------------------------------- */

void KeccakP1600_AVX512_ExtractBytes(const void* state, unsigned char* data, unsigned int offset, unsigned int length)
{
    copyMem(data, (unsigned char*)state + offset, length);
}

/* ---------------------------------------------------------------- */

const unsigned long long KeccakP1600RoundConstants[24] = {
    0x0000000000000001ULL,
    0x0000000000008082ULL,
    0x800000000000808aULL,
    0x8000000080008000ULL,
    0x000000000000808bULL,
    0x0000000080000001ULL,
    0x8000000080008081ULL,
    0x8000000000008009ULL,
    0x000000000000008aULL,
    0x0000000000000088ULL,
    0x0000000080008009ULL,
    0x000000008000000aULL,
    0x000000008000808bULL,
    0x800000000000008bULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800aULL,
    0x800000008000000aULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL };

#define KeccakP_DeclareVars \
    V512    b0, b1, b2, b3, b4; \
    V512    Baeiou, Gaeiou, Kaeiou, Maeiou, Saeiou; \
    V512    moveThetaPrev = _mm512_setr_epi64(4, 0, 1, 2, 3, 5, 6, 7); \
    V512    moveThetaNext = _mm512_setr_epi64(1, 2, 3, 4, 0, 5, 6, 7); \
    V512    rhoB = _mm512_setr_epi64( 0,  1, 62, 28, 27, 0, 0, 0); \
    V512    rhoG = _mm512_setr_epi64(36, 44,  6, 55, 20, 0, 0, 0); \
    V512    rhoK = _mm512_setr_epi64( 3, 10, 43, 25, 39, 0, 0, 0); \
    V512    rhoM = _mm512_setr_epi64(41, 45, 15, 21,  8, 0, 0, 0); \
    V512    rhoS = _mm512_setr_epi64(18,  2, 61, 56, 14, 0, 0, 0); \
    V512    pi1B = _mm512_setr_epi64(0, 3, 1, 4, 2, 5, 6, 7); \
    V512    pi1G = _mm512_setr_epi64(1, 4, 2, 0, 3, 5, 6, 7); \
    V512    pi1K = _mm512_setr_epi64(2, 0, 3, 1, 4, 5, 6, 7); \
    V512    pi1M = _mm512_setr_epi64(3, 1, 4, 2, 0, 5, 6, 7); \
    V512    pi1S = _mm512_setr_epi64(4, 2, 0, 3, 1, 5, 6, 7); \
    V512    pi2S1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 0+8, 2+8); \
    V512    pi2S2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 1+8, 3+8); \
    V512    pi2BG = _mm512_setr_epi64(0, 1, 0+8, 1+8, 6, 5, 6, 7); \
    V512    pi2KM = _mm512_setr_epi64(2, 3, 2+8, 3+8, 7, 5, 6, 7); \
    V512    pi2S3 = _mm512_setr_epi64(4, 5, 4+8, 5+8, 4, 5, 6, 7);

#define KeccakP_DeclareVars2 \
    V512    b0, b1, b2, b3, b4; \
    V512    moveThetaPrev = _mm512_setr_epi64(4, 0, 1, 2, 3, 5, 6, 7); \
    V512    moveThetaNext = _mm512_setr_epi64(1, 2, 3, 4, 0, 5, 6, 7); \
    V512    rhoB = _mm512_setr_epi64( 0,  1, 62, 28, 27, 0, 0, 0); \
    V512    rhoG = _mm512_setr_epi64(36, 44,  6, 55, 20, 0, 0, 0); \
    V512    rhoK = _mm512_setr_epi64( 3, 10, 43, 25, 39, 0, 0, 0); \
    V512    rhoM = _mm512_setr_epi64(41, 45, 15, 21,  8, 0, 0, 0); \
    V512    rhoS = _mm512_setr_epi64(18,  2, 61, 56, 14, 0, 0, 0); \
    V512    pi1B = _mm512_setr_epi64(0, 3, 1, 4, 2, 5, 6, 7); \
    V512    pi1G = _mm512_setr_epi64(1, 4, 2, 0, 3, 5, 6, 7); \
    V512    pi1K = _mm512_setr_epi64(2, 0, 3, 1, 4, 5, 6, 7); \
    V512    pi1M = _mm512_setr_epi64(3, 1, 4, 2, 0, 5, 6, 7); \
    V512    pi1S = _mm512_setr_epi64(4, 2, 0, 3, 1, 5, 6, 7); \
    V512    pi2S1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 0+8, 2+8); \
    V512    pi2S2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 1+8, 3+8); \
    V512    pi2BG = _mm512_setr_epi64(0, 1, 0+8, 1+8, 6, 5, 6, 7); \
    V512    pi2KM = _mm512_setr_epi64(2, 3, 2+8, 3+8, 7, 5, 6, 7); \
    V512    pi2S3 = _mm512_setr_epi64(4, 5, 4+8, 5+8, 4, 5, 6, 7);

#define copyFromState_AVX512(pState) \
    Baeiou = LOAD_Plane(pState+ 0); \
    Gaeiou = LOAD_Plane(pState+ 5); \
    Kaeiou = LOAD_Plane(pState+10); \
    Maeiou = LOAD_Plane(pState+15); \
    Saeiou = LOAD_Plane(pState+20);

#define copyToState_AVX512(pState) \
    STORE_Plane(pState+ 0, Baeiou); \
    STORE_Plane(pState+ 5, Gaeiou); \
    STORE_Plane(pState+10, Kaeiou); \
    STORE_Plane(pState+15, Maeiou); \
    STORE_Plane(pState+20, Saeiou);

#define KeccakP_Round(i) \
    /* Theta */ \
    b0 = XOR5( Baeiou, Gaeiou, Kaeiou, Maeiou, Saeiou ); \
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0); \
    b0 = _mm512_permutexvar_epi64(moveThetaNext, b0); \
    b0 = _mm512_rol_epi64(b0, 1); \
    Baeiou = XOR3( Baeiou, b0, b1 ); \
    Gaeiou = XOR3( Gaeiou, b0, b1 ); \
    Kaeiou = XOR3( Kaeiou, b0, b1 ); \
    Maeiou = XOR3( Maeiou, b0, b1 ); \
    Saeiou = XOR3( Saeiou, b0, b1 ); \
    /* Rho */ \
    Baeiou = _mm512_rolv_epi64(Baeiou, rhoB); \
    Gaeiou = _mm512_rolv_epi64(Gaeiou, rhoG); \
    Kaeiou = _mm512_rolv_epi64(Kaeiou, rhoK); \
    Maeiou = _mm512_rolv_epi64(Maeiou, rhoM); \
    Saeiou = _mm512_rolv_epi64(Saeiou, rhoS); \
    /* Pi 1 */ \
    b0 = _mm512_permutexvar_epi64(pi1B, Baeiou); \
    b1 = _mm512_permutexvar_epi64(pi1G, Gaeiou); \
    b2 = _mm512_permutexvar_epi64(pi1K, Kaeiou); \
    b3 = _mm512_permutexvar_epi64(pi1M, Maeiou); \
    b4 = _mm512_permutexvar_epi64(pi1S, Saeiou); \
    /* Chi */ \
    Baeiou = Chi(b0, b1, b2); \
    Gaeiou = Chi(b1, b2, b3); \
    Kaeiou = Chi(b2, b3, b4); \
    Maeiou = Chi(b3, b4, b0); \
    Saeiou = Chi(b4, b0, b1); \
    /* Iota */ \
    Baeiou = XOR(Baeiou, LOAD_Lane(KeccakP1600RoundConstants+i)); \
    /* Pi 2 */ \
    b0 = _mm512_unpacklo_epi64(Baeiou, Gaeiou); \
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou); \
    b0 = _mm512_permutex2var_epi64(b0, pi2S1, Saeiou); \
    b2 = _mm512_unpackhi_epi64(Baeiou, Gaeiou); \
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou); \
    b2 = _mm512_permutex2var_epi64(b2, pi2S2, Saeiou); \
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1); \
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3); \
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1); \
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3); \
    b0 = _mm512_permutex2var_epi64(b0, pi2S3, b1); \
    Saeiou = _mm512_mask_blend_epi64(0x10, b0, Saeiou)

#define rounds12_AVX512 \
    KeccakP_Round( 12 ); \
    KeccakP_Round( 13 ); \
    KeccakP_Round( 14 ); \
    KeccakP_Round( 15 ); \
    KeccakP_Round( 16 ); \
    KeccakP_Round( 17 ); \
    KeccakP_Round( 18 ); \
    KeccakP_Round( 19 ); \
    KeccakP_Round( 20 ); \
    KeccakP_Round( 21 ); \
    KeccakP_Round( 22 ); \
    KeccakP_Round( 23 )

/* ---------------------------------------------------------------- */

void KeccakP1600_AVX512_Permute_12rounds(void* state)
{
    KeccakP_DeclareVars
        unsigned long long* stateAsLanes = (unsigned long long*)state;

    copyFromState_AVX512(stateAsLanes);
    rounds12_AVX512;
    copyToState_AVX512(stateAsLanes);
}

/* ---------------------------------------------------------------- */
//
//
//size_t KeccakP1600_AVX512_12rounds_FastLoop_Absorb(void* state, unsigned int laneCount, const unsigned char* data, size_t dataByteLen)
//{
//    size_t originalDataByteLen = dataByteLen;
//
//    assert(laneCount == 21 || laneCount == 17);
//
//    KeccakP_DeclareVars;
//    unsigned long long* stateAsLanes = (unsigned long long*)state;
//    unsigned long long* inDataAsLanes = (unsigned long long*)data;
//
//    if (laneCount == 21) {
//#define laneCount 21
//        copyFromState(stateAsLanes);
//        while (dataByteLen >= 21 * 8) {
//            Baeiou = XOR(Baeiou, LOAD_Plane(inDataAsLanes + 0));
//            Gaeiou = XOR(Gaeiou, LOAD_Plane(inDataAsLanes + 5));
//            Kaeiou = XOR(Kaeiou, LOAD_Plane(inDataAsLanes + 10));
//            Maeiou = XOR(Maeiou, LOAD_Plane(inDataAsLanes + 15));
//            Saeiou = XOR(Saeiou, LOAD_Lane(inDataAsLanes + 20));
//            rounds12;
//            inDataAsLanes += 21;
//            dataByteLen -= 21 * 8;
//        }
//#undef laneCount
//        copyToState(stateAsLanes);
//    }
//    else if (laneCount == 17) {
//        // TODO: further optimization needed for this case, laneCount == 17.
//        while (dataByteLen >= laneCount * 8) {
//            KeccakP1600_AddBytes(state, data, 0, laneCount * 8);
//            KeccakP1600_Permute_12rounds(state);
//            data += laneCount * 8;
//            dataByteLen -= laneCount * 8;
//        }
//    }
//
//    return originalDataByteLen - dataByteLen;
//}

static void randomAVX512(const unsigned char* publicKey, const unsigned char* nonce, unsigned char* output, unsigned long long outputSize)
{
    unsigned char state[200];
    *((__m256i*) & state[0]) = *((__m256i*)publicKey);
    *((__m256i*) & state[32]) = *((__m256i*)nonce);
    setMem(&state[64], sizeof(state) - 64, 0);

    for (unsigned long long i = 0; i < outputSize / sizeof(state); i++)
    {
        KeccakP1600_AVX512_Permute_12rounds(state);
        copyMem(output, state, sizeof(state));
        output += sizeof(state);
    }
    if (outputSize % sizeof(state))
    {
        KeccakP1600_AVX512_Permute_12rounds(state);
        copyMem(output, state, outputSize % sizeof(state));
    }
}

#endif
