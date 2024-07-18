#pragma once


#ifdef __AVX512F__

/*
* This part is get from
https://github.com/XKCP/K12/blob/master/lib/Optimized64/KeccakP-1600-AVX512-plainC.c

K12 based on the eXtended Keccak Code Package (XKCP)
https://github.com/XKCP/XKCP

The Keccak-p permutations, designed by Guido Bertoni, Joan Daemen, Michaël Peeters and Gilles Van Assche.

Implementation by Ronny Van Keer, hereby denoted as "the implementer".

For more information, feedback or questions, please refer to the Keccak Team website:
https://keccak.team/

To the extent possible under law, the implementer has waived all copyright
and related or neighboring rights to the source code in this file.
http://creativecommons.org/publicdomain/zero/1.0/

---

We would like to thank Vladimir Sedach, we have used parts of his Keccak AVX-512 C++ code.
 */

#include <intrin.h>

#include "platform/memory.h"

#ifdef ALIGN
#undef ALIGN
#endif

#if defined(__GNUC__)
#define ALIGN(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#elif defined(__ARMCC_VERSION)
#define ALIGN(x) __align(x)
#else
#define ALIGN(x)
#endif

#define KeccakP1600_stateSizeInBytes    200
#define KeccakP1600_stateAlignment      8
// #define KeccakP1600_12rounds_FastLoop_supported

//#define K12_security        128
//#define K12_capacity        (2 * K12_security)
//#define K12_capacityInBytes (K12_capacity / 8)
//#define K12_rateInBytes     ((1600 - K12_capacity) / 8)
//#define K12_chunkSize       8192
//#define K12_suffixLeaf      0x0B


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

#define copyFromState_xkpc_AVX512(pState) \
    Baeiou = LOAD_Plane(pState+ 0); \
    Gaeiou = LOAD_Plane(pState+ 5); \
    Kaeiou = LOAD_Plane(pState+10); \
    Maeiou = LOAD_Plane(pState+15); \
    Saeiou = LOAD_Plane(pState+20);

#define copyToState_xkpc_AVX512(pState) \
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
    Baeiou = XOR(Baeiou, LOAD_Lane(xkpc_avx512::KeccakP1600RoundConstants+i)); \
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

#define rounds12_xkpc_AVX512 \
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


namespace xkpc_avx512
{

    /* ---------------------------------------------------------------- */

    void KeccakP1600_Initialize(void* state)
    {
        setMem(state, 200, 0);
    }

    /* ---------------------------------------------------------------- */

    void KeccakP1600_AddBytes(void* state, const unsigned char* data, unsigned int offset, unsigned int length)
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

    void KeccakP1600_ExtractBytes(const void* state, unsigned char* data, unsigned int offset, unsigned int length)
    {
        copyMem(data, (unsigned char*)state + offset, length);
    }

    void KeccakP1600_AddByte(void* state, unsigned char data, unsigned int offset)
    {
        ((unsigned char*)(state))[offset] ^= data;
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

    /* ---------------------------------------------------------------- */

    void KeccakP1600_Permute_12rounds(void* state)
    {
        KeccakP_DeclareVars
            unsigned long long* stateAsLanes = (unsigned long long*)state;

        copyFromState_xkpc_AVX512(stateAsLanes);
        rounds12_xkpc_AVX512;
        copyToState_xkpc_AVX512(stateAsLanes);
    }

    size_t KeccakP1600_12rounds_FastLoop_Absorb(void* state, unsigned int laneCount, const unsigned char* data, size_t dataByteLen)
    {
        size_t originalDataByteLen = dataByteLen;

        KeccakP_DeclareVars
        unsigned long long* stateAsLanes = (unsigned long long*)state;
        unsigned long long* inDataAsLanes = (unsigned long long*)data;

        if (laneCount == 21) {
#define laneCount 21
            copyToState_xkpc_AVX512(stateAsLanes);
            while (dataByteLen >= 21 * 8) {
                Baeiou = XOR(Baeiou, LOAD_Plane(inDataAsLanes + 0));
                Gaeiou = XOR(Gaeiou, LOAD_Plane(inDataAsLanes + 5));
                Kaeiou = XOR(Kaeiou, LOAD_Plane(inDataAsLanes + 10));
                Maeiou = XOR(Maeiou, LOAD_Plane(inDataAsLanes + 15));
                Saeiou = XOR(Saeiou, LOAD_Lane(inDataAsLanes + 20));
                rounds12_xkpc_AVX512;
                inDataAsLanes += 21;
                dataByteLen -= 21 * 8;
            }
#undef laneCount
            copyToState_xkpc_AVX512(stateAsLanes);
        }
        else if (laneCount == 17) {
            // TODO: further optimization needed for this case, laneCount == 17.
            while (dataByteLen >= laneCount * 8) {
                KeccakP1600_AddBytes(state, data, 0, laneCount * 8);
                KeccakP1600_Permute_12rounds(state);
                data += laneCount * 8;
                dataByteLen -= laneCount * 8;
            }
        }

        return originalDataByteLen - dataByteLen;
    }


    static void random(const unsigned char* publicKey, const unsigned char* nonce, unsigned char* output, unsigned long long outputSize)
    {
        unsigned char state[200];
        *((__m256i*) & state[0]) = *((__m256i*)publicKey);
        *((__m256i*) & state[32]) = *((__m256i*)nonce);
        setMem(&state[64], sizeof(state) - 64, 0);

        for (unsigned long long i = 0; i < outputSize / sizeof(state); i++)
        {
            KeccakP1600_Permute_12rounds(state);
            copyMem(output, state, sizeof(state));
            output += sizeof(state);
        }
        if (outputSize % sizeof(state))
        {
            KeccakP1600_Permute_12rounds(state);
            copyMem(output, state, outputSize % sizeof(state));
        }
    }

    typedef unsigned char uint8_t;

    /* ---------------------------------------------------------------- */
    typedef struct TurboSHAKE_InstanceStruct {
        uint8_t state[KeccakP1600_stateSizeInBytes];
        unsigned int rate;
        uint8_t byteIOIndex;
        uint8_t squeezing;
    } TurboSHAKE_Instance;

    static void TurboSHAKE_Initialize(TurboSHAKE_Instance* instance, int securityLevel)
    {
        KeccakP1600_Initialize(instance->state);
        instance->rate = (1600 - (2 * securityLevel));
        instance->byteIOIndex = 0;
        instance->squeezing = 0;
    }

    static void TurboSHAKE_Absorb(TurboSHAKE_Instance* instance, const unsigned char* data, size_t dataByteLen)
    {
        size_t i, j;
        uint8_t partialBlock;
        const unsigned char* curData;

        const uint8_t rateInBytes = instance->rate / 8;

        //assert(instance->squeezing == 0);

        i = 0;
        curData = data;
        while (i < dataByteLen) {
            if ((instance->byteIOIndex == 0) && (dataByteLen - i >= rateInBytes)) {
#ifdef KeccakP1600_12rounds_FastLoop_supported
                /* processing full blocks first */
                j = KeccakP1600_12rounds_FastLoop_Absorb(instance->state, instance->rate / 64, curData, dataByteLen - i);
                i += j;
                curData += j;
#endif
                for (j = dataByteLen - i; j >= rateInBytes; j -= rateInBytes) {
                    KeccakP1600_AddBytes(instance->state, curData, 0, rateInBytes);
                    KeccakP1600_Permute_12rounds(instance->state);
                    curData += rateInBytes;
                }
                i = dataByteLen - j;
            }
            else {
                /* normal lane: using the message queue */
                if (dataByteLen - i > (size_t)rateInBytes - instance->byteIOIndex) {
                    partialBlock = rateInBytes - instance->byteIOIndex;
                }
                else {
                    partialBlock = (uint8_t)(dataByteLen - i);
                }
                i += partialBlock;

                KeccakP1600_AddBytes(instance->state, curData, instance->byteIOIndex, partialBlock);
                curData += partialBlock;
                instance->byteIOIndex += partialBlock;
                if (instance->byteIOIndex == rateInBytes) {
                    KeccakP1600_Permute_12rounds(instance->state);
                    instance->byteIOIndex = 0;
                }
            }
        }
    }

    static void TurboSHAKE_AbsorbDomainSeparationByte(TurboSHAKE_Instance* instance, unsigned char D)
    {
        const unsigned int rateInBytes = instance->rate / 8;

        //assert(D != 0);
        //assert(instance->squeezing == 0);

        /* Last few bits, whose delimiter coincides with first bit of padding */
        KeccakP1600_AddByte(instance->state, D, instance->byteIOIndex);
        /* If the first bit of padding is at position rate-1, we need a whole new block for the second bit of padding */
        if ((D >= 0x80) && (instance->byteIOIndex == (rateInBytes - 1)))
            KeccakP1600_Permute_12rounds(instance->state);
        /* Second bit of padding */
        KeccakP1600_AddByte(instance->state, 0x80, rateInBytes - 1);
        KeccakP1600_Permute_12rounds(instance->state);
        instance->byteIOIndex = 0;
        instance->squeezing = 1;
    }

    static void TurboSHAKE_Squeeze(TurboSHAKE_Instance* instance, unsigned char* data, size_t dataByteLen)
    {
        size_t i, j;
        unsigned int partialBlock;
        const unsigned int rateInBytes = instance->rate / 8;
        unsigned char* curData;

        if (!instance->squeezing)
            TurboSHAKE_AbsorbDomainSeparationByte(instance, 0x01);

        i = 0;
        curData = data;
        while (i < dataByteLen) {
            if ((instance->byteIOIndex == rateInBytes) && (dataByteLen - i >= rateInBytes)) {
                for (j = dataByteLen - i; j >= rateInBytes; j -= rateInBytes) {
                    KeccakP1600_Permute_12rounds(instance->state);
                    KeccakP1600_ExtractBytes(instance->state, curData, 0, rateInBytes);
                    curData += rateInBytes;
                }
                i = dataByteLen - j;
            }
            else {
                /* normal lane: using the message queue */
                if (instance->byteIOIndex == rateInBytes) {
                    KeccakP1600_Permute_12rounds(instance->state);
                    instance->byteIOIndex = 0;
                }
                if (dataByteLen - i > rateInBytes - instance->byteIOIndex)
                    partialBlock = rateInBytes - instance->byteIOIndex;
                else
                    partialBlock = (unsigned int)(dataByteLen - i);
                i += partialBlock;

                KeccakP1600_ExtractBytes(instance->state, curData, instance->byteIOIndex, partialBlock);
                curData += partialBlock;
                instance->byteIOIndex += partialBlock;
            }
        }
    }

    typedef enum {
        NOT_INITIALIZED,
        ABSORBING,
        FINAL,
        SQUEEZING
    } KCP_Phases;
    typedef KCP_Phases KangarooTwelve_Phases;

    #define K12_chunkSize       8192
    #define K12_suffixLeaf      0x0B /* '110': message hop, simple padding, inner node */
    #define KT128_capacityInBytes   32
    #define KT256_capacityInBytes   64

    #define maxCapacityInBytes 64


    typedef struct KangarooTwelve_InstanceStruct {
        ALIGN(KeccakP1600_stateAlignment) TurboSHAKE_Instance queueNode;
        ALIGN(KeccakP1600_stateAlignment) TurboSHAKE_Instance finalNode;
        size_t fixedOutputLength;
        size_t blockNumber;
        unsigned int queueAbsorbedLen;
        int phase;
        int securityLevel;
    } KangarooTwelve_Instance;

    int KangarooTwelve_Initialize(KangarooTwelve_Instance* ktInstance, int securityLevel, size_t outputByteLen)
    {
        ktInstance->fixedOutputLength = outputByteLen;
        ktInstance->queueAbsorbedLen = 0;
        ktInstance->blockNumber = 0;
        ktInstance->phase = ABSORBING;
        ktInstance->securityLevel = securityLevel;
        TurboSHAKE_Initialize(&ktInstance->finalNode, securityLevel);
        return 0;
    }

    int KangarooTwelve_Update(KangarooTwelve_Instance* ktInstance, const unsigned char* input, size_t inputByteLen)
    {
        if (ktInstance->phase != ABSORBING)
            return 1;

        if (ktInstance->blockNumber == 0) {
            /* First block, absorb in final node */
            unsigned int len = (inputByteLen < (K12_chunkSize - ktInstance->queueAbsorbedLen)) ? (unsigned int)inputByteLen : (K12_chunkSize - ktInstance->queueAbsorbedLen);
            TurboSHAKE_Absorb(&ktInstance->finalNode, input, len);
            input += len;
            inputByteLen -= len;
            ktInstance->queueAbsorbedLen += len;
            if ((ktInstance->queueAbsorbedLen == K12_chunkSize) && (inputByteLen != 0)) {
                /* First block complete and more input data available, finalize it */
                const unsigned char padding = 0x03; /* '110^6': message hop, simple padding */
                ktInstance->queueAbsorbedLen = 0;
                ktInstance->blockNumber = 1;
                TurboSHAKE_Absorb(&ktInstance->finalNode, &padding, 1);
                ktInstance->finalNode.byteIOIndex = (ktInstance->finalNode.byteIOIndex + 7) & ~7; /* Zero padding up to 64 bits */
            }
        }
        else if (ktInstance->queueAbsorbedLen != 0) {
            /* There is data in the queue, absorb further in queue until block complete */
            unsigned int len = (inputByteLen < (K12_chunkSize - ktInstance->queueAbsorbedLen)) ? (unsigned int)inputByteLen : (K12_chunkSize - ktInstance->queueAbsorbedLen);
            TurboSHAKE_Absorb(&ktInstance->queueNode, input, len);
            input += len;
            inputByteLen -= len;
            ktInstance->queueAbsorbedLen += len;
            if (ktInstance->queueAbsorbedLen == K12_chunkSize) {
                int capacityInBytes = 2 * (ktInstance->securityLevel) / 8;
                unsigned char intermediate[maxCapacityInBytes];
                //assert(capacityInBytes <= maxCapacityInBytes);
                ktInstance->queueAbsorbedLen = 0;
                ++ktInstance->blockNumber;
                TurboSHAKE_AbsorbDomainSeparationByte(&ktInstance->queueNode, K12_suffixLeaf);
                TurboSHAKE_Squeeze(&ktInstance->queueNode, intermediate, capacityInBytes);
                TurboSHAKE_Absorb(&ktInstance->finalNode, intermediate, capacityInBytes);
            }
        }

        while (inputByteLen > 0) {
            unsigned int len = (inputByteLen < K12_chunkSize) ? (unsigned int)inputByteLen : K12_chunkSize;
            TurboSHAKE_Initialize(&ktInstance->queueNode, ktInstance->securityLevel);
            TurboSHAKE_Absorb(&ktInstance->queueNode, input, len);
            input += len;
            inputByteLen -= len;
            if (len == K12_chunkSize) {
                int capacityInBytes = 2 * (ktInstance->securityLevel) / 8;
                unsigned char intermediate[maxCapacityInBytes];
                //assert(capacityInBytes <= maxCapacityInBytes);
                ++ktInstance->blockNumber;
                TurboSHAKE_AbsorbDomainSeparationByte(&ktInstance->queueNode, K12_suffixLeaf);
                TurboSHAKE_Squeeze(&ktInstance->queueNode, intermediate, capacityInBytes);
                TurboSHAKE_Absorb(&ktInstance->finalNode, intermediate, capacityInBytes);
            }
            else {
                ktInstance->queueAbsorbedLen = len;
            }
        }

        return 0;
    }

    static unsigned int right_encode(unsigned char* encbuf, size_t value)
    {
        unsigned int n, i;
        size_t v;

        for (v = value, n = 0; v && (n < sizeof(size_t)); ++n, v >>= 8)
            ; /* empty */
        for (i = 1; i <= n; ++i) {
            encbuf[i - 1] = (unsigned char)(value >> (8 * (n - i)));
        }
        encbuf[n] = (unsigned char)n;
        return n + 1;
    }

    int KangarooTwelve_Final(KangarooTwelve_Instance* ktInstance, unsigned char* output, const unsigned char* customization, size_t customByteLen)
    {
        unsigned char encbuf[sizeof(size_t) + 1 + 2];
        unsigned char padding;

        if (ktInstance->phase != ABSORBING)
            return 1;

        /* Absorb customization | right_encode(customByteLen) */
        if ((customByteLen != 0) && (KangarooTwelve_Update(ktInstance, customization, customByteLen) != 0))
            return 1;
        if (KangarooTwelve_Update(ktInstance, encbuf, right_encode(encbuf, customByteLen)) != 0)
            return 1;

        if (ktInstance->blockNumber == 0) {
            /* Non complete first block in final node, pad it */
            padding = 0x07; /*  '11': message hop, final node */
        }
        else {
            unsigned int n;

            if (ktInstance->queueAbsorbedLen != 0) {
                /* There is data in the queue node */
                int capacityInBytes = 2 * (ktInstance->securityLevel) / 8;
                unsigned char intermediate[maxCapacityInBytes];
                //assert(capacityInBytes <= maxCapacityInBytes);
                ++ktInstance->blockNumber;
                TurboSHAKE_AbsorbDomainSeparationByte(&ktInstance->queueNode, K12_suffixLeaf);
                TurboSHAKE_Squeeze(&ktInstance->queueNode, intermediate, capacityInBytes);
                TurboSHAKE_Absorb(&ktInstance->finalNode, intermediate, capacityInBytes);
            }
            --ktInstance->blockNumber; /* Absorb right_encode(number of Chaining Values) || 0xFF || 0xFF */
            n = right_encode(encbuf, ktInstance->blockNumber);
            encbuf[n++] = 0xFF;
            encbuf[n++] = 0xFF;
            TurboSHAKE_Absorb(&ktInstance->finalNode, encbuf, n);
            padding = 0x06; /* '01': chaining hop, final node */
        }
        TurboSHAKE_AbsorbDomainSeparationByte(&ktInstance->finalNode, padding);
        if (ktInstance->fixedOutputLength != 0) {
            ktInstance->phase = FINAL;
            TurboSHAKE_Squeeze(&ktInstance->finalNode, output, ktInstance->fixedOutputLength);
            return 0;
        }
        ktInstance->phase = SQUEEZING;
        return 0;
    }

    int KangarooTwelve_Squeeze(KangarooTwelve_Instance* ktInstance, unsigned char* output, size_t outputByteLen)
    {
        if (ktInstance->phase != SQUEEZING)
            return 1;
        TurboSHAKE_Squeeze(&ktInstance->finalNode, output, outputByteLen);
        return 0;
    }

    /** Extendable ouput function KangarooTwelve.
  * @param  input           Pointer to the input message (M).
  * @param  inputByteLen    The length of the input message in bytes.
  * @param  output          Pointer to the output buffer.
  * @param  outputByteLen   The desired number of output bytes.
  * @param  customization   Pointer to the customization string (C).
  * @param  customByteLen   The length of the customization string in bytes.
  * @param  securityLevel   The desired security strength level (128 bits or 256 bits).
  * @return 0 if successful, 1 otherwise.
  */

    int KangarooTwelveXKPC(int securityLevel, const unsigned char* input, size_t inputByteLen,
        unsigned char* output, size_t outputByteLen,
        const unsigned char* customization, size_t customByteLen)
    {
        KangarooTwelve_Instance ktInstance;

        if (outputByteLen == 0)
            return 1;
        KangarooTwelve_Initialize(&ktInstance, securityLevel, outputByteLen);
        if (KangarooTwelve_Update(&ktInstance, input, inputByteLen) != 0)
            return 1;
        return KangarooTwelve_Final(&ktInstance, output, customization, customByteLen);
    }


    static void KangarooTwelve(const unsigned char* input, unsigned int inputByteLen, unsigned char* output, unsigned int outputByteLen)
    {
        unsigned char a = 'a';
        int sts = KangarooTwelveXKPC(128, input, (size_t)inputByteLen, output, (size_t) outputByteLen, &a, 0);
    }
}
#endif

