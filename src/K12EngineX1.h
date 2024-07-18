#pragma once


struct synapseCheckpoint {
    unsigned long long ckp[25];
    int ignoreByteInState;
};

#ifdef __AVX512F__

#include "kangaroo_twelve_XKCP_avx512.h"

struct K12EngineX1_AVX512 {
    unsigned long long scatteredStates[25];
    int leftByteInCurrentState;
    V512  Baeiou, Gaeiou, Kaeiou, Maeiou, Saeiou;
private:
    void _scatterFromVector() {
        copyToState_xkpc_AVX512(scatteredStates);
    }
    void hashNewChunk() {
        KeccakP_DeclareVars2
        rounds12_xkpc_AVX512;
    }
    void hashNewChunkAndSaveToState() {
        hashNewChunk();
        _scatterFromVector();
        leftByteInCurrentState = 200;
    }
public:
    K12EngineX1_AVX512() {}
    void initState(const unsigned long long* comp_u64, const unsigned long long* nonce_u64) {
        setMem(scatteredStates, 25 * sizeof(unsigned long long), 0);

        scatteredStates[0] = comp_u64[0];
        scatteredStates[1] = comp_u64[1];
        scatteredStates[2] = comp_u64[2];
        scatteredStates[3] = comp_u64[3];
        scatteredStates[4] = nonce_u64[0];
        scatteredStates[5] = nonce_u64[1];
        scatteredStates[6] = nonce_u64[2];
        scatteredStates[7] = nonce_u64[3];
        leftByteInCurrentState = 0;

        copyFromState_xkpc_AVX512(scatteredStates);
    }
    void write(unsigned char* out0, int size) {
        unsigned char* s0 = (unsigned char*)scatteredStates;
        if (leftByteInCurrentState) {
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            copyMem(out0, s0 + 200 - leftByteInCurrentState, copySize);
            size -= copySize;
            leftByteInCurrentState -= copySize;
            out0 += copySize;
        }
        while (size) {
            if (!leftByteInCurrentState) {
                hashNewChunkAndSaveToState();
            }
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            copyMem(out0, s0 + 200 - leftByteInCurrentState, copySize);
            size -= copySize;
            leftByteInCurrentState -= copySize;
            out0 += copySize;
        }
    }

    void saveCheckpoint(synapseCheckpoint** p_sckp) {
        synapseCheckpoint& sckp_0 = *(p_sckp[0]);
        unsigned long long* output0 = sckp_0.ckp;
        copyToState_xkpc_AVX512(output0);
            sckp_0.ignoreByteInState = 200 - leftByteInCurrentState;
    }

    void scatterFromVector() {
        _scatterFromVector();
    }
    void hashWithoutWrite(int size) {
        if (leftByteInCurrentState) {
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            size -= copySize;
            leftByteInCurrentState -= copySize;
        }
        while (size) {
            if (!leftByteInCurrentState) {
                hashNewChunk();
                leftByteInCurrentState = 200;
            }
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            size -= copySize;
            leftByteInCurrentState -= copySize;
        }
    }
};

#endif


struct K12EngineX1 {
    unsigned long long Aba, Abe, Abi, Abo, Abu;
    unsigned long long Aga, Age, Agi, Ago, Agu;
    unsigned long long Aka, Ake, Aki, Ako, Aku;
    unsigned long long Ama, Ame, Ami, Amo, Amu;
    unsigned long long Asa, Ase, Asi, Aso, Asu;
    unsigned long long scatteredStates[25];
    int leftByteInCurrentState;
private:
    void _scatterFromVector() {
        copyToStateScalar(scatteredStates)
    }
    void hashNewChunk() {
        declareBCDEScalar
            rounds12Scalar
    }
    void hashNewChunkAndSaveToState() {
        hashNewChunk();
        _scatterFromVector();
        leftByteInCurrentState = 200;
    }
public:
    K12EngineX1() {}
    void initState(const unsigned long long* comp_u64, const unsigned long long* nonce_u64) {
        Aba = comp_u64[0];
        Abe = comp_u64[1];
        Abi = comp_u64[2];
        Abo = comp_u64[3];
        Abu = nonce_u64[0];
        Aga = nonce_u64[1];
        Age = nonce_u64[2];
        Agi = nonce_u64[3];
        Ago = Agu = Aka = Ake = Aki = Ako = Aku = Ama = Ame = Ami = Amo = Amu = Asa = Ase = Asi = Aso = Asu = 0;
        leftByteInCurrentState = 0;
    }
    void write(unsigned char* out0, int size) {
        unsigned char* s0 = (unsigned char*)scatteredStates;
        if (leftByteInCurrentState) {
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            copyMem(out0, s0 + 200 - leftByteInCurrentState, copySize);
            size -= copySize;
            leftByteInCurrentState -= copySize;
            out0 += copySize;
        }
        while (size) {
            if (!leftByteInCurrentState) {
                hashNewChunkAndSaveToState();
            }
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            copyMem(out0, s0 + 200 - leftByteInCurrentState, copySize);
            size -= copySize;
            leftByteInCurrentState -= copySize;
            out0 += copySize;
        }
    }

    void saveCheckpoint(synapseCheckpoint** p_sckp) {
        synapseCheckpoint& sckp_0 = *(p_sckp[0]);
        unsigned long long* output0 = sckp_0.ckp;
        copyToStateScalar(output0)
            sckp_0.ignoreByteInState = 200 - leftByteInCurrentState;
    }

    void scatterFromVector() {
        _scatterFromVector();
    }
    void hashWithoutWrite(int size) {
        if (leftByteInCurrentState) {
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            size -= copySize;
            leftByteInCurrentState -= copySize;
        }
        while (size) {
            if (!leftByteInCurrentState) {
                hashNewChunk();
                leftByteInCurrentState = 200;
            }
            int copySize = size < leftByteInCurrentState ? size : leftByteInCurrentState;
            size -= copySize;
            leftByteInCurrentState -= copySize;
        }
    }
};

