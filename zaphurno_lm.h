#ifndef ZAPHURNO_LM_H
#define ZAPHURNO_LM_H

#include <stdint.h>
#include <avr/pgmspace.h>

#define ZP_BLOCK_DIM   16
#define ZP_N_MADRE     16
#define ZP_N_HIJA      4
#define ZP_N_NIETA     4
#define ZP_N_RES       12

#define LM_VOCAB       27
#define LM_CTX         8
#define LM_EMB         16
#define LM_H1          64
#define LM_H2          32
#define LM_IN_DIM      (LM_CTX * LM_EMB)

typedef struct {
    uint8_t vocab, ctx_len, emb_dim, h1_dim, h2_dim;
    uint16_t n_blocks;

    uint16_t off_fc1_w;
    uint8_t  fc1_bpr;
    uint8_t  shift_fc1, shift_fc2, shift_fc3;

    const int8_t*  cb_base;
    const int8_t*  cb_delta;
    const int8_t*  cb_micro;
    const uint8_t* block_paths;
    const uint8_t* res_pos;
    const int8_t*  res_val;

    const int8_t*  emb_raw;
    const int8_t*  fc2_raw;
    const int8_t*  fc3_raw;
} ZaphurnoLmConfig;

typedef struct {
    int8_t  emb_table[LM_VOCAB * LM_EMB];
    int8_t  in_buf[LM_IN_DIM];
    int8_t  h1_buf[LM_H1];
    int8_t  h2_buf[LM_H2];
    int32_t logits[LM_VOCAB];
    uint8_t ctx[LM_CTX];
    uint8_t ctx_pos;
} ZaphurnoLmState;

void    zaphurno_lm_init(const ZaphurnoLmConfig*, ZaphurnoLmState*);
uint8_t zaphurno_lm_next(const ZaphurnoLmConfig*, ZaphurnoLmState*);

#endif