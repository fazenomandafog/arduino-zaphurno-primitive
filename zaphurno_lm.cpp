#include "zaphurno_lm.h"
#include <string.h>

static inline int8_t clamp_i8(int16_t v) {
    return v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v);
}

// Reconstruye un bloque con el MISMO orden de operaciones que el encoder
static void recon_block(const ZaphurnoLmConfig* cfg, uint16_t bid, int8_t* out) {
    uint8_t p = pgm_read_byte(cfg->block_paths + bid);
    uint8_t m = p & 0x0F, h = (p >> 4) & 0x03, n = (p >> 6) & 0x03;
    const int8_t* base  = cfg->cb_base  + (uint16_t)m * ZP_BLOCK_DIM;
    const int8_t* delta = cfg->cb_delta + ((uint16_t)m*ZP_N_HIJA + h) * ZP_BLOCK_DIM;
    const int8_t* micro = cfg->cb_micro +
                          (((uint16_t)m*ZP_N_HIJA + h)*ZP_N_NIETA + n) * ZP_BLOCK_DIM;
    for (uint8_t k = 0; k < ZP_BLOCK_DIM; k++) {
        int16_t v = (int16_t)(int8_t)pgm_read_byte(base + k)
                  + (int16_t)(int8_t)pgm_read_byte(delta + k)
                  + (int16_t)(int8_t)pgm_read_byte(micro + k);
        out[k] = clamp_i8(v);
    }
    uint16_t base_idx = bid * ZP_N_RES;
    for (uint8_t r = 0; r < ZP_N_RES; r++) {
        uint8_t pos = pgm_read_byte(cfg->res_pos + base_idx + r);
        int8_t  val = (int8_t)pgm_read_byte(cfg->res_val + base_idx + r);
        out[pos] = clamp_i8((int16_t)out[pos] + val);
    }
}

static void recon_row(const ZaphurnoLmConfig* cfg, uint16_t start,
                      uint8_t n, int8_t* out) {
    for (uint8_t b = 0; b < n; b++)
        recon_block(cfg, start + b, out + b * ZP_BLOCK_DIM);
}

void zaphurno_lm_init(const ZaphurnoLmConfig* cfg, ZaphurnoLmState* st) {
    for (uint8_t c = 0; c < LM_VOCAB; c++) {
        for (uint8_t k = 0; k < LM_EMB; k++) {
            st->emb_table[c * LM_EMB + k] =
                (int8_t)pgm_read_byte(cfg->emb_raw + c * LM_EMB + k);
        }
    }
    memset(st->ctx, 0, LM_CTX);
    st->ctx_pos = 0;
}

static void push_ctx(ZaphurnoLmState* st, uint8_t c) {
    st->ctx[st->ctx_pos] = c;
    st->ctx_pos = (st->ctx_pos + 1) % LM_CTX;
}

uint8_t zaphurno_lm_next(const ZaphurnoLmConfig* cfg, ZaphurnoLmState* st) {
    // Input: embeddings del contexto
    for (uint8_t i = 0; i < LM_CTX; i++) {
        uint8_t c = st->ctx[(st->ctx_pos + i) % LM_CTX];
        memcpy(st->in_buf + i * LM_EMB, st->emb_table + c * LM_EMB, LM_EMB);
    }

    // fc1: comprimido con Zaphurno
    int8_t row[128];
    for (uint8_t j = 0; j < LM_H1; j++) {
        recon_row(cfg, cfg->off_fc1_w + j * cfg->fc1_bpr, cfg->fc1_bpr, row);
        int32_t acc = 0;
        for (uint16_t k = 0; k < LM_IN_DIM; k++)
            acc += (int32_t)row[k] * (int32_t)st->in_buf[k];
        int16_t v = (int16_t)(acc >> cfg->shift_fc1);
        st->h1_buf[j] = (v > 0) ? clamp_i8(v) : 0;
    }

    // fc2: SIN comprimir (int8 exacto)
    for (uint8_t j = 0; j < LM_H2; j++) {
        int32_t acc = 0;
        for (uint8_t k = 0; k < LM_H1; k++) {
            int8_t w = (int8_t)pgm_read_byte(cfg->fc2_raw + j * LM_H1 + k);
            acc += (int32_t)w * (int32_t)st->h1_buf[k];
        }
        int16_t v = (int16_t)(acc >> cfg->shift_fc2);
        st->h2_buf[j] = (v > 0) ? clamp_i8(v) : 0;
    }

    // fc3: SIN comprimir
    for (uint8_t j = 0; j < LM_VOCAB; j++) {
        int32_t acc = 0;
        for (uint8_t k = 0; k < LM_H2; k++) {
            int8_t w = (int8_t)pgm_read_byte(cfg->fc3_raw + j * LM_H2 + k);
            acc += (int32_t)w * (int32_t)st->h2_buf[k];
        }
        st->logits[j] = acc >> cfg->shift_fc3;
    }

    uint8_t best = 0;
    int32_t bv = st->logits[0];
    for (uint8_t j = 1; j < LM_VOCAB; j++)
        if (st->logits[j] > bv) { bv = st->logits[j]; best = j; }

    push_ctx(st, best);
    return best;
}