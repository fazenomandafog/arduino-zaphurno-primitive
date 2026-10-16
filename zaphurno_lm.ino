#include "zaphurno_lm.h"
#include "zaphurno_lm_weights.h"

static const char VOCAB[28] = " abcdefghijklmnopqrstuvwxyz";

static ZaphurnoLmState st;
static char input_buf[64];
static uint8_t input_len = 0;

static inline uint8_t c2i(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a' + 1;
    if (c >= 'A' && c <= 'Z') return c - 'A' + 1;
    return 0;
}

void setup() {
    Serial.begin(115200);
    while (!Serial) {}
    zaphurno_lm_init(&zaphurno_lm_cfg, &st);
    Serial.println(F("=== Zaphurno LM QAT ==="));
    Serial.print(F("Shifts: fc1="));
    Serial.print(zaphurno_lm_cfg.shift_fc1);
    Serial.print(F(" fc2="));
    Serial.print(zaphurno_lm_cfg.shift_fc2);
    Serial.print(F(" fc3="));
    Serial.println(zaphurno_lm_cfg.shift_fc3);
    Serial.println();
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (input_len > 0) {
                input_buf[input_len] = 0;
                Serial.print(F("Seed: "));
                Serial.println(input_buf);
                Serial.println();

                memset(st.ctx, 0, LM_CTX);
                st.ctx_pos = 0;
                for (uint8_t i = 0; i < input_len; i++) {
                    uint8_t id = c2i(input_buf[i]);
                    st.ctx[st.ctx_pos] = id;
                    st.ctx_pos = (st.ctx_pos + 1) % LM_CTX;
                }

                Serial.print(F("Gen:  "));
                for (uint8_t i = 0; i < 60; i++) {
                    uint8_t cc = zaphurno_lm_next(&zaphurno_lm_cfg, &st);
                    Serial.print(VOCAB[cc]);
                }
                Serial.println();
                Serial.println();
                input_len = 0;
            }
        } else if (input_len < sizeof(input_buf) - 1) {
            input_buf[input_len++] = c;
        }
    }
}