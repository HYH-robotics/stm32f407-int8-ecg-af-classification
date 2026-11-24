/* P6 public-ECG replay firmware for STM32F407ZGT6. No clinical input path. */
#include "stm32f4xx.h"
#include "p6_model_data.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define RAW_N 1500
#define UART_BAUD 460800UL
#define STACK_RESERVED 8192UL
#define STACK_GUARD 32UL

extern uint8_t __stack_low;
extern uint8_t __stack_high;
uint32_t SystemCoreClock = 16000000UL;

static int16_t raw_i16[RAW_N];
static float sort_buf[RAW_N];
static int8_t input_q[RAW_N];
static int8_t act_a[16 * 748];
static int8_t act_b[32 * 372];
static int8_t pool_q[64];
static uint8_t quality_gate = 0;
static uint8_t clock_ok = 0;
static USART_TypeDef *active_uart = 0;

void SystemInit(void) {
    SCB->CPACR |= (0xFUL << 20);
    __DSB();
    __ISB();
}

static uint8_t clock_168mhz(void) {
    uint32_t limit = 2000000UL;
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR |= PWR_CR_VOS;
    RCC->CR |= RCC_CR_HSEON;
    while (!(RCC->CR & RCC_CR_HSERDY) && --limit) { }
    if (!limit) return 0;
    FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_PRFTEN | FLASH_ACR_LATENCY_5WS;
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
        | RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2;
    RCC->PLLCFGR = 8UL | (336UL << 6) | RCC_PLLCFGR_PLLSRC_HSE | (7UL << 24);
    RCC->CR |= RCC_CR_PLLON;
    limit = 2000000UL;
    while (!(RCC->CR & RCC_CR_PLLRDY) && --limit) { }
    if (!limit) return 0;
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    limit = 2000000UL;
    while (((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) && --limit) { }
    if (!limit) return 0;
    SystemCoreClock = 168000000UL;
    return 1;
}

static void uart_init(void) {
    uint32_t pclk1 = clock_ok ? 42000000UL : 16000000UL;
    uint32_t pclk2 = clock_ok ? 84000000UL : 16000000UL;
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    /* PA2/PA3 = USART2, PA9/PA10 = USART1, alternate function 7. */
    GPIOA->MODER = (GPIOA->MODER & ~((3UL<<4)|(3UL<<6)|(3UL<<18)|(3UL<<20)))
                  | (2UL<<4)|(2UL<<6)|(2UL<<18)|(2UL<<20);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~((15UL<<8)|(15UL<<12))) | (7UL<<8)|(7UL<<12);
    GPIOA->AFR[1] = (GPIOA->AFR[1] & ~((15UL<<4)|(15UL<<8))) | (7UL<<4)|(7UL<<8);
    GPIOA->OSPEEDR |= (3UL<<4)|(3UL<<6)|(3UL<<18)|(3UL<<20);
    GPIOA->PUPDR = (GPIOA->PUPDR & ~((3UL<<6)|(3UL<<20))) | (1UL<<6)|(1UL<<20);
    USART1->BRR = (pclk2 + UART_BAUD / 2) / UART_BAUD;
    USART2->BRR = (pclk1 + UART_BAUD / 2) / UART_BAUD;
    USART1->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

static void tx(USART_TypeDef *u, uint8_t ch) {
    while (!(u->SR & USART_SR_TXE)) { }
    u->DR = ch;
}

static uint8_t rx(USART_TypeDef *u) {
    while (!(u->SR & USART_SR_RXNE)) { }
    return (uint8_t)u->DR;
}

static void send_text(USART_TypeDef *u, const char *s) {
    while (*s) tx(u, (uint8_t)*s++);
}

static void send_u32(USART_TypeDef *u, uint32_t x) {
    tx(u, (uint8_t)x); tx(u, (uint8_t)(x>>8)); tx(u, (uint8_t)(x>>16)); tx(u, (uint8_t)(x>>24));
}

static void send_float(USART_TypeDef *u, float value) {
    union { float f; uint32_t u; } v;
    v.f = value;
    send_u32(u, v.u);
}

static void receive_raw(USART_TypeDef *u) {
    uint32_t i;
    for (i = 0; i < RAW_N; ++i) {
        uint16_t lo = rx(u);
        uint16_t hi = rx(u);
        raw_i16[i] = (int16_t)(lo | (hi << 8));
    }
}

static int float_cmp(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static double quantile(float *s, double p) {
    double h = (RAW_N - 1) * p;
    uint32_t j = (uint32_t)h;
    double g = h - j;
    if (j >= RAW_N - 1) return s[RAW_N - 1];
    return (1.0 - g) * (double)s[j] + g * (double)s[j+1];
}

static int32_t round_away(double x) {
    return (int32_t)((x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5));
}

static int8_t quant_input(double x) {
    int32_t q = round_away(x);
    if (q < -127) q = -127;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static int8_t quant_relu(double x) {
    int32_t q = round_away(x);
    if (q < 0) q = 0;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static void conv(const int8_t *input, int32_t in_ch, int32_t in_len,
                 int8_t *output, int32_t out_ch, const int8_t *weight,
                 const int32_t *bias, const double *ratio) {
    int32_t out_len = (in_len - 5) / 2 + 1;
    int32_t o, t, c, k;
    for (o = 0; o < out_ch; ++o) {
        for (t = 0; t < out_len; ++t) {
            int32_t acc = bias[o];
            for (c = 0; c < in_ch; ++c) {
                const int8_t *a = input + c * in_len + t * 2;
                const int8_t *w = weight + (o * in_ch + c) * 5;
                for (k = 0; k < 5; ++k) acc += (int32_t)a[k] * (int32_t)w[k];
            }
            output[o * out_len + t] = quant_relu((double)acc * ratio[o]);
        }
    }
}

static float infer(uint8_t *quality) {
    uint32_t i, plateau = 0;
    double q05, median, q95, spread, denom, max_dev = 0.0;
    int32_t acc;
    for (i = 0; i < RAW_N; ++i) {
        sort_buf[i] = (float)raw_i16[i];
        if (i && raw_i16[i] == raw_i16[i-1]) ++plateau;
    }
    qsort(sort_buf, RAW_N, sizeof(float), float_cmp);
    q05 = quantile(sort_buf, 0.05);
    median = quantile(sort_buf, 0.50);
    q95 = quantile(sort_buf, 0.95);
    spread = q95 - q05;
    denom = spread + 1e-6;
    for (i = 0; i < RAW_N; ++i) {
        double d = fabs((double)raw_i16[i] - median);
        if (d > max_dev) max_dev = d;
    }
    *quality = (uint8_t)(quality_gate && ((double)plateau / (RAW_N - 1) >= 0.95
                         || spread <= 1e-6 || max_dev / denom > 25.0));
    if (*quality) return -1.0f;
    for (i = 0; i < RAW_N; ++i) {
        double norm = ((double)raw_i16[i] - median) / denom;
        float x;
        if (norm < -8.0) norm = -8.0;
        if (norm > 8.0) norm = 8.0;
        x = (float)norm;  /* Match NumPy's float32 normalization output. */
        input_q[i] = quant_input((double)x / P6_INPUT_SCALE);
    }
    conv(input_q, 1, 1500, act_a, 16, p6_w0, p6_b0, p6_ratio0);
    conv(act_a, 16, 748, act_b, 32, p6_w1, p6_b1, p6_ratio1);
    conv(act_b, 32, 372, act_a, 64, p6_w2, p6_b2, p6_ratio2);
    for (i = 0; i < 64; ++i) {
        uint32_t t;
        int32_t sum = 0;
        for (t = 0; t < 184; ++t) sum += act_a[i * 184 + t];
        pool_q[i] = quant_input(((double)sum / 184.0) * P6_POOL_RATIO);
    }
    acc = p6_b3[0];
    for (i = 0; i < 64; ++i) acc += (int32_t)pool_q[i] * (int32_t)p6_w3[i];
    return (float)(1.0 / (1.0 + exp(-(double)acc * P6_LOGIT_SCALE)));
}

static uint32_t timed_infer(float *score, uint8_t *quality) {
    uint32_t mask = __get_PRIMASK(), start, cycles;
    __disable_irq();
    start = DWT->CYCCNT;
    *score = infer(quality);
    cycles = DWT->CYCCNT - start;
    __set_PRIMASK(mask);
    return cycles;
}

static uint32_t stack_peak(void) {
    uint8_t *p = &__stack_low;
    while (p < &__stack_high && *p == 0xA5) ++p;
    return (uint32_t)(&__stack_high - p);
}

static void arm_stack_watermark(void) {
    uint8_t *p = &__stack_low;
    uint8_t *safe = (uint8_t *)(__get_MSP() - 256UL);
    if (safe > &__stack_high) safe = &__stack_high;
    /* Leave 256 B below current SP untouched; all other reserved stack bytes
       are marked after C runtime zero-initialization and before warm-up. */
    while (p < safe) *p++ = 0xA5;
}

static uint8_t stack_guard_ok(void) {
    uint32_t i;
    for (i = 0; i < STACK_GUARD; ++i) if ((&__stack_low)[i] != 0xA5) return 0;
    return 1;
}

int main(void) {
    uint8_t cmd, q, i;
    float score;
    uint32_t cycles;
    clock_ok = clock_168mhz();
    uart_init();
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    send_text(USART1, "P6HELLO\n");
    send_text(USART2, "P6HELLO\n");
    for (;;) {
        if (USART1->SR & USART_SR_RXNE) active_uart = USART1;
        else if (USART2->SR & USART_SR_RXNE) active_uart = USART2;
        else continue;
        cmd = rx(active_uart);
        if (cmd == 'P') {
            send_text(active_uart, clock_ok ? "P6OK 168000000\n" : "P6ERR HSE\n");
        } else if (cmd == 'C') {
            quality_gate = rx(active_uart) ? 1 : 0;
            tx(active_uart, 'C'); tx(active_uart, quality_gate);
        } else if (cmd == 'B') {
            arm_stack_watermark();
            tx(active_uart, 'B');
        } else if (cmd == 'W' && clock_ok) {
            receive_raw(active_uart);
            tx(active_uart, 'W');
            for (i = 0; i < 100; ++i) {
                cycles = timed_infer(&score, &q);
                send_u32(active_uart, cycles);
            }
        } else if (cmd == 'M' && clock_ok) {
            receive_raw(active_uart);
            cycles = timed_infer(&score, &q);
            tx(active_uart, 'M'); send_u32(active_uart, cycles);
            send_float(active_uart, score); tx(active_uart, q);
        } else if (cmd == 'S') {
            tx(active_uart, 'S'); send_u32(active_uart, stack_peak());
            tx(active_uart, stack_guard_ok()); send_u32(active_uart, STACK_RESERVED);
        } else {
            tx(active_uart, '?');
        }
    }
}
