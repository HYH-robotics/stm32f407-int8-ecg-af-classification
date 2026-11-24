; P6 experiment startup: explicit 8192-byte main stack.
; A5 watermark is armed in main after the C runtime has initialized RAM.
; Cortex-M4 vector table, interrupts default to a halt loop.
Stack_Size      EQU     0x00002000
Heap_Size       EQU     0x00000100

                AREA    STACK, NOINIT, READWRITE, ALIGN=3
                EXPORT  __stack_low
__stack_low
Stack_Mem       SPACE   Stack_Size
                EXPORT  __stack_high
                EXPORT  __initial_sp
__stack_high
__initial_sp

                AREA    HEAP, NOINIT, READWRITE, ALIGN=3
                EXPORT  __heap_base
__heap_base
Heap_Mem        SPACE   Heap_Size
                EXPORT  __heap_limit
__heap_limit

                PRESERVE8
                THUMB
                AREA    RESET, DATA, READONLY
                EXPORT  __Vectors
__Vectors       DCD     __initial_sp
                DCD     Reset_Handler
                DCD     Default_Handler ; NMI
                DCD     Default_Handler ; HardFault
                DCD     Default_Handler ; MemManage
                DCD     Default_Handler ; BusFault
                DCD     Default_Handler ; UsageFault
                DCD     0,0,0,0
                DCD     Default_Handler ; SVC
                DCD     Default_Handler ; DebugMon
                DCD     0
                DCD     Default_Handler ; PendSV
                DCD     Default_Handler ; SysTick
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler
                DCD     Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler,Default_Handler

                AREA    |.text|, CODE, READONLY
Reset_Handler   PROC
                EXPORT  Reset_Handler
                IMPORT  SystemInit
                IMPORT  __main
                LDR     R0, =SystemInit
                BLX     R0
                LDR     R0, =__main
                BX      R0
                ENDP

Default_Handler PROC
                EXPORT  Default_Handler
halt_loop       B       halt_loop
                ENDP

                EXPORT  __user_initial_stackheap
__user_initial_stackheap
                LDR     R0, =Heap_Mem
                LDR     R1, =__stack_high
                LDR     R2, =__heap_limit
                LDR     R3, =__stack_low
                BX      LR
                END
