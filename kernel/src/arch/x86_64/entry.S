# Low-level interrupt/exception entry stubs.
#
# Generates one `interrupt_entry_<vector>` label per IDT vector (0-255),
# referenced by address from `INTERRUPT_ENTRIES` below and installed into
# the IDT by `idt.rs`. Each stub normalises the CPU-pushed frame (padding in
# a fake error code for vectors that don't push one), saves the
# caller-saved registers, calls into `__interrupt_handler` (see
# `cpu.rs`) with the vector number and a pointer to the resulting
# `InterruptStackFrame`, then reverses all of it and `iretq`s back.

# Usage: IRQ <vector> <error-code?>
.macro IRQ vector, errorcode
.align 16
.global interrupt_entry_\vector
interrupt_entry_\vector:
        # Push fake error code if exception doesn't provide one
	.if \errorcode == 0
		pushq $0
	.endif

        # Swap user and kernel %gs (if necessary)
	testb $0x3, 0x10(%rsp)
	je 1f
	swapgs
1:
        # Save the interrupted code's base pointer, part of the frame
        pushq %rbp

        # Save caller-saved registers
	pushq %r11
	pushq %r10
	pushq %r9
	pushq %r8
	pushq %rdi
	pushq %rsi
	pushq %rdx
	pushq %rcx
	pushq %rax

	cld

        # End the frame-pointer chain here: a backtrace taken inside the
        # handler stops at this entry rather than walking into the frames of
        # the code that was interrupted, whose own position is the saved
        # %rip. The saved %rbp lets an exception handler walk those frames
        # separately.
        mov CALL_STACK_END_MARKER(%rip), %rbp

	# Call the high-level interrupt handler routine
	movq $\vector, %rdi
	movq %rsp, %rsi
	call __interrupt_handler

	# Restore caller-saved registers
	popq %rax
	popq %rcx
	popq %rdx
	popq %rsi
	popq %rdi
	popq %r8
	popq %r9
	popq %r10
	popq %r11

        # Restore base pointer
        popq %rbp

	# Drop error code
	addq $8, %rsp

        # Swap kernel and user %gs (if necessary)
	testb $0x3, 0x8(%rsp)
	je 2f
	swapgs
2:
	iretq
.endm

.section .text

# Generate interrupt entries
IRQ  0, 0
IRQ  1, 0
IRQ  2, 0
IRQ  3, 0
IRQ  4, 0
IRQ  5, 0
IRQ  6, 0
IRQ  7, 0
IRQ  8, 1
IRQ  9, 0
IRQ 10, 1
IRQ 11, 1
IRQ 12, 1
IRQ 13, 1
IRQ 14, 1
IRQ 15, 0
IRQ 16, 0
IRQ 17, 1
IRQ 18, 0
IRQ 19, 0
IRQ 20, 0
IRQ 21, 0
IRQ 22, 0
IRQ 23, 0
IRQ 24, 0
IRQ 25, 0
IRQ 26, 0
IRQ 27, 0
IRQ 28, 0
IRQ 29, 0
IRQ 30, 0
IRQ 31, 0
IRQ 32, 0
IRQ 33, 0
IRQ 34, 0
IRQ 35, 0
IRQ 36, 0
IRQ 37, 0
IRQ 38, 0
IRQ 39, 0
IRQ 40, 0
IRQ 41, 0
IRQ 42, 0
IRQ 43, 0
IRQ 44, 0
IRQ 45, 0
IRQ 46, 0
IRQ 47, 0
IRQ 48, 0
IRQ 49, 0
IRQ 50, 0
IRQ 51, 0
IRQ 52, 0
IRQ 53, 0
IRQ 54, 0
IRQ 55, 0
IRQ 56, 0
IRQ 57, 0
IRQ 58, 0
IRQ 59, 0
IRQ 60, 0
IRQ 61, 0
IRQ 62, 0
IRQ 63, 0
IRQ 64, 0
IRQ 65, 0
IRQ 66, 0
IRQ 67, 0
IRQ 68, 0
IRQ 69, 0
IRQ 70, 0
IRQ 71, 0
IRQ 72, 0
IRQ 73, 0
IRQ 74, 0
IRQ 75, 0
IRQ 76, 0
IRQ 77, 0
IRQ 78, 0
IRQ 79, 0
IRQ 80, 0
IRQ 81, 0
IRQ 82, 0
IRQ 83, 0
IRQ 84, 0
IRQ 85, 0
IRQ 86, 0
IRQ 87, 0
IRQ 88, 0
IRQ 89, 0
IRQ 90, 0
IRQ 91, 0
IRQ 92, 0
IRQ 93, 0
IRQ 94, 0
IRQ 95, 0
IRQ 96, 0
IRQ 97, 0
IRQ 98, 0
IRQ 99, 0
IRQ 100, 0
IRQ 101, 0
IRQ 102, 0
IRQ 103, 0
IRQ 104, 0
IRQ 105, 0
IRQ 106, 0
IRQ 107, 0
IRQ 108, 0
IRQ 109, 0
IRQ 110, 0
IRQ 111, 0
IRQ 112, 0
IRQ 113, 0
IRQ 114, 0
IRQ 115, 0
IRQ 116, 0
IRQ 117, 0
IRQ 118, 0
IRQ 119, 0
IRQ 120, 0
IRQ 121, 0
IRQ 122, 0
IRQ 123, 0
IRQ 124, 0
IRQ 125, 0
IRQ 126, 0
IRQ 127, 0
IRQ 128, 0
IRQ 129, 0
IRQ 130, 0
IRQ 131, 0
IRQ 132, 0
IRQ 133, 0
IRQ 134, 0
IRQ 135, 0
IRQ 136, 0
IRQ 137, 0
IRQ 138, 0
IRQ 139, 0
IRQ 140, 0
IRQ 141, 0
IRQ 142, 0
IRQ 143, 0
IRQ 144, 0
IRQ 145, 0
IRQ 146, 0
IRQ 147, 0
IRQ 148, 0
IRQ 149, 0
IRQ 150, 0
IRQ 151, 0
IRQ 152, 0
IRQ 153, 0
IRQ 154, 0
IRQ 155, 0
IRQ 156, 0
IRQ 157, 0
IRQ 158, 0
IRQ 159, 0
IRQ 160, 0
IRQ 161, 0
IRQ 162, 0
IRQ 163, 0
IRQ 164, 0
IRQ 165, 0
IRQ 166, 0
IRQ 167, 0
IRQ 168, 0
IRQ 169, 0
IRQ 170, 0
IRQ 171, 0
IRQ 172, 0
IRQ 173, 0
IRQ 174, 0
IRQ 175, 0
IRQ 176, 0
IRQ 177, 0
IRQ 178, 0
IRQ 179, 0
IRQ 180, 0
IRQ 181, 0
IRQ 182, 0
IRQ 183, 0
IRQ 184, 0
IRQ 185, 0
IRQ 186, 0
IRQ 187, 0
IRQ 188, 0
IRQ 189, 0
IRQ 190, 0
IRQ 191, 0
IRQ 192, 0
IRQ 193, 0
IRQ 194, 0
IRQ 195, 0
IRQ 196, 0
IRQ 197, 0
IRQ 198, 0
IRQ 199, 0
IRQ 200, 0
IRQ 201, 0
IRQ 202, 0
IRQ 203, 0
IRQ 204, 0
IRQ 205, 0
IRQ 206, 0
IRQ 207, 0
IRQ 208, 0
IRQ 209, 0
IRQ 210, 0
IRQ 211, 0
IRQ 212, 0
IRQ 213, 0
IRQ 214, 0
IRQ 215, 0
IRQ 216, 0
IRQ 217, 0
IRQ 218, 0
IRQ 219, 0
IRQ 220, 0
IRQ 221, 0
IRQ 222, 0
IRQ 223, 0
IRQ 224, 0
IRQ 225, 0
IRQ 226, 0
IRQ 227, 0
IRQ 228, 0
IRQ 229, 0
IRQ 230, 0
IRQ 231, 0
IRQ 232, 0
IRQ 233, 0
IRQ 234, 0
IRQ 235, 0
IRQ 236, 0
IRQ 237, 0
IRQ 238, 0
IRQ 239, 0
IRQ 240, 0
IRQ 241, 0
IRQ 242, 0
IRQ 243, 0
IRQ 244, 0
IRQ 245, 0
IRQ 246, 0
IRQ 247, 0
IRQ 248, 0
IRQ 249, 0
IRQ 250, 0
IRQ 251, 0
IRQ 252, 0
IRQ 253, 0
IRQ 254, 0
IRQ 255, 0

.section .data

# Generate an array of all interrupt entries
.global INTERRUPT_ENTRIES
INTERRUPT_ENTRIES:
        .quad interrupt_entry_0
        .quad interrupt_entry_1
        .quad interrupt_entry_2
        .quad interrupt_entry_3
        .quad interrupt_entry_4
        .quad interrupt_entry_5
        .quad interrupt_entry_6
        .quad interrupt_entry_7
        .quad interrupt_entry_8
        .quad interrupt_entry_9
        .quad interrupt_entry_10
        .quad interrupt_entry_11
        .quad interrupt_entry_12
        .quad interrupt_entry_13
        .quad interrupt_entry_14
        .quad interrupt_entry_15
        .quad interrupt_entry_16
        .quad interrupt_entry_17
        .quad interrupt_entry_18
        .quad interrupt_entry_19
        .quad interrupt_entry_20
        .quad interrupt_entry_21
        .quad interrupt_entry_22
        .quad interrupt_entry_23
        .quad interrupt_entry_24
        .quad interrupt_entry_25
        .quad interrupt_entry_26
        .quad interrupt_entry_27
        .quad interrupt_entry_28
        .quad interrupt_entry_29
        .quad interrupt_entry_30
        .quad interrupt_entry_31
        .quad interrupt_entry_32
        .quad interrupt_entry_33
        .quad interrupt_entry_34
        .quad interrupt_entry_35
        .quad interrupt_entry_36
        .quad interrupt_entry_37
        .quad interrupt_entry_38
        .quad interrupt_entry_39
        .quad interrupt_entry_40
        .quad interrupt_entry_41
        .quad interrupt_entry_42
        .quad interrupt_entry_43
        .quad interrupt_entry_44
        .quad interrupt_entry_45
        .quad interrupt_entry_46
        .quad interrupt_entry_47
        .quad interrupt_entry_48
        .quad interrupt_entry_49
        .quad interrupt_entry_50
        .quad interrupt_entry_51
        .quad interrupt_entry_52
        .quad interrupt_entry_53
        .quad interrupt_entry_54
        .quad interrupt_entry_55
        .quad interrupt_entry_56
        .quad interrupt_entry_57
        .quad interrupt_entry_58
        .quad interrupt_entry_59
        .quad interrupt_entry_60
        .quad interrupt_entry_61
        .quad interrupt_entry_62
        .quad interrupt_entry_63
        .quad interrupt_entry_64
        .quad interrupt_entry_65
        .quad interrupt_entry_66
        .quad interrupt_entry_67
        .quad interrupt_entry_68
        .quad interrupt_entry_69
        .quad interrupt_entry_70
        .quad interrupt_entry_71
        .quad interrupt_entry_72
        .quad interrupt_entry_73
        .quad interrupt_entry_74
        .quad interrupt_entry_75
        .quad interrupt_entry_76
        .quad interrupt_entry_77
        .quad interrupt_entry_78
        .quad interrupt_entry_79
        .quad interrupt_entry_80
        .quad interrupt_entry_81
        .quad interrupt_entry_82
        .quad interrupt_entry_83
        .quad interrupt_entry_84
        .quad interrupt_entry_85
        .quad interrupt_entry_86
        .quad interrupt_entry_87
        .quad interrupt_entry_88
        .quad interrupt_entry_89
        .quad interrupt_entry_90
        .quad interrupt_entry_91
        .quad interrupt_entry_92
        .quad interrupt_entry_93
        .quad interrupt_entry_94
        .quad interrupt_entry_95
        .quad interrupt_entry_96
        .quad interrupt_entry_97
        .quad interrupt_entry_98
        .quad interrupt_entry_99
        .quad interrupt_entry_100
        .quad interrupt_entry_101
        .quad interrupt_entry_102
        .quad interrupt_entry_103
        .quad interrupt_entry_104
        .quad interrupt_entry_105
        .quad interrupt_entry_106
        .quad interrupt_entry_107
        .quad interrupt_entry_108
        .quad interrupt_entry_109
        .quad interrupt_entry_110
        .quad interrupt_entry_111
        .quad interrupt_entry_112
        .quad interrupt_entry_113
        .quad interrupt_entry_114
        .quad interrupt_entry_115
        .quad interrupt_entry_116
        .quad interrupt_entry_117
        .quad interrupt_entry_118
        .quad interrupt_entry_119
        .quad interrupt_entry_120
        .quad interrupt_entry_121
        .quad interrupt_entry_122
        .quad interrupt_entry_123
        .quad interrupt_entry_124
        .quad interrupt_entry_125
        .quad interrupt_entry_126
        .quad interrupt_entry_127
        .quad interrupt_entry_128
        .quad interrupt_entry_129
        .quad interrupt_entry_130
        .quad interrupt_entry_131
        .quad interrupt_entry_132
        .quad interrupt_entry_133
        .quad interrupt_entry_134
        .quad interrupt_entry_135
        .quad interrupt_entry_136
        .quad interrupt_entry_137
        .quad interrupt_entry_138
        .quad interrupt_entry_139
        .quad interrupt_entry_140
        .quad interrupt_entry_141
        .quad interrupt_entry_142
        .quad interrupt_entry_143
        .quad interrupt_entry_144
        .quad interrupt_entry_145
        .quad interrupt_entry_146
        .quad interrupt_entry_147
        .quad interrupt_entry_148
        .quad interrupt_entry_149
        .quad interrupt_entry_150
        .quad interrupt_entry_151
        .quad interrupt_entry_152
        .quad interrupt_entry_153
        .quad interrupt_entry_154
        .quad interrupt_entry_155
        .quad interrupt_entry_156
        .quad interrupt_entry_157
        .quad interrupt_entry_158
        .quad interrupt_entry_159
        .quad interrupt_entry_160
        .quad interrupt_entry_161
        .quad interrupt_entry_162
        .quad interrupt_entry_163
        .quad interrupt_entry_164
        .quad interrupt_entry_165
        .quad interrupt_entry_166
        .quad interrupt_entry_167
        .quad interrupt_entry_168
        .quad interrupt_entry_169
        .quad interrupt_entry_170
        .quad interrupt_entry_171
        .quad interrupt_entry_172
        .quad interrupt_entry_173
        .quad interrupt_entry_174
        .quad interrupt_entry_175
        .quad interrupt_entry_176
        .quad interrupt_entry_177
        .quad interrupt_entry_178
        .quad interrupt_entry_179
        .quad interrupt_entry_180
        .quad interrupt_entry_181
        .quad interrupt_entry_182
        .quad interrupt_entry_183
        .quad interrupt_entry_184
        .quad interrupt_entry_185
        .quad interrupt_entry_186
        .quad interrupt_entry_187
        .quad interrupt_entry_188
        .quad interrupt_entry_189
        .quad interrupt_entry_190
        .quad interrupt_entry_191
        .quad interrupt_entry_192
        .quad interrupt_entry_193
        .quad interrupt_entry_194
        .quad interrupt_entry_195
        .quad interrupt_entry_196
        .quad interrupt_entry_197
        .quad interrupt_entry_198
        .quad interrupt_entry_199
        .quad interrupt_entry_200
        .quad interrupt_entry_201
        .quad interrupt_entry_202
        .quad interrupt_entry_203
        .quad interrupt_entry_204
        .quad interrupt_entry_205
        .quad interrupt_entry_206
        .quad interrupt_entry_207
        .quad interrupt_entry_208
        .quad interrupt_entry_209
        .quad interrupt_entry_210
        .quad interrupt_entry_211
        .quad interrupt_entry_212
        .quad interrupt_entry_213
        .quad interrupt_entry_214
        .quad interrupt_entry_215
        .quad interrupt_entry_216
        .quad interrupt_entry_217
        .quad interrupt_entry_218
        .quad interrupt_entry_219
        .quad interrupt_entry_220
        .quad interrupt_entry_221
        .quad interrupt_entry_222
        .quad interrupt_entry_223
        .quad interrupt_entry_224
        .quad interrupt_entry_225
        .quad interrupt_entry_226
        .quad interrupt_entry_227
        .quad interrupt_entry_228
        .quad interrupt_entry_229
        .quad interrupt_entry_230
        .quad interrupt_entry_231
        .quad interrupt_entry_232
        .quad interrupt_entry_233
        .quad interrupt_entry_234
        .quad interrupt_entry_235
        .quad interrupt_entry_236
        .quad interrupt_entry_237
        .quad interrupt_entry_238
        .quad interrupt_entry_239
        .quad interrupt_entry_240
        .quad interrupt_entry_241
        .quad interrupt_entry_242
        .quad interrupt_entry_243
        .quad interrupt_entry_244
        .quad interrupt_entry_245
        .quad interrupt_entry_246
        .quad interrupt_entry_247
        .quad interrupt_entry_248
        .quad interrupt_entry_249
        .quad interrupt_entry_250
        .quad interrupt_entry_251
        .quad interrupt_entry_252
        .quad interrupt_entry_253
        .quad interrupt_entry_254
        .quad interrupt_entry_255
