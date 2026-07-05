/*
MIT License

Copyright (c) 2026 Magnus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
 * Embedded MNUE P2/P2Pro network — compiled as a standalone object to avoid any
 * interaction with LTO / PGO instrumentation on the C++ translation units.
 *
 * Provides two global symbols:
 *   mnue_p2_embedded_data  — start of the raw .MNUE file
 *   mnue_p2_embedded_end   — one past the last byte
 */

    .section .rodata
    .global mnue_p2_embedded_data
    .global mnue_p2_embedded_end
    .balign 64

mnue_p2_embedded_data:
    .incbin MNUE_EMBEDDED_PATH

mnue_p2_embedded_end:
