#pragma once

const int N_THREAD_GPU = 1024 / 2;
const int N_WALK_LIMIT = N_THREAD_GPU * 32;
const int NI_LIMIT     = 128 * N_WALK_LIMIT;
const int NJ_LIMIT     = 16 * NI_LIMIT;

