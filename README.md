# HPC_assignment

Laboratorio del corso di High Performance Computing, laurea magistrale in Informatica, Università di Modena e Reggio Emilia. Progetto svolto da:

Giovanni Malaguti   - 268684

Lorenzo Racca       - 269760

Giovanni Rinaldi    - 256106


Per eseguire i file, copiare la versione desiderata in reg_detect.c, ad esempio:

> cat reg_detect_GPUv3.3.c > reg_detect.c

E successivamente eseguire il comando make con le opzioni desiderate, ad esempio:

> make EXT_CFLAGS="-DPOLYBENCH_TIME -DLARGE_DATASET" clean all run

