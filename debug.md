# Debugging 
- Increased number of allowed open files to fix error "OS Too many open files" via `ulimit -n 5000`
- RuntimeError: CUDA out of memory. Tried to allocate 12.00 MiB (GPU 0; 1.95 GiB total capacity; 1023.93 MiB already allocated; 9.94 MiB free; 1.11 GiB reserved in total by PyTorch)
