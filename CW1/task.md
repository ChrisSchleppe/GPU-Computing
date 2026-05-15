# Coursework 1: Parallelizing a basic linear algebra C++ program

Your assignment for this lab class is to transform the given C++ into an equivalent CUDA program.
Specifically, all Matrix/Vector computations should take place in parallel on the GPU.

1) Run it, benchmark it and take a screenshot of the runtime of your first attempt.

2) Then, play with the kernel launch configuration (thread count, block count, etc.) to see how it impacts performance and take a screenshot of your best performing configuration.

Finally, hand in a ZIP containing your screenshots, CUDA code and Make file to build it, and the launch configurations you tried either in the form of comments in the CUDA code or as a standalone Markdown file.
