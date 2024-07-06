//#pragma OPENCL EXTENSION cl_intel_channels : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

channel int c0 __attribute__((depth(8))); // Example for setting buffer depth, adjust syntax based on your OpenCL extension

__kernel void producer() {
    for (int i = 0; i < 8; i++) {
        printf("producer %d\n", i);
        write_channel_intel(c0, i);
    }
}

__kernel void consumer (__global double * restrict dst) {
    for (int i = 0; i < 8; i++) {
        printf("consumer %d\n", i);
        //dst[i] = read_channel_intel(c0);
        read_channel_intel(c0);
    }
}


channel int foo_bar_channel __attribute__((depth(1024)));
channel float bar_baz_channel __attribute__((depth(1024)));

__kernel void foo() {
  for (int i = 0; i < 1024; ++i) {
    int value = i;
    value = clamp(value, 0, 255);                 // do some work
    write_channel_intel(foo_bar_channel, value); // send data to the next kernel
  }
}

__kernel void bar() {
  for (int i = 0; i < 1024; ++i) {
    int value = read_channel_intel(foo_bar_channel); // take data from foo
    float fvalue = (float) value;
    write_channel_intel(bar_baz_channel, value); // send data to the next kernel
  }
}

__kernel void baz() {
  for (int i = 0; i < 1024; ++i) {
    float value = read_channel_intel(bar_baz_channel);
  }
}