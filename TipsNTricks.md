##Some progamming tips and tricks which I happen to know :)
* ####Checking membership frequntly:
  
  use sets instead of lists. It does speed up a lot. Sets are made for this. Oh come on, just use sets.
* ####Theano with GPU:
  
  when I started using Theano with GPU, I was disappointed as I was not getting the expected speed up.
But I soon realized the mistakes I have been making. I have listed some of them here.
  + **Scan** : Avoid making use of `theano.scan` for looping. If can't be avoided, then write code such that you can scan along smaller dimension.
  For example, if your input matrix if of size `1000000x200`, then looping along `2nd dimension (200)` will be much faster than the first dimension.
  If both dimensions are high, God bless you :) .
  + **Use GPU memory** : Whenever possible, make use of GPU memory. There are certain inputs which are neither parameters nor training/test/valid data, 
  but part of the computation graph. If they are not stored in GPU memory, it will make the code run very slow.
  So, make sure all inputs to the computation graph are stored in GPU memory (by making them theano shared variables or something).
