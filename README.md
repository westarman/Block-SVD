# Block-SVD
Block-wise SVD image compression


## Questions & Plans...
- do we use RGB or just grayscale images? should we maybe try out how SVD performs on different image representations such as HSL,HSV and YCrCb?
- whats our SVD strategy when dealing with multiple channels (matrices), do we block SVD each matrix individually? do we append them to form a singular vertical/horizontal matrix and then apply SVD? what about *Tensor SVD* (thats probbly too complicated...)?
- how do we compute compression error when comparing results? MSE,RMSE maybe MAE error or maybe BPP (bits per pixel) for performance? whole picture-wise or per pixel-wise?
- we need some ideas to increase quality/compression ratio (preprocessing with algorithms mybi?)
- how are we gonna store these images with their matrices and singular values? bitwise encodings perhaps (Huffman,Arithmetic...)? <-- not sure if this is what the question from the task is asking for ("Kako bi shranili tako stisnjeno sliko?")
- memory efficiency of SVD matrices of an image need to be compared to original image memory (is this what the previous question pertains?) <-- probbly a  balance between **quality retention <-> memory efficiency**


## TODO
- ~~test~out normal SVD img compression~~ **DONE**
- make a simple `k x k` block-wise SVD compression with adaptable rank for singular values
- compare some results for different `k` sizes as well as different number of singular values saved (from 1 to `k/2` number of singular values)
- add the ability to adapt the singular value rank on individual blocks, based on `q` - quality  (also compare results for different `k` and `q` )