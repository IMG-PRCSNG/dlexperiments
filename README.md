# dlexperiments
A bunch of notebooks and scripts experimenting with Caffe, TF, Keras, PyTorch etc.

## Update [30-Sep-2017]

#### Summary
 - Added PyCrayon, Now you can create a client an experiment and pass it to the Experiment class - https://github.com/torrvision/crayon
 - Re-wrote the experiment class and separated model, dataset from it and left the control to user
 - Added a notebook walking through how it can be used.


## Update [28-Sep-2017]

#### Summary
 - My new found Love - PyTorch!
 - Included script to train/finetune any of the Imagenet model in PyTorch Model Zoo with any Image Dataset

#### ToDo
 - ~~Integrate with pycrayon and check logging on TensorBoard~~
 - ~~Fix the pretrain - freeze - unfreeze - classes dependency hell~~
 - ~~Create a Tutorial Notebook and include Citations~~
 - ~~Write Documentation for the arguments for the script~~
 - Maybe we can create factory methods for the optimizer, scheduler, criterion so that  
   the script can become declarative and configurable with just arguments.
