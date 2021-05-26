# CSE272 HW2 Recommendation System 

Author: Anthony Liu ([e-mail](mailto:hliu35@ucsc.edu))

Date: 05/20/2021

----

## ! READ BEFORE RUNNING !
The program uses Jupyter Notebook to run. It is completely built from scratch without commercial level optimizations. 

Therefore, due to the scale of the default dataset ([Amazon video game reviews](http://jmcauley.ucsd.edu/data/amazon/)), running the program may require expansive memory space (ideally >= 32GB to run without any issue). For the same reason, they are divided into 4 subsequent notebooks.

Furthermore, to free up memory spaces, it is advised that you do not run a new notebook before closing the previous running ones. 

----

## Downloading the dataset
Visit this [link](http://jmcauley.ucsd.edu/data/amazon/) and find "Video Games" > "5-cores".

----

## Running the program
The notebooks were written in the order of HW questions. As such, subsequent notebooks uses the saved results (`*.pickle` files) from previous ones. Please run them in the order of names `part1` > `part2a` > `part2b` > `part3`.

### Note for Matrix Factorization
Because of memory usage, the default device for training models is `device='cpu'`. To change the device to GPU, you can manually set `device='cuda'` in part2b.