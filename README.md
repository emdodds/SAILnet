# SAILnet
Python implementation of Zylberberg et al.'s SAILnet sparse coding algorithm, extended to support wider variety of inputs.

SAILnet.py implements the algorithm described in Zylberberg, Murphy & DeWeese (2011) "A sparse coding model with synaptically
local plasticity and spiking neurons can account for the diverse shapes of V1
simple cell receptive fields", PLoS Computational Biology 7(10).
and that paper should be cited by any work stemming from use of SAILnet. Zylberberg's original Matlab code can be found here: http://faculty.washington.edu/joelzy/sailcodes.html
along with the whitened natural image patches used in Zylberberg, Murphy & DeWeese (as well as other studies of sparse coding of natural images).

To imitate the behavior of Zylberberg's Matlab code, simple import SAILnet, create a SAILnet object with the default parameters and call the instance's .run() method.
