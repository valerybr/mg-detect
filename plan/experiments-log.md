
# Experiments log - Step 1

1. No lesion in MG - MG without findings
2. CC view only


## cut_bilateral_exp2_0416

* Flip right-side view to have the same orientation as left side. 
* Use 50 epochs for constant learning rate, then 50 epoch linear decay
* Image size: W384, H512

### Results 
[wandb](https://wandb.ai/valery-brodsky-ariel-university/mg-detect-cut-bilateral/runs/cut_bilateral_exp2_0416)
The model trains successfull. \
**Conclusion**: Use this size for future training




## cut_bilateral_exp2_0416

* Right side and left side have different orientations. 
* Use 100 epochs for constant learning rate, then 100 epoch linear decay
* Image size: W384, H512

### Results

[wandb](https://wandb.ai/valery-brodsky-ariel-university/mg-detect-cut-bilateral/runs/cut_bilateral_exp1_0416?nw=nwuservalerybrodsky)


* Discriminator collapsed at epoch 12 = 0.018. Possible reason - easy to catch orientation. 
* Size reduced time to train one epoch to 35min on L4 GPU